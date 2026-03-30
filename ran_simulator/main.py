"""
main.py
=======
Entry point for the RAN meta-scheduling simulator.

Execution flow (one episode)
----------------------------
For each discrete timestep *t* in ``[0, num_timesteps)``:

    1. **Traffic generation** – each slice's ``TrafficGenerator`` samples
       packet arrivals and enqueues them.
    2. **DU scheduling** – ``DUScheduler`` drains each slice's queue up to
       the capacity implied by its current PRB allocation.
    3. **Metric snapshot** – each slice produces an immutable ``SliceMetrics``.
    4. **SLA evaluation** – ``SLAChecker`` tests every snapshot against its
       slice's SLA constraint.
    5. **Performance recording** – ``PerformanceMonitor`` logs the stamped
       metrics and streams them to CSV.
    6. **Meta-scheduling** – ``MetaScheduler`` reads the violation map and
       adjusts PRB allocations for the *next* timestep.

This is a **closed-loop control system**: SLA violations feed back into
resource allocation, which changes future queue dynamics.

Usage
-----
    cd ran_simulator
    python main.py                   # run with default config
    python main.py --steps 500       # override timestep count
    python main.py --seed 7          # override random seed
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── project imports ───────────────────────────────────────────────────
from config.simulation_config import SIMULATION, SLICES, LOGGING
from core.slice import NetworkSlice, SliceMetrics
from core.traffic_generator import TrafficGenerator
from core.du_scheduler import DUScheduler
from core.meta_scheduler import MetaScheduler
from core.sla_checker import SLAChecker
from core.performance_monitor import PerformanceMonitor, CSV_FIELDS
from utils.logger import setup_logger, MetricCSVWriter


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAN meta-scheduling simulator",
    )
    p.add_argument(
        "--steps", type=int, default=None,
        help="Number of simulation timesteps (overrides config)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config; None → non-deterministic)",
    )
    return p.parse_args()


# ── simulation ────────────────────────────────────────────────────────

def run_simulation(num_steps: int, seed: int | None) -> None:
    """
    Execute a full simulation episode.

    Parameters
    ----------
    num_steps : int
        Number of discrete timesteps.
    seed : int | None
        Seed for the numpy RNG.  ``None`` → non-deterministic.
    """
    log = logging.getLogger("main")

    # ── reproducibility ───────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    log.info("Simulation starts │ steps=%d  seed=%s", num_steps, seed)

    # ── build slices ──────────────────────────────────────────────────
    slices: list[NetworkSlice] = [NetworkSlice(cfg) for cfg in SLICES]

    # ── build components ──────────────────────────────────────────────
    traffic_gens: list[TrafficGenerator] = [
        TrafficGenerator(s, rng) for s in slices
    ]
    du_scheduler = DUScheduler()
    meta_scheduler = MetaScheduler()
    sla_checker = SLAChecker()

    csv_writer = MetricCSVWriter(fieldnames=CSV_FIELDS)
    monitor = PerformanceMonitor(csv_writer=csv_writer, window=20)

    log.info(
        "Components ready │ slices=%s",
        [s.name for s in slices],
    )

    # ── main simulation loop ──────────────────────────────────────────
    wall_start = time.perf_counter()

    for t in range(num_steps):

        # 1. Traffic generation → enqueue
        for gen, s in zip(traffic_gens, slices):
            arrivals = gen.generate(timestep=t)
            s.enqueue(arrivals)

        # 2. DU scheduling → dequeue + compute metrics
        du_scheduler.schedule(slices, timestep=t)

        # 3. Metric snapshots
        snapshots: list[SliceMetrics] = [s.snapshot(t) for s in slices]

        # 4. SLA evaluation (stamps snapshots with sla_satisfied)
        violations, stamped = sla_checker.check_all(slices, snapshots, t)

        # 5. Record to monitor + CSV
        monitor.record(stamped)

        # 6. Meta-scheduler adjusts PRBs for the NEXT timestep
        meta_scheduler.allocate(slices, violations, timestep=t)

    # ── wrap-up ───────────────────────────────────────────────────────
    wall_elapsed = time.perf_counter() - wall_start
    monitor.flush()
    monitor.close()

    log.info("─" * 72)
    log.info("Simulation complete │ %.2f s wall-clock", wall_elapsed)
    log.info("─" * 72)

    # Pretty-print final summary
    summary = monitor.summary()
    for name, stats in summary.items():
        log.info(
            "%-6s │ avg_lat=%6.2f  avg_tput=%6.1f  avg_q=%6.1f  "
            "drops=%d  sla_violations=%d/%d",
            name,
            stats["avg_latency"],
            stats["avg_throughput"],
            stats["avg_queue"],
            stats["total_dropped"],
            stats["sla_violations"],
            stats["window_size"],
        )

    log.info(
        "Results written to %s/ (CSV + log)",
        Path(LOGGING["log_dir"]).resolve(),
    )


# ── entry point ───────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    num_steps = args.steps or SIMULATION["num_timesteps"]
    seed = args.seed if args.seed is not None else SIMULATION["random_seed"]

    # Logging must be set up before any module-level logger is used.
    setup_logger()

    run_simulation(num_steps=num_steps, seed=seed)


if __name__ == "__main__":
    main()
