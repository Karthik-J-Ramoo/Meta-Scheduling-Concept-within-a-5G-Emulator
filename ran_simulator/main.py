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
resource allocation, which changes future queue dynamics.  The loop itself
lives in ``core/simulation.py`` so tests and experiment scripts share it.

Usage
-----
    cd ran_simulator
    python main.py                   # run with default config
    python main.py --steps 500       # override timestep count
    python main.py --seed 7          # override random seed
    python main.py --config csv_repro --policy static
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── project imports ───────────────────────────────────────────────────
from config.experiments import CONFIGS, get_config
from config.simulation_config import LOGGING
from core.performance_monitor import PerformanceMonitor, CSV_FIELDS
from core.simulation import POLICIES, run_episode
from utils.logger import setup_logger, MetricCSVWriter


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAN meta-scheduling simulator",
    )
    p.add_argument(
        "--config", default="checked_in", choices=sorted(CONFIGS),
        help="Named experiment config (config/experiments.py)",
    )
    p.add_argument(
        "--policy", default=None, choices=POLICIES,
        help="PRB allocation policy: static initial shares or the meta-scheduler "
             "(default: the config's 'policy', else meta)",
    )
    p.add_argument(
        "--steps", type=int, default=None,
        help="Number of simulation timesteps (overrides config)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config)",
    )
    return p.parse_args()


# ── simulation ────────────────────────────────────────────────────────

def run_simulation(cfg: dict, policy: str, num_steps: int, seed: int | None) -> None:
    """
    Execute a full simulation episode; write CSV, log and the active config.

    Parameters
    ----------
    cfg : dict
        Experiment config from ``config.experiments.get_config``.
    policy : str
        ``"static"`` or ``"meta"``.
    num_steps : int
        Number of discrete timesteps.
    seed : int | None
        Seed for the numpy RNG.  ``None`` → non-deterministic.
    """
    log = logging.getLogger("main")
    log.info(
        "Simulation starts │ config=%s  policy=%s  steps=%d  seed=%s",
        cfg["name"], policy, num_steps, seed,
    )

    # Record exactly what produced this run next to its outputs.
    out_dir = Path(LOGGING["log_dir"])
    active = {**cfg, "run": {"policy": policy, "seed": seed, "num_steps": num_steps}}
    (out_dir / "active_config.json").write_text(json.dumps(active, indent=2), encoding="utf-8")

    csv_writer = MetricCSVWriter(fieldnames=CSV_FIELDS)
    monitor = PerformanceMonitor(csv_writer=csv_writer, window=num_steps)

    # ── main simulation loop ──────────────────────────────────────────
    wall_start = time.perf_counter()
    run_episode(cfg, seed=seed, policy=policy, num_steps=num_steps, monitor=monitor)
    wall_elapsed = time.perf_counter() - wall_start

    # ── wrap-up ───────────────────────────────────────────────────────
    monitor.flush()
    monitor.close()

    log.info("─" * 72)
    log.info("Simulation complete │ %.2f s wall-clock", wall_elapsed)
    log.info("─" * 72)

    # Pretty-print final summary (window = whole episode)
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
        "Results written to %s/ (CSV + log + active_config.json)",
        out_dir.resolve(),
    )


# ── entry point ───────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = get_config(args.config)
    num_steps = args.steps or cfg["simulation"]["num_timesteps"]
    seed = args.seed if args.seed is not None else cfg["simulation"]["random_seed"]

    # Logging must be set up before any module-level logger is used.
    setup_logger()

    policy = args.policy or cfg.get("policy", "meta")
    run_simulation(cfg, policy=policy, num_steps=num_steps, seed=seed)


if __name__ == "__main__":
    main()
