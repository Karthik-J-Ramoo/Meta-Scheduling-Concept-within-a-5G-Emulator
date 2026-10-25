"""
simulation.py
=============
One closed-loop episode, shared by ``main.py``, the tests and the experiment
scripts so that every number in the paper comes from the same code path.

Per timestep *t* (see ``main.py`` for the narrative):
    traffic → enqueue → DU scheduling → snapshot → SLA check → record →
    meta-scheduling (policy "meta") for t+1.

Policy "static" skips the meta-scheduler, so allocations stay at the initial
shares (or at ``static_split`` when given) for the whole episode.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from core.du_scheduler import DUScheduler
from core.meta_scheduler import MetaScheduler
from core.performance_monitor import PerformanceMonitor
from core.sla_checker import SLAChecker
from core.slice import NetworkSlice, SliceMetrics
from core.traffic_generator import TraceTraffic, TrafficGenerator

POLICIES = ("static", "meta")


def build_slices(cfg: dict, static_split: Optional[Sequence[int]] = None) -> list[NetworkSlice]:
    """Create slices from *cfg*; *static_split* overrides the initial PRBs."""
    slices = [NetworkSlice(c) for c in cfg["slices"]]
    if static_split is not None:
        for s, prbs in zip(slices, static_split):
            s.allocated_prbs = int(prbs)
    return slices


def build_traffic(cfg: dict, slices: Sequence[NetworkSlice], seed: int | None) -> list[TrafficGenerator]:
    """
    One generator per slice.  ``rng_mode`` "shared" reproduces the legacy
    draw order (one RNG, slices interleaved); "per_slice" gives slice *i*
    the stream ``SeedSequence(seed).spawn(n)[i]``.
    """
    mode = cfg.get("rng_mode", "shared")
    if mode == "shared":
        rng = np.random.default_rng(seed)
        return [TrafficGenerator(s, rng) for s in slices]
    if mode == "per_slice":
        streams = np.random.SeedSequence(seed).spawn(len(slices))
        return [TrafficGenerator(s, np.random.default_rng(ss)) for s, ss in zip(slices, streams)]
    raise ValueError(f"unknown rng_mode {mode!r}")


def arrival_traces(cfg: dict, seed: int, num_steps: int | None = None) -> list[list[int]]:
    """Per-slice arrival sequences for *seed* (``rng_mode`` "per_slice" only)."""
    if cfg.get("rng_mode") != "per_slice":
        raise ValueError("arrival traces are policy-independent only with rng_mode='per_slice'")
    num_steps = cfg["simulation"]["num_timesteps"] if num_steps is None else num_steps
    gens = build_traffic(cfg, build_slices(cfg), seed)
    return [[g.generate(t) for t in range(num_steps)] for g in gens]


def build_meta_scheduler(cfg: dict) -> MetaScheduler:
    meta = cfg["meta"]
    return MetaScheduler(
        total_prbs=cfg["simulation"]["total_prbs"],
        reallocation_step=meta["reallocation_step"],
        fairness_decay=meta["fairness_decay"],
        min_prbs=meta["min_prbs_per_slice"],
    )


def run_episode(
    cfg: dict,
    seed: int | None = None,
    policy: str = "meta",
    num_steps: int | None = None,
    static_split: Optional[Sequence[int]] = None,
    monitor: Optional[PerformanceMonitor] = None,
    traces: Optional[Sequence[Sequence[int]]] = None,
) -> list[list[SliceMetrics]]:
    """
    Run one episode and return the SLA-stamped snapshots, ``[t][slice]``.

    *seed* defaults to ``cfg["simulation"]["random_seed"]``; *num_steps* to
    ``cfg["simulation"]["num_timesteps"]``.  *traces* (one arrival sequence
    per slice) replaces the stochastic generators.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    sim = cfg["simulation"]
    seed = sim["random_seed"] if seed is None else seed
    num_steps = sim["num_timesteps"] if num_steps is None else num_steps

    slices = build_slices(cfg, static_split)
    if traces is not None:
        traffic = [TraceTraffic(tr) for tr in traces]
    else:
        traffic = build_traffic(cfg, slices, seed)
    du = DUScheduler()
    checker = SLAChecker()
    meta = build_meta_scheduler(cfg) if policy == "meta" else None

    history: list[list[SliceMetrics]] = []
    for t in range(num_steps):
        for gen, s in zip(traffic, slices):
            s.enqueue(gen.generate(timestep=t))
        du.schedule(slices, timestep=t)
        snapshots = [s.snapshot(t) for s in slices]
        violations, stamped = checker.check_all(slices, snapshots, t)
        history.append(stamped)
        if monitor is not None:
            monitor.record(stamped)
        if meta is not None:
            meta.allocate(slices, violations, timestep=t)
    return history
