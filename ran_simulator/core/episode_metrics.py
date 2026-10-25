"""
episode_metrics.py
==================
Per-episode KPIs computed from the ``[t][slice]`` history returned by
``core.simulation.run_episode``.  Shared by tests and experiment scripts.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from core.slice import SliceMetrics


def violation_events(history: Sequence[Sequence[SliceMetrics]]) -> Dict[str, List[int]]:
    """``{slice_name: [timesteps with an SLA violation]}``."""
    events: Dict[str, List[int]] = {m.slice_name: [] for m in history[0]}
    for step in history:
        for m in step:
            if not m.sla_satisfied:
                events[m.slice_name].append(m.timestep)
    return events


def allocation_changes(history: Sequence[Sequence[SliceMetrics]]) -> int:
    """Number of timesteps whose PRB vector differs from the previous one."""
    allocs = [tuple(m.allocated_prbs for m in step) for step in history]
    return sum(1 for a, b in zip(allocs, allocs[1:]) if a != b)


def summarize(history: Sequence[Sequence[SliceMetrics]]) -> dict:
    """Flat KPI dict for one episode (one row of results/summary.csv)."""
    n = len(history)
    events = violation_events(history)
    row: dict = {f"viol_{name}": len(ts) for name, ts in events.items()}
    row["viol_total"] = sum(len(ts) for ts in events.values())
    for name in events:
        series = [m for step in history for m in step if m.slice_name == name]
        row[f"drops_{name}"] = sum(m.dropped for m in series)
        row[f"served_per_ts_{name}"] = sum(m.served for m in series) / n
        row[f"avg_latency_{name}"] = sum(m.latency for m in series) / n
    row["drops_total"] = sum(row[f"drops_{name}"] for name in events)
    row["prb_changes"] = allocation_changes(history)
    return row
