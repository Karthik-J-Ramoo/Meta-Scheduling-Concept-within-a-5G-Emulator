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


# ── load audit (T4) ───────────────────────────────────────────────────

class OverloadError(RuntimeError):
    """The workload cannot be served by the PRB budget on average."""


def _phase_spans(traffic: dict, num_steps: int) -> list[tuple[int, float]]:
    """``[(length, multiplier), ...]`` covering ``[0, num_steps)``."""
    phases = sorted(traffic.get("phases", [])) or [[0, 1.0]]
    if phases[0][0] > 0:
        phases = [[0, 1.0]] + phases
    bounds = [p[0] for p in phases[1:]] + [num_steps]
    return [(end - start, m) for (start, m), end in zip(phases, bounds) if end > start]


def prb_demand(cfg: dict) -> dict:
    """
    Mean PRB demand per slice, ``lam·(1 − p_b + p_b·m)/r``, time-averaged
    over phases; also the peak phase demand.  Returns
    ``{"mean": {slice: x}, "peak": {slice: x}, "mean_sum": x, "peak_sum": x}``.
    """
    n = cfg["simulation"]["num_timesteps"]
    mean, peak = {}, {}
    for c in cfg["slices"]:
        tp, r = c["traffic_params"], c["service_rate"]
        base = tp["mean_arrival"] * (1 - tp["burst_prob"] + tp["burst_prob"] * tp["burst_mult"]) / r
        spans = _phase_spans(tp, n)
        mean[c["name"]] = base * sum(length * m for length, m in spans) / n
        peak[c["name"]] = base * max(m for _, m in spans)
    return {"mean": mean, "peak": peak,
            "mean_sum": sum(mean.values()), "peak_sum": sum(peak.values())}


def saturation(history: Sequence[Sequence[SliceMetrics]]) -> Dict[str, float]:
    """
    Fraction of steps each slice's queue hit capacity on enqueue (i.e. it
    dropped packets).  The post-service queue never equals capacity when
    anything is served, so ``queue_length >= capacity`` would miss it.
    """
    n = len(history)
    names = [m.slice_name for m in history[0]]
    return {name: sum(1 for step in history for m in step
                      if m.slice_name == name and m.dropped > 0) / n
            for name in names}


def audit_load(cfg: dict, history: Sequence[Sequence[SliceMetrics]],
               max_saturation: float = 0.20) -> dict:
    """Raise ``OverloadError`` if mean demand > budget or a queue saturates."""
    demand = prb_demand(cfg)
    sat = saturation(history)
    report = {**demand, "saturation": sat}
    problems = []
    if demand["mean_sum"] > cfg["simulation"]["total_prbs"]:
        problems.append(f"mean PRB demand {demand['mean_sum']:.1f} > "
                        f"{cfg['simulation']['total_prbs']}")
    problems += [f"{name} at capacity {frac:.0%} of steps"
                 for name, frac in sat.items() if frac > max_saturation]
    if problems:
        raise OverloadError(f"{cfg.get('name', '?')}: " + "; ".join(problems))
    return report
