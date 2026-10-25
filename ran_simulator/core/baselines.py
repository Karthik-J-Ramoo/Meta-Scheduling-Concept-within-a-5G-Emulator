"""
baselines.py
============
Fixed-split (static) baselines: grid evaluation, hindsight-best and tuned.

Under a static policy every slice evolves independently – its queue, SLA
and drops depend only on its own arrivals and PRBs.  So a split
``(u, e, i)`` is scored by running each slice alone at its PRB count through
the normal episode loop (``run_episode``) and summing; each (slice, PRBs)
pair is simulated once per trace.  ``tests/test_baselines.py`` checks the
decomposition against full three-slice runs.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Sequence, Tuple

from core.episode_metrics import violation_events
from core.simulation import run_episode

Split = Tuple[int, ...]


def split_grid(total: int = 100, lo: int = 20, hi: int = 70, step: int = 3,
               min_last: int = 5) -> List[Split]:
    """All (u, e, total-u-e) with u, e in range(lo, hi+1, step), last >= min_last."""
    return [(u, e, total - u - e)
            for u in range(lo, hi + 1, step)
            for e in range(lo, hi + 1, step)
            if total - u - e >= min_last]


def _single_slice_cfg(cfg: dict, idx: int) -> dict:
    single = copy.deepcopy(cfg)
    single["slices"] = [single["slices"][idx]]
    return single


def slice_scores(cfg: dict, traces: Sequence[Sequence[int]], grid: Sequence[Split]
                 ) -> List[Dict[int, Tuple[int, int]]]:
    """``[slice][prbs] -> (violations, drops)`` for every PRB value in *grid*."""
    scores: List[Dict[int, Tuple[int, int]]] = []
    for idx, trace in enumerate(traces):
        single = _single_slice_cfg(cfg, idx)
        table = {}
        for prbs in sorted({split[idx] for split in grid}):
            hist = run_episode(single, policy="static", static_split=[prbs],
                               traces=[trace], num_steps=len(trace))
            viol = sum(len(v) for v in violation_events(hist).values())
            drops = sum(m.dropped for step in hist for m in step)
            table[prbs] = (viol, drops)
        scores.append(table)
    return scores


def grid_evaluate(cfg: dict, traces: Sequence[Sequence[int]], grid: Sequence[Split]
                  ) -> Dict[Split, Tuple[int, int]]:
    """``split -> (total violations, total drops)`` on one trace set."""
    scores = slice_scores(cfg, traces, grid)
    return {split: (sum(scores[k][p][0] for k, p in enumerate(split)),
                    sum(scores[k][p][1] for k, p in enumerate(split)))
            for split in grid}


def best_split(evaluations: Sequence[Dict[Split, Tuple[int, int]]]) -> Split:
    """
    Split with the fewest mean violations over *evaluations* (one dict per
    trace); ties broken by mean drops, then by the split itself.
    """
    grid = list(evaluations[0])
    def key(split):
        return (sum(ev[split][0] for ev in evaluations),
                sum(ev[split][1] for ev in evaluations),
                split)
    return min(grid, key=key)


# ── static-safe reservations ─────────────────────────────────────────

def static_safe_reservation(slice_cfg: dict, sla_mode: str) -> int | None:
    """
    Smallest PRB count at which the slice's SLA can never be violated, for
    *any* arrival sequence; ``None`` if no such count exists (the SLA's
    requirement depends on load).

    * latency L, capacity C: the backlog after service is at most
      ``C − floor(P·r)``, so P is safe iff ``(C − floor(P·r)) / (P·r) ≤ L``.
    * throughput T, demand_aware: safe iff ``floor(P·r) ≥ T`` (and C ≥ T).
    * throughput T, legacy: violated whenever fewer than T packets are
      offered, whatever P is → None.
    * relaxed: 0.
    * any ``sla_drop_cap``: overflow depends on load → None.
    """
    if slice_cfg.get("sla_drop_cap") is not None:
        return None
    r, cap = slice_cfg["service_rate"], slice_cfg["queue_capacity"]
    thr, kind = slice_cfg["sla_threshold"], slice_cfg["sla_type"]
    if kind == "relaxed":
        return 0
    if kind == "throughput" and sla_mode == "legacy":
        return None
    for prbs in range(1, 10_000):
        rate = math.floor(prbs * r)
        if kind == "latency" and max(0, cap - rate) / (prbs * r) <= thr:
            return prbs
        if kind == "throughput" and rate >= thr:
            return prbs
    return None


def phase_requirements(cfg: dict) -> List[dict]:
    """
    Per phase (union of all slices' phase boundaries), the PRBs each slice
    needs on average to meet its SLA: its mean PRB demand
    ``lam·(1 − p_b + p_b·m)/r``, capped at its static-safe reservation when
    one exists (relaxed slices need 0).  Returns
    ``[{"start", "end", "req": {slice: prbs}, "sum"}]``.
    """
    n = cfg["simulation"]["num_timesteps"]
    bounds = sorted({0, n} | {p[0] for c in cfg["slices"]
                              for p in c["traffic_params"].get("phases", []) if p[0] < n})
    out = []
    for start, end in zip(bounds, bounds[1:]):
        req = {}
        for c in cfg["slices"]:
            tp = c["traffic_params"]
            mult = 1.0
            for s, m in sorted(tp.get("phases", [])):
                if start >= s:
                    mult = m
            demand = (tp["mean_arrival"] * mult
                      * (1 - tp["burst_prob"] + tp["burst_prob"] * tp["burst_mult"])
                      / c["service_rate"])
            reserve = static_safe_reservation(c, cfg["sla_mode"])
            if c["sla_type"] == "relaxed" and c.get("sla_drop_cap") is None:
                demand = 0.0
            req[c["name"]] = demand if reserve is None else min(demand, reserve)
        out.append({"start": start, "end": end, "req": req, "sum": sum(req.values())})
    return out
