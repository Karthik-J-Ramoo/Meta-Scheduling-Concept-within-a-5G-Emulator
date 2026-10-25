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
