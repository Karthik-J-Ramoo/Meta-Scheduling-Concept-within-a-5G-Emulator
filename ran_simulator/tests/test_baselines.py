"""
T2 – fixed-split sweep on the seed-42 paper trace, and the per-slice
decomposition used by every static baseline.
"""

import pytest

from config.experiments import get_config
from core.baselines import best_split, grid_evaluate, split_grid
from core.episode_metrics import violation_events
from core.simulation import arrival_traces, run_episode
from test_policies import _recorded_traces

GRID = split_grid()


def test_grid_shape():
    assert all(sum(s) == 100 and s[2] >= 5 for s in GRID)
    assert (40, 40, 20) not in GRID   # 40, 45, 46, 49 are off the 20..70 step-3 lattice
    assert len(GRID) == len(set(GRID)) == 184


@pytest.mark.parametrize("split", [(40, 40, 20), (46, 44, 10), (50, 40, 10), (20, 20, 60)])
def test_decomposition_matches_full_run(split):
    cfg = get_config("csv_repro")
    traces = arrival_traces(cfg, 400)
    full = run_episode(cfg, policy="static", static_split=split, traces=traces)
    viol = sum(len(v) for v in violation_events(full).values())
    drops = sum(m.dropped for step in full for m in step)
    assert grid_evaluate(cfg, traces, [split])[split] == (viol, drops)


def test_fixed_splits_on_paper_trace():
    """The paper's 57.14 % gain is (7 - 3) / 7 against the 40/40/20 split."""
    cfg, traces = get_config("csv_original"), _recorded_traces()
    splits = [(46, 44, 10), (49, 42, 9), (45, 40, 15), (50, 40, 10), (40, 40, 20)]
    ev = grid_evaluate(cfg, traces, splits)
    assert [ev[s][0] for s in splits] == [0, 0, 0, 0, 7]
    grid_ev = grid_evaluate(cfg, traces, GRID)
    assert grid_ev[best_split([grid_ev])][0] == 0
