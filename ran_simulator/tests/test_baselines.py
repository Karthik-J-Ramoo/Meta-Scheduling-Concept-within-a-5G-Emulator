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


# ── static-safe reservation theorem ──────────────────────────────────

from core.baselines import phase_requirements, static_safe_reservation  # noqa: E402


def test_reservations_for_paper_slas():
    cfg = get_config("feasible")
    r = [static_safe_reservation(c, cfg["sla_mode"]) for c in cfg["slices"]]
    assert r == [45, 38, 0]
    urllc = cfg["slices"][0]
    assert (200 - 66) / 66 > 2.0       # P = 44: worst-case latency 2.03 ms
    # legacy throughput SLA has no safe reservation
    assert static_safe_reservation(get_config("csv_fitted")["slices"][1], "legacy") is None
    # IoT minimum-service 12 pkts/ts at r = 0.8 needs 15 PRBs
    assert static_safe_reservation(get_config("t6_reliability")["slices"][2], "demand_aware") == 15
    assert static_safe_reservation(get_config("t6_reliability")["slices"][0], "demand_aware") is None
    assert urllc["queue_capacity"] == 200


def _random_traces(rng, n=200):
    """Arbitrary per-slice loads, from idle to far beyond capacity."""
    return [list(rng.poisson(rng.uniform(0, lam_max), size=n)) for lam_max in (300, 600, 200)]


@pytest.mark.parametrize("name", ["feasible", "regime", "anticorr"])
def test_split_at_reservation_never_violates(name):
    """Any split >= (45, 38, 5) is violation-free under ANY traffic."""
    import numpy as np
    cfg, rng = get_config(name), np.random.default_rng(7)
    for _ in range(30):
        traces = _random_traces(rng)
        ev = grid_evaluate(cfg, traces, [(45, 38, 17), (47, 38, 15), (44, 38, 18)])
        assert ev[(45, 38, 17)][0] == 0 and ev[(47, 38, 15)][0] == 0
    saturating = [[400] * 200, [0] * 200, [0] * 200]
    assert grid_evaluate(cfg, saturating, [(44, 38, 18)])[(44, 38, 18)][0] > 0


def test_tuned_split_is_smallest_lattice_point_above_reservation():
    # 47 = first lattice value >= 45, 38 = eMBB reservation, IoT takes the rest
    assert (47, 38, 15) in GRID and (44, 38, 18) in GRID


def test_t6_requirements_peak_above_budget_phases_fit():
    ph = phase_requirements(get_config("t6_reliability"))
    peak = {n: max(p["req"][n] for p in ph) for n in ph[0]["req"]}
    assert sum(peak.values()) > 100                 # no fixed split covers every peak
    assert max(p["sum"] for p in ph) <= 100          # but every phase fits
