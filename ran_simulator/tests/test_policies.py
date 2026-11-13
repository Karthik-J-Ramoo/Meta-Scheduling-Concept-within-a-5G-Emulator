"""
T1 – static counterfactual.

With ``rng_mode='per_slice'`` every slice draws from its own stream, so the
arrival sequence is identical whichever policy runs.  The counterfactual on
the *recorded* paper trace is exact; the seed-42 per-slice run is a fresh
sample of the same workload and is reported, not pinned.
"""

import csv

import pytest

from conftest import DATA
from config.experiments import get_config
from core.episode_metrics import violation_events
from core.simulation import arrival_traces, run_episode


def _arrivals(history):
    return [[m.arrivals for m in step] for step in history]


def _recorded_traces(fname="metrics_csv_original.csv"):
    with open(DATA / fname, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = list(dict.fromkeys(r["slice_name"] for r in rows))
    return [[int(r["arrivals"]) for r in rows if r["slice_name"] == n] for n in names]


def test_per_slice_arrivals_identical_across_policies():
    cfg = get_config("csv_repro")
    static = run_episode(cfg, seed=42, policy="static")
    meta = run_episode(cfg, seed=42, policy="meta")
    assert _arrivals(static) == _arrivals(meta)
    # and equal to the standalone trace helper
    assert _arrivals(static) == [list(a) for a in zip(*arrival_traces(cfg, 42))]


def test_per_slice_streams_are_independent_of_other_slices():
    cfg = get_config("csv_repro")
    base = arrival_traces(cfg, 42)
    cfg["slices"][1]["traffic_params"]["mean_arrival"] = 10
    changed = arrival_traces(cfg, 42)
    assert changed[0] == base[0] and changed[2] == base[2]
    assert changed[1] != base[1]


def test_static_policy_never_reallocates():
    hist = run_episode(get_config("csv_repro"), seed=42, policy="static")
    assert {tuple(m.allocated_prbs for m in step) for step in hist} == {(40, 40, 20)}


def test_counterfactual_on_recorded_trace():
    cfg = get_config("csv_original")
    traces = _recorded_traces()
    meta = violation_events(run_episode(cfg, policy="meta", traces=traces))
    static = violation_events(run_episode(cfg, policy="static", traces=traces))
    # meta on the recorded trace is the paper run itself
    assert meta == {"URLLC": [41, 42], "eMBB": [42], "IoT": []}
    assert static == {"URLLC": [41, 42, 92, 110, 111, 194, 197], "eMBB": [], "IoT": []}


def test_policy_switch_rejects_unknown():
    with pytest.raises(ValueError):
        run_episode(get_config("csv_repro"), policy="oracle")
