"""
T0 replay validation and legacy anchors.

* ``test_replay_*``  – rebuild every slice queue from the recorded
  ``arrivals`` and ``allocated_prbs`` alone, using the model equations

      tot  = min(Q + A, C)          drop = max(0, Q + A - C)
      S    = min(tot, floor(P * r)) Q'   = tot - S
      lat  = Q' / (P * r)

  and require zero mismatches against the CSV.
* ``test_legacy_anchor_*`` – the legacy code path (shared RNG, legacy SLA
  checker, legacy meta-scheduler) must regenerate each CSV byte-for-byte.
"""

import csv
import math

import pytest

from conftest import DATA
from config.experiments import get_config
from core.performance_monitor import CSV_FIELDS, PerformanceMonitor
from core.simulation import run_episode

# fixture CSV → config that produced it
FIXTURES = {
    "metrics_csv_original.csv": "csv_original",       # 2.0 ms / 38 pkts (paper run)
    "metrics_checked_in.csv": "checked_in",     # 5.0 ms / 30 pkts
}


def _rows(fname):
    with open(DATA / fname, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def replay_mismatches(rows, slice_cfgs):
    """Return a list of (timestep, slice, field, csv, model) mismatches."""
    params = {c["name"]: (c["service_rate"], c["queue_capacity"]) for c in slice_cfgs}
    queue = {name: 0 for name in params}
    bad = []
    for row in rows:
        name = row["slice_name"]
        r, cap = params[name]
        q, a, p = queue[name], int(row["arrivals"]), int(row["allocated_prbs"])
        tot = min(q + a, cap)
        drop = max(0, q + a - cap)
        served = min(tot, math.floor(p * r))
        q_next = tot - served
        lat = q_next / (p * r)
        queue[name] = q_next
        model = {"queue_length": q_next, "served": served, "dropped": drop}
        for field, val in model.items():
            if int(row[field]) != val:
                bad.append((row["timestep"], name, field, row[field], val))
        if abs(float(row["latency"]) - round(lat, 4)) > 1e-9:
            bad.append((row["timestep"], name, "latency", row["latency"], lat))
        if float(row["throughput"]) != served:
            bad.append((row["timestep"], name, "throughput", row["throughput"], served))
    return bad


@pytest.mark.parametrize("fname", sorted(FIXTURES))
def test_replay_zero_mismatches(fname):
    rows = _rows(fname)
    assert len(rows) == 600
    bad = replay_mismatches(rows, get_config(FIXTURES[fname])["slices"])
    assert bad == [], f"{len(bad)} mismatches, first: {bad[:5]}"


@pytest.mark.parametrize("fname", sorted(FIXTURES))
def test_legacy_anchor_reproduces_csv(fname):
    rows = _rows(fname)
    history = run_episode(get_config(FIXTURES[fname]), seed=42, policy="meta")
    got = [PerformanceMonitor._metrics_to_dict(m) for step in history for m in step]
    assert len(got) == len(rows)
    for want, have in zip(rows, got):
        for field in CSV_FIELDS:
            assert str(have[field]) == want[field], (want["timestep"], want["slice_name"], field)
