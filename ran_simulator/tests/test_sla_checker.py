"""SLA checker modes: legacy vs demand-aware throughput SLAs."""

import pytest

from core.sla_checker import SLAChecker
from core.slice import NetworkSlice


def _embb(queue_before, arrivals, prbs, threshold=38.0):
    s = NetworkSlice({
        "name": "eMBB", "slice_type": "embb", "initial_prbs": prbs,
        "queue_capacity": 500, "service_rate": 1.0, "sla_type": "throughput",
        "sla_threshold": threshold,
        "traffic_params": {"mean_arrival": 0, "burst_prob": 0, "burst_mult": 1},
    })
    s.queue = queue_before
    s.enqueue(arrivals)
    s.dequeue(int(prbs * s.service_rate))
    s.compute_metrics()
    return s, s.snapshot(0)


@pytest.mark.parametrize("q, a, prbs, legacy, aware", [
    (0, 20, 40, True, False),     # little demand: legacy falsely violates
    (100, 10, 40, False, False),  # backlogged, served 40 >= 38
    (100, 10, 37, True, True),    # backlogged, served 37 < 38
    (0, 38, 37, True, True),      # offered exactly 38, served 37
    (0, 37, 30, True, False),     # offered 37 < 38: not a violation
])
def test_throughput_modes(q, a, prbs, legacy, aware):
    s, snap = _embb(q, a, prbs)
    assert SLAChecker("legacy").check_slice(s, snap) is legacy
    assert SLAChecker("demand_aware").check_slice(s, snap) is aware


def test_offered_counts_dropped_packets():
    s, snap = _embb(495, 20, 10)   # 15 dropped at capacity 500
    assert snap.dropped == 15
    assert SLAChecker.offered(snap) == 515


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        SLAChecker("strict")
