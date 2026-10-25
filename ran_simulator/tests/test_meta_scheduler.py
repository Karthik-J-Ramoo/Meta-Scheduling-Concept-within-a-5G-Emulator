"""
T3 – allocator unit tests for the fixed meta-scheduler.

Legacy-mode counterparts are marked ``xfail(strict=True)``: they document
the defects (ratchet, donor below requirement) and fail loudly if the legacy
path ever changes behaviour.
"""

import copy
import math

import numpy as np
import pytest

from config.experiments import LEGACY_META
from core.meta_scheduler import MetaScheduler
from core.slice import NetworkSlice

TOTAL = 100
CFGS = [
    ("URLLC", "urllc", 40, 200, 1.5, "latency", 2.0),
    ("eMBB", "embb", 40, 500, 1.0, "throughput", 38.0),
    ("IoT", "iot", 20, 300, 0.8, "relaxed", 0.0),
]


def make_slices(allocs, queues=(0, 0, 0), arrivals=(0, 0, 0)):
    slices = []
    for (name, typ, init, cap, rate, sla, thr), p, q, a in zip(CFGS, allocs, queues, arrivals):
        s = NetworkSlice({
            "name": name, "slice_type": typ, "initial_prbs": init,
            "queue_capacity": cap, "service_rate": rate, "sla_type": sla,
            "sla_threshold": thr,
            "traffic_params": {"mean_arrival": 0, "burst_prob": 0, "burst_mult": 1},
        })
        s.allocated_prbs, s.queue, s.arrivals = p, q, a
        s.latency = q / (p * rate)
        slices.append(s)
    return slices


def fixed(**kw):
    return MetaScheduler(total_prbs=TOTAL, reallocation_step=5, fairness_decay=0.02,
                         min_prbs=10, mode="fixed", early_warning=0.5, need_margin=1, **kw)


def legacy():
    return MetaScheduler(total_prbs=TOTAL, reallocation_step=LEGACY_META["reallocation_step"],
                         fairness_decay=LEGACY_META["fairness_decay"],
                         min_prbs=LEGACY_META["min_prbs_per_slice"], mode="legacy")


def random_cases(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        u = int(rng.integers(10, 81))
        e = int(rng.integers(10, TOTAL - u - 10 + 1))
        allocs = (u, e, TOTAL - u - e)
        queues = (int(rng.integers(0, 201)), int(rng.integers(0, 501)), int(rng.integers(0, 301)))
        arrivals = tuple(int(x) for x in rng.integers(0, 150, size=3))
        viol = {name: bool(v) for (name, *_), v in zip(CFGS, rng.integers(0, 2, size=3))}
        yield allocs, queues, arrivals, viol


def test_a_budget_exact_and_floor():
    ms = fixed()
    for allocs, queues, arrivals, viol in random_cases():
        out = ms.allocate(make_slices(allocs, queues, arrivals), viol, 0)
        assert sum(out.values()) == TOTAL
        assert min(out.values()) >= ms.min_prbs, (allocs, queues, arrivals, viol, out)


def test_b_violated_slice_never_decreases():
    ms = fixed()
    for allocs, queues, arrivals, viol in random_cases(seed=1):
        out = ms.allocate(make_slices(allocs, queues, arrivals), viol, 0)
        for (name, *_), before in zip(CFGS, allocs):
            if viol[name]:
                assert out[name] >= before, (allocs, queues, arrivals, viol, out)


def test_c_donor_never_below_own_requirement():
    ms = fixed()
    for allocs, queues, arrivals, viol in random_cases(seed=2):
        slices = make_slices(allocs, queues, arrivals)
        need = {s.name: ms.requirement(s) for s in slices}
        out = ms.allocate(slices, viol, 0)
        for (name, *_), before in zip(CFGS, allocs):
            if not viol[name]:
                assert out[name] >= min(before, need[name]), (allocs, queues, arrivals, viol, out)


def test_c_backlogged_embb_keeps_sla_share():
    # Paper CSV t=41 state: URLLC violated, eMBB backlogged (queue 460).
    # Legacy cut eMBB 40 -> 37 and caused the eMBB violation at t=42.
    state = dict(allocs=(40, 40, 20), queues=(123, 460, 193), arrivals=(115, 56, 7))
    viol = {"URLLC": True, "eMBB": False, "IoT": False}
    out = fixed().allocate(make_slices(**state), viol, 41)
    assert out["eMBB"] >= math.ceil(38 / 1.0) + 1
    assert out["URLLC"] >= 45                       # full +5 boost
    leg = legacy().allocate(make_slices(**state), viol, 41)
    assert (leg["URLLC"], leg["eMBB"], leg["IoT"]) == (44, 37, 19)  # the CSV's t=42 row


def _restore_steps(ms, start, max_steps=40):
    slices = make_slices(start)
    initial = [c[2] for c in CFGS]
    quiet = {name: False for name, *_ in CFGS}
    for step in range(1, max_steps + 1):
        for s in slices:        # light traffic: no backlog, small arrivals
            s.queue, s.arrivals, s.latency = 0, 5, 0.0
        out = ms.allocate(slices, quiet, step)
        if all(abs(out[n] - i) <= 2 for (n, *_), i in zip(CFGS, initial)):
            return step
    return None


@pytest.mark.parametrize("start", [(49, 42, 9), (55, 35, 10), (60, 30, 10), (30, 55, 15)])
def test_d_restoration_within_20_steps(start):
    steps = _restore_steps(fixed(), start)
    assert steps is not None and steps <= 20, steps


@pytest.mark.xfail(strict=True, reason="legacy ratchet: int(round(diff*0.02)) == 0 for |diff| < 25")
def test_d_legacy_ratchet():
    steps = _restore_steps(legacy(), (49, 42, 9))
    assert steps is not None and steps <= 20


def test_e_deterministic_and_side_effect_free():
    ms = fixed()
    for allocs, queues, arrivals, viol in random_cases(n=300, seed=3):
        a, b = make_slices(allocs, queues, arrivals), make_slices(allocs, queues, arrivals)
        before = [{k: v for k, v in vars(s).items() if k != "allocated_prbs"} for s in a]
        ms_state = copy.deepcopy(vars(ms))
        out_a = ms.allocate(a, viol, 0)
        out_b = ms.allocate(b, viol, 0)
        assert out_a == out_b
        assert [{k: v for k, v in vars(s).items() if k != "allocated_prbs"} for s in a] == before
        assert vars(ms) == ms_state
        assert [s.allocated_prbs for s in a] == [out_a[n] for n, *_ in CFGS]


def test_zero_values_are_honoured():
    ms = MetaScheduler(total_prbs=TOTAL, reallocation_step=0, fairness_decay=0.0,
                       min_prbs=0, mode="fixed", need_margin=0)
    assert (ms.realloc_step, ms.fairness_decay, ms.min_prbs, ms.need_margin) == (0, 0.0, 0, 0)


def test_early_warning_boosts_before_violation():
    # URLLC latency 1.2 ms > 0.5 * 2.0 ms but below the 2.0 ms SLA
    slices = make_slices((40, 40, 20), queues=(72, 0, 0), arrivals=(60, 10, 5))
    quiet = {name: False for name, *_ in CFGS}
    assert fixed().allocate(slices, quiet, 0)["URLLC"] > 40
    slices = make_slices((40, 40, 20), queues=(72, 0, 0), arrivals=(60, 10, 5))
    assert MetaScheduler(total_prbs=TOTAL, reallocation_step=5, fairness_decay=0.02,
                         min_prbs=10, mode="fixed", early_warning=None,
                         need_margin=1).allocate(slices, quiet, 0)["URLLC"] <= 40
