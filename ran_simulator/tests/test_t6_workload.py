"""
T6 – can any fixed split be beaten?  (pre-registered)

Why the earlier workloads cannot answer this: every SLA there has a
load-independent static-safe reservation (URLLC 45, eMBB 38, IoT 0/5;
see ``core.baselines.static_safe_reservation``), and 45 + 38 + 5 <= 100.
Any split above those values is violation-free under *any* traffic, so the
hindsight/tuned static baselines score ~0 by construction.

Workload ``t6_reliability`` (config/experiments.py), fixed before the run:
  * URLLC latency <= 2.0 AND zero loss  → requirement grows with load
  * eMBB demand-aware 38 pkts/ts; IoT min-service 12 pkts/ts when backlogged
  * W4 phase timing; eMBB trough x0.3 (instead of x0.5)
  * per-slice peak requirements sum > 100, every phase sum <= 100

Pass criteria (20 eval seeds 400..419, tuned static tuned on 0..19,
95 % bootstrap CI on total violations per run):
  (i)  lower CI bound of hindsight_static and of tuned_static > 0
  (ii) upper CI bound of meta_fixed < lower CI bound of both
The meta-scheduler is the frozen default (config.META_SCHEDULER).

Outcome: (i) holds, (ii) does not – see the note on test_ii.
"""

import pytest

from config.experiments import get_config
from core.comparison import aggregate, compare_policies

CFG = "t6_reliability"


@pytest.fixture(scope="module")
def agg():
    rows = compare_policies(get_config(CFG),
                            policies=("hindsight_static", "tuned_static", "meta_fixed"))
    return aggregate(rows, ["viol_total"])


def test_i_best_static_baselines_are_nonzero(agg):
    for policy in ("hindsight_static", "tuned_static"):
        mean, lo, hi = agg[policy]["viol_total"]
        assert lo > 0, (policy, mean, lo, hi)


# Result of the single pre-registered run (2026-10-25): (i) holds – hindsight
# and tuned static both pick 47/38/15 and score 19.50 [16.85, 22.25] – but
# (ii) fails: meta_fixed scores 23.90 [21.05, 26.90].  The GBR reservation
# for eMBB (39) and IoT (16) caps URLLC at 45 PRBs, so meta cannot move the
# eMBB trough to URLLC.  An exploratory post-hoc run with
# throughput_reservation="backlogged" scored 27.80 [24.60, 31.00] (worse).
@pytest.mark.xfail(strict=True, reason="T6 (ii) not met: no workload found where the "
                   "reactive meta-scheduler beats the best fixed split")
def test_ii_meta_is_clearly_lower(agg):
    _, _, meta_hi = agg["meta_fixed"]["viol_total"]
    static_lo = min(agg[p]["viol_total"][1] for p in ("hindsight_static", "tuned_static"))
    assert meta_hi < static_lo, {p: agg[p]["viol_total"] for p in agg}
