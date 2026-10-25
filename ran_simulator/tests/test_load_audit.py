"""
T4 – load audit.

Mean PRB demand per slice is lam·(1 − p_b + p_b·m)/r.  A workload whose mean
demand exceeds the PRB budget, or whose queues overflow on > 20 % of steps,
is overloaded: SLA violations there measure the budget, not the scheduler.
"""

import pytest

from config.experiments import get_config
from core.episode_metrics import OverloadError, audit_load, prb_demand
from core.simulation import run_episode


def test_csv_fitted_demand_matches_replica():
    d = prb_demand(get_config("csv_fitted"))
    assert d["mean"] == pytest.approx({"URLLC": 32.53, "eMBB": 64.42, "IoT": 26.62}, abs=0.01)
    assert d["mean_sum"] == pytest.approx(123.6, abs=0.05)


def test_feasible_demand_matches_replica():
    assert prb_demand(get_config("feasible"))["mean_sum"] == pytest.approx(90.7, abs=0.05)


@pytest.mark.parametrize("name", ["csv_repro", "csv_fitted"])
@pytest.mark.parametrize("policy", ["static", "meta"])
def test_paper_workloads_fail_audit(name, policy):
    cfg = get_config(name)
    with pytest.raises(OverloadError, match="mean PRB demand"):
        audit_load(cfg, run_episode(cfg, seed=42, policy=policy))


@pytest.mark.parametrize("name", ["feasible", "regime", "anticorr"])
def test_feasible_workloads_pass_audit(name):
    cfg = get_config(name)
    report = audit_load(cfg, run_episode(cfg, seed=42, policy="meta"))
    assert report["mean_sum"] <= 100


def test_time_varying_workloads_have_peaks_above_budget():
    # Phase peaks exceed 100 PRBs although the time-average fits.
    for name in ("regime", "anticorr"):
        d = prb_demand(get_config(name))
        assert d["mean_sum"] <= 100 < d["peak_sum"]
