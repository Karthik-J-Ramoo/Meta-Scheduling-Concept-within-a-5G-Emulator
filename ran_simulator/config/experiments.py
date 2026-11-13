"""
experiments.py
==============
Named experiment configurations.

Each experiment is one self-contained dict, so a run can be reproduced from
its name alone and the exact dict can be written next to its results:

    simulation  – total_prbs / num_timesteps / random_seed
    rng_mode    – "shared"    : one RNG for all slices (legacy draw order)
                  "per_slice" : SeedSequence(seed).spawn(n), one stream per
                                slice, so arrivals do not depend on policy
    sla_mode    – "legacy" | "demand_aware"  (see core/sla_checker.py)
    slices      – slice dicts as in simulation_config.SLICES
    meta        – MetaScheduler parameters (see core/meta_scheduler.py)

``get_config(name)`` returns a deep copy, so callers may mutate it freely.
"""

from __future__ import annotations

import copy

from config.simulation_config import META_SCHEDULER, SIMULATION, SLICES


def _slice(name, slice_type, initial, cap, rate, sla_type, thr, lam, p_b, m, phases=None):
    traffic = {"mean_arrival": lam, "burst_prob": p_b, "burst_mult": m}
    if phases:
        traffic["phases"] = phases
    return {
        "name": name,
        "slice_type": slice_type,
        "initial_prbs": initial,
        "queue_capacity": cap,
        "service_rate": rate,
        "sla_type": sla_type,
        "sla_threshold": thr,
        "traffic_params": traffic,
    }


# Meta-scheduler exactly as shipped in the initial commit (floor 5, no-op decay).
LEGACY_META: dict = {
    "mode": "legacy",
    "reallocation_step": 5,
    "fairness_decay": 0.02,
    "min_prbs_per_slice": 5,
}

# Fixed allocator with the paper Table 3 parameters.
FIXED_META: dict = dict(META_SCHEDULER)


CONFIGS: dict[str, dict] = {
    # The config checked in at commit 8a9aec3; produced results/metrics.csv
    # in this repo (5.0 ms / 30 pkts, zero violations at seed 42).
    "checked_in": {
        "description": "Checked-in simulation_config.py (5.0 ms / 30 pkts).",
        "simulation": dict(SIMULATION),
        "rng_mode": "shared",
        "sla_mode": "legacy",
        "slices": copy.deepcopy(SLICES),
        "meta": dict(LEGACY_META),
    },
    # Recovered from the sibling working copy Meta_Scheduling_Test/ (its
    # simulation.log header and config/simulation_config.py). This is the run
    # behind the 2.0 ms / 38 pkts metrics.csv used in the paper draft.
    "csv_original": {
        "description": "Original stress-test run behind the paper CSV (2.0 ms / 38 pkts).",
        "simulation": {"total_prbs": 100, "num_timesteps": 200, "random_seed": 42},
        "rng_mode": "shared",
        "sla_mode": "legacy",
        "slices": [
            _slice("URLLC", "urllc", 40, 200, 1.5, "latency", 2.0, 32, 0.18, 3.5),
            _slice("eMBB", "embb", 40, 500, 1.0, "throughput", 38.0, 48, 0.20, 2.8),
            _slice("IoT", "iot", 20, 300, 0.8, "relaxed", 0.0, 18, 0.12, 2.4),
        ],
        "meta": dict(LEGACY_META),
    },
}


# Same workload with one RNG stream per slice: arrivals no longer depend on
# the policy, so static and meta runs see identical traffic (T1).
CONFIGS["csv_repro"] = {
    **copy.deepcopy(CONFIGS["csv_original"]),
    "description": "csv_original workload, per-slice RNG streams (policy-independent arrivals).",
    "rng_mode": "per_slice",
    "meta": dict(FIXED_META),
}

# Checked-in workload with per-slice streams (multi-seed baseline).
CONFIGS["checked_in_ps"] = {
    **copy.deepcopy(CONFIGS["checked_in"]),
    "description": "checked_in workload, per-slice RNG streams.",
    "rng_mode": "per_slice",
    "meta": dict(FIXED_META),
}


def _workload(description, sla_mode, urllc, embb, iot, embb_thr=38.0,
              iot_sla=("relaxed", 0.0), urllc_drop_cap=None):
    """Three-slice workload; urllc/embb/iot = (lam, p_b, m[, phases])."""
    cfg = {
        "description": description,
        "simulation": {"total_prbs": 100, "num_timesteps": 200, "random_seed": 42},
        "rng_mode": "per_slice",
        "sla_mode": sla_mode,
        "slices": [
            _slice("URLLC", "urllc", 40, 200, 1.5, "latency", 2.0, *urllc),
            _slice("eMBB", "embb", 40, 500, 1.0, "throughput", embb_thr, *embb),
            _slice("IoT", "iot", 20, 300, 0.8, *iot_sla, *iot),
        ],
        "meta": dict(FIXED_META),
    }
    if urllc_drop_cap is not None:
        cfg["slices"][0]["sla_drop_cap"] = urllc_drop_cap
    return cfg


# W1 – parameters fitted to the paper CSV by an external replica.
CONFIGS["csv_fitted"] = _workload(
    "W1: CSV-fitted workload (overloaded, mean demand 123.6 PRBs).", "legacy",
    (32, 0.21, 3.5), (48, 0.19, 2.8), (17.5, 0.155, 2.4))

# W2 – feasible stationary load (mean demand 90.7 PRBs).
CONFIGS["feasible"] = _workload(
    "W2: feasible stationary workload, demand-aware eMBB SLA.", "demand_aware",
    (32, 0.21, 3.5), (32, 0.19, 2.8), (10, 0.155, 2.4))

# W3 – regime shifts on W2.
CONFIGS["regime"] = _workload(
    "W3: W2 with regime shifts at t=70 and t=140.", "demand_aware",
    (32, 0.21, 3.5, [[0, 1.0], [70, 1.5], [140, 0.8]]),
    (32, 0.19, 2.8, [[0, 1.0], [70, 0.7], [140, 1.15]]),
    (10, 0.155, 2.4, [[0, 1.0], [70, 1.0], [140, 1.6]]))

# W4 – anti-correlated URLLC / eMBB phases on W2.
CONFIGS["anticorr"] = _workload(
    "W4: W2 with anti-correlated URLLC/eMBB phases every 50 steps.", "demand_aware",
    (32, 0.21, 3.5, [[0, 1.0], [50, 1.8], [100, 0.6], [150, 1.8]]),
    (32, 0.19, 2.8, [[0, 1.0], [50, 0.5], [100, 1.4], [150, 0.5]]),
    (10, 0.155, 2.4))


def get_config(name: str) -> dict:
    """Return a deep copy of the named experiment config (with its name)."""
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}; choose from {sorted(CONFIGS)}")
    cfg = copy.deepcopy(CONFIGS[name])
    cfg["name"] = name
    return cfg


# T6 (pre-registered, see tests/test_t6_workload.py).  Every SLA above has a
# load-independent static-safe reservation (URLLC 45, eMBB 38, IoT 5), so a
# fixed split can always reach zero violations.  T6 makes one requirement
# load-dependent (URLLC must also be loss-free), adds an IoT minimum-service
# SLA (12 pkts/ts when backlogged), and keeps W4's anti-correlated timing
# with a lighter eMBB trough (x0.3) so that bursts rarely offer >= 38 pkts.
CONFIGS["t6_reliability"] = _workload(
    "T6: anti-correlated phases, URLLC latency + zero loss, IoT min-service 12.",
    "demand_aware",
    (32, 0.21, 3.5, [[0, 1.0], [50, 1.8], [100, 0.6], [150, 1.8]]),
    (32, 0.19, 2.8, [[0, 1.0], [50, 0.3], [100, 1.4], [150, 0.3]]),
    (10, 0.155, 2.4),
    iot_sla=("throughput", 12.0),
    urllc_drop_cap=0.0,
)
