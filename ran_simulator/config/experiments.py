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

from config.simulation_config import SIMULATION, SLICES


def _slice(name, slice_type, initial, cap, rate, sla_type, thr, lam, p_b, m):
    return {
        "name": name,
        "slice_type": slice_type,
        "initial_prbs": initial,
        "queue_capacity": cap,
        "service_rate": rate,
        "sla_type": sla_type,
        "sla_threshold": thr,
        "traffic_params": {"mean_arrival": lam, "burst_prob": p_b, "burst_mult": m},
    }


# Meta-scheduler exactly as shipped in the initial commit (floor 5, no-op decay).
LEGACY_META: dict = {
    "mode": "legacy",
    "reallocation_step": 5,
    "fairness_decay": 0.02,
    "min_prbs_per_slice": 5,
}


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
}


def get_config(name: str) -> dict:
    """Return a deep copy of the named experiment config (with its name)."""
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}; choose from {sorted(CONFIGS)}")
    cfg = copy.deepcopy(CONFIGS[name])
    cfg["name"] = name
    return cfg
