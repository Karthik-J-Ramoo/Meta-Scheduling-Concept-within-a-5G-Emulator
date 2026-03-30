"""
simulation_config.py
====================
Central configuration registry for the RAN meta-scheduling simulator.

Design rationale
----------------
* Every tuneable parameter lives here so experiments are reproducible and
  diffable (just compare config files).
* Slice definitions use plain dicts so they can later be loaded from YAML /
  JSON without touching downstream code.
* The META_SCHEDULER section is intentionally kept separate from per-slice
  parameters to allow a future Active-Inference agent to own its own config
  namespace.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Global simulation parameters
# ---------------------------------------------------------------------------
SIMULATION: dict = {
    "total_prbs": 100,          # Total Physical Resource Blocks in the system
    "num_timesteps": 200,       # Duration of one simulation episode
    "random_seed": 42,          # For reproducibility (None → non-deterministic)
}

# ---------------------------------------------------------------------------
# Network-slice definitions
# ---------------------------------------------------------------------------
# Each slice is a dict consumed by the Slice dataclass at construction time.
#
#   name             – human-readable label
#   slice_type       – one of {"urllc", "embb", "iot"}
#   initial_prbs     – PRBs allocated at t=0
#   queue_capacity   – max packets the queue can hold (overflow → drop)
#   service_rate     – packets served per PRB per timestep
#   sla_type         – "latency" | "throughput" | "relaxed"
#   sla_threshold    – numeric threshold for the chosen SLA metric
#   traffic_params   – passed verbatim to the TrafficGenerator
#       mean_arrival – Poisson λ for arrivals per timestep
#       burst_prob   – probability of a burst event each timestep
#       burst_mult   – multiplier on mean_arrival during a burst
# ---------------------------------------------------------------------------

SLICES: list[dict] = [
    {
        "name": "URLLC",
        "slice_type": "urllc",
        "initial_prbs": 40,
        "queue_capacity": 200,
        "service_rate": 1.5,       # aggressive service per PRB
        "sla_type": "latency",
        "sla_threshold": 5.0,      # max tolerable latency (abstract units)
        "traffic_params": {
            "mean_arrival": 20,
            "burst_prob": 0.10,
            "burst_mult": 3.0,
        },
    },
    {
        "name": "eMBB",
        "slice_type": "embb",
        "initial_prbs": 40,
        "queue_capacity": 500,
        "service_rate": 1.0,
        "sla_type": "throughput",
        "sla_threshold": 30.0,     # minimum throughput per timestep
        "traffic_params": {
            "mean_arrival": 35,
            "burst_prob": 0.15,
            "burst_mult": 2.5,
        },
    },
    {
        "name": "IoT",
        "slice_type": "iot",
        "initial_prbs": 20,
        "queue_capacity": 300,
        "service_rate": 0.8,
        "sla_type": "relaxed",
        "sla_threshold": 0.0,      # no hard constraint
        "traffic_params": {
            "mean_arrival": 12,
            "burst_prob": 0.05,
            "burst_mult": 2.0,
        },
    },
]

# ---------------------------------------------------------------------------
# Meta-scheduler tunables
# ---------------------------------------------------------------------------
META_SCHEDULER: dict = {
    "reallocation_step": 5,     # PRBs to shift per adjustment
    "fairness_decay": 0.02,     # rate at which allocations drift back to default
    "min_prbs_per_slice": 5,    # floor – never starve a slice completely
}

# ---------------------------------------------------------------------------
# Logging / output
# ---------------------------------------------------------------------------
LOGGING: dict = {
    "console_level": "INFO",         # DEBUG | INFO | WARNING | ERROR
    "file_level": "DEBUG",
    "log_dir": "results",
    "csv_filename": "metrics.csv",
    "log_filename": "simulation.log",
}
