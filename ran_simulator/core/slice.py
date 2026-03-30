"""
slice.py
========
Data model for a single network slice.

Design rationale
----------------
* A ``NetworkSlice`` owns its **queue state** and the bookkeeping counters that
  other modules read from / write to each timestep.  It does *not* contain any
  scheduling logic – that is the responsibility of ``DUScheduler`` and
  ``MetaScheduler``.
* The class deliberately exposes mutable attributes (``queue``,
  ``allocated_prbs``, …) rather than hiding them behind getters, because the
  simulation loop is trusted code and the extra ceremony would hurt readability
  without adding safety.
* ``SliceMetrics`` is a lightweight snapshot dataclass used to decouple metric
  recording from live slice state.  It is the *only* object passed to
  ``PerformanceMonitor`` and ``SLAChecker``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timestep metrics snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SliceMetrics:
    """
    Immutable snapshot of one slice's KPIs at a single timestep.

    Consumers: PerformanceMonitor, SLAChecker, MetricCSVWriter.
    """
    timestep: int
    slice_name: str
    slice_type: str
    allocated_prbs: int
    arrivals: int
    served: int
    queue_length: int
    dropped: int
    throughput: float
    latency: float
    sla_type: str
    sla_threshold: float
    sla_satisfied: bool = True          # filled in by SLAChecker


# ---------------------------------------------------------------------------
# Network slice entity
# ---------------------------------------------------------------------------
class NetworkSlice:
    """
    Represents one logical network slice (e.g. URLLC, eMBB, IoT).

    Attributes that change every timestep
    --------------------------------------
    queue           – current number of queued packets
    allocated_prbs  – PRBs assigned by the meta-scheduler this step
    arrivals        – packets that arrived this step (set by traffic gen)
    served          – packets dequeued this step (set by DU scheduler)
    dropped         – packets lost to queue overflow this step
    throughput      – effective throughput this step
    latency         – instantaneous latency estimate this step
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        # ── identity ──────────────────────────────────────────────────
        self.name: str = cfg["name"]
        self.slice_type: str = cfg["slice_type"]

        # ── static parameters ─────────────────────────────────────────
        self.queue_capacity: int = cfg["queue_capacity"]
        self.service_rate: float = cfg["service_rate"]
        self.sla_type: str = cfg["sla_type"]
        self.sla_threshold: float = cfg["sla_threshold"]
        self.initial_prbs: int = cfg["initial_prbs"]
        self.traffic_params: dict = cfg["traffic_params"]

        # ── mutable per-timestep state ────────────────────────────────
        self.queue: int = 0
        self.allocated_prbs: int = self.initial_prbs
        self.arrivals: int = 0
        self.served: int = 0
        self.dropped: int = 0
        self.throughput: float = 0.0
        self.latency: float = 0.0

        logger.info(
            "Slice %-6s created  │ type=%-6s │ init_prbs=%d │ "
            "capacity=%d │ sla=%s<%.1f",
            self.name, self.slice_type, self.initial_prbs,
            self.queue_capacity, self.sla_type, self.sla_threshold,
        )

    # ── queue dynamics (called by DUScheduler) ────────────────────────

    def enqueue(self, arrivals: int) -> int:
        """
        Add *arrivals* packets to the queue, respecting capacity.

        Returns the number of dropped packets (overflow).
        """
        self.arrivals = arrivals
        space = max(0, self.queue_capacity - self.queue)
        accepted = min(arrivals, space)
        self.dropped = arrivals - accepted
        self.queue += accepted
        return self.dropped

    def dequeue(self, served: int) -> int:
        """
        Remove up to *served* packets from the queue.

        Returns the actual number of packets dequeued.
        """
        actual = min(served, self.queue)
        self.queue -= actual
        self.served = actual
        return actual

    # ── metric computation ────────────────────────────────────────────

    def compute_metrics(self) -> None:
        """
        Derive throughput and latency from current queue / service state.

        Called once per timestep *after* enqueue + dequeue have run.
        """
        self.throughput = float(self.served)

        # Latency ≈ queue_length / service_rate  (Little's-law inspired)
        effective_rate = self.allocated_prbs * self.service_rate
        if effective_rate > 0:
            self.latency = self.queue / effective_rate
        else:
            self.latency = float("inf") if self.queue > 0 else 0.0

    def snapshot(self, timestep: int) -> SliceMetrics:
        """Return an immutable metrics snapshot for this timestep."""
        return SliceMetrics(
            timestep=timestep,
            slice_name=self.name,
            slice_type=self.slice_type,
            allocated_prbs=self.allocated_prbs,
            arrivals=self.arrivals,
            served=self.served,
            queue_length=self.queue,
            dropped=self.dropped,
            throughput=self.throughput,
            latency=self.latency,
            sla_type=self.sla_type,
            sla_threshold=self.sla_threshold,
        )

    # ── reset (useful between episodes) ───────────────────────────────

    def reset(self) -> None:
        """Reset all mutable state to initial conditions."""
        self.queue = 0
        self.allocated_prbs = self.initial_prbs
        self.arrivals = 0
        self.served = 0
        self.dropped = 0
        self.throughput = 0.0
        self.latency = 0.0

    def __repr__(self) -> str:
        return (
            f"NetworkSlice(name={self.name!r}, prbs={self.allocated_prbs}, "
            f"queue={self.queue}, lat={self.latency:.2f}, "
            f"tput={self.throughput:.1f})"
        )
