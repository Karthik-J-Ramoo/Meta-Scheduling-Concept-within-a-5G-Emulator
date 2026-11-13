"""
sla_checker.py
==============
Evaluates whether each slice's Service-Level Agreement is satisfied.

Design rationale
----------------
* Each SLA type (``latency``, ``throughput``, ``relaxed``) has a simple
  predicate.  The checker returns a ``{slice_name: bool}`` violations dict
  that the meta-scheduler consumes directly.
* The class is **stateless** – it evaluates the current timestep's metrics
  only, making it easy to swap in a predictive checker (e.g. ChaosNet-based)
  that looks at a sliding window instead.
* Two modes for throughput SLAs:
    - ``legacy``       – violated iff ``served < threshold``.  This is false
      when the slice simply had less than ``threshold`` packets to send, and
      cannot detect starvation of a saturated queue beyond the threshold.
    - ``demand_aware`` – violated iff the slice *offered* at least
      ``threshold`` packets (``queue_before + arrivals``) and was served
      fewer than ``threshold``.  A minimum-service SLA for IoT (e.g. 12
      pkts/ts when backlogged) is a throughput SLA in this mode.

Extensibility hook
------------------
Override ``check_slice()`` to implement:
  - ChaosNet SLA-violation *prediction* (proactive, not reactive).
  - Composite SLA constraints (latency AND throughput).
  - Probabilistic SLA (e.g. 99th-percentile latency).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, Sequence

from core.slice import NetworkSlice, SliceMetrics


logger = logging.getLogger(__name__)


class SLAChecker:
    """
    Stateless SLA evaluator.

    For each slice it tests the relevant metric against the configured
    threshold and returns whether the SLA is **violated**.

    Parameters
    ----------
    mode : str
        ``"legacy"`` or ``"demand_aware"`` (throughput SLAs only).
    """

    MODES = ("legacy", "demand_aware")

    def __init__(self, mode: str = "legacy") -> None:
        if mode not in self.MODES:
            raise ValueError(f"SLA mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode

    def check_all(
        self,
        slices: Sequence[NetworkSlice],
        snapshots: list[SliceMetrics],
        timestep: int,
    ) -> tuple[Dict[str, bool], list[SliceMetrics]]:
        """
        Evaluate SLA satisfaction for every slice.

        Parameters
        ----------
        slices : sequence of NetworkSlice
            Live slice objects (read-only here).
        snapshots : list of SliceMetrics
            Pre-built metric snapshots for this timestep.
        timestep : int
            Current timestep (for logging).

        Returns
        -------
        violations : dict[str, bool]
            ``{slice_name: True}`` means the SLA *was* violated.
        updated_snapshots : list[SliceMetrics]
            The same snapshots with ``sla_satisfied`` field filled in.
        """
        violations: Dict[str, bool] = {}
        updated: list[SliceMetrics] = []

        for s, snap in zip(slices, snapshots):
            violated = self.check_slice(s, snap)
            violations[s.name] = violated

            # Stamp the snapshot with the SLA result
            updated.append(replace(snap, sla_satisfied=not violated))

            if violated:
                logger.warning(
                    "t=%d │ SLA VIOLATION │ %s │ type=%s  value=%.2f  "
                    "threshold=%.2f",
                    timestep, s.name, s.sla_type,
                    self._metric_value(s, snap), s.sla_threshold,
                )

        return violations, updated

    # ── per-slice evaluation ──────────────────────────────────────────

    def check_slice(self, s: NetworkSlice, snap: SliceMetrics) -> bool:
        """
        Return ``True`` if the SLA is **violated** for slice *s*.

        SLA rules
        ---------
        * **latency**   – violated when ``latency > sla_threshold``
        * **throughput** – legacy: violated when ``throughput < sla_threshold``;
          demand_aware: violated when ``offered >= sla_threshold`` and
          ``served < sla_threshold``
        * **relaxed**    – never violated (IoT best-effort)
        * optional ``sla_drop_cap`` (any type) – also violated when
          ``dropped > sla_drop_cap × offered`` (reliability)
        """
        if s.sla_drop_cap is not None and snap.dropped > s.sla_drop_cap * self.offered(snap):
            return True

        if s.sla_type == "latency":
            return snap.latency > s.sla_threshold

        if s.sla_type == "throughput":
            if self.mode == "demand_aware":
                return (self.offered(snap) >= s.sla_threshold
                        and snap.served < s.sla_threshold)
            return snap.throughput < s.sla_threshold

        # "relaxed" or unknown → never violated
        return False

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def offered(snap: SliceMetrics) -> int:
        """Packets the slice could have sent this step: queue_before + arrivals."""
        return snap.queue_length + snap.served + snap.dropped

    @staticmethod
    def _metric_value(s: NetworkSlice, snap: SliceMetrics) -> float:
        """Return the metric relevant to the slice's SLA type."""
        if s.sla_type == "latency":
            return snap.latency
        if s.sla_type == "throughput":
            return snap.throughput
        return 0.0
