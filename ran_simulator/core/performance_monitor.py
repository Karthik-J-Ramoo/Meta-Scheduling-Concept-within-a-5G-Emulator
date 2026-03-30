"""
performance_monitor.py
======================
Collects, aggregates, and exposes per-slice and system-wide KPIs.

Design rationale
----------------
* Acts as the **read-side** of the simulation's data plane.  Every timestep,
  the main loop feeds it ``SliceMetrics`` snapshots; the monitor accumulates
  them and can produce running statistics or final summaries.
* Also drives the ``MetricCSVWriter`` so that the CSV file is written
  incrementally (no large in-memory buffer needed for long runs).
* Provides convenience methods for the meta-scheduler to query recent
  performance (e.g. moving-average latency).

Extensibility hook
------------------
Add Prometheus / InfluxDB exporters, or a live Matplotlib dashboard, by
subscribing to the ``record()`` call.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from core.slice import SliceMetrics
from utils.logger import MetricCSVWriter


logger = logging.getLogger(__name__)

# Fields written to the CSV file each timestep, per slice.
CSV_FIELDS: list[str] = [
    "timestep",
    "slice_name",
    "slice_type",
    "allocated_prbs",
    "arrivals",
    "served",
    "queue_length",
    "dropped",
    "throughput",
    "latency",
    "sla_type",
    "sla_threshold",
    "sla_satisfied",
]


class PerformanceMonitor:
    """
    Central KPI collector and CSV writer.

    Parameters
    ----------
    csv_writer : MetricCSVWriter, optional
        If provided, rows are streamed to CSV on every ``record()`` call.
    window : int
        Number of recent timesteps kept for moving-average queries.
    """

    def __init__(
        self,
        csv_writer: Optional[MetricCSVWriter] = None,
        window: int = 20,
    ) -> None:
        self._csv = csv_writer
        self._window = window

        # Per-slice history: name → list[SliceMetrics]  (bounded by window)
        self._history: Dict[str, List[SliceMetrics]] = defaultdict(list)

        # Global counters
        self._total_records = 0

    # ── recording ─────────────────────────────────────────────────────

    def record(self, metrics: Sequence[SliceMetrics]) -> None:
        """
        Ingest a batch of per-slice metrics for one timestep.

        Parameters
        ----------
        metrics : sequence of SliceMetrics
            One entry per slice, all sharing the same ``timestep``.
        """
        for m in metrics:
            # Maintain bounded history
            hist = self._history[m.slice_name]
            hist.append(m)
            if len(hist) > self._window:
                hist.pop(0)

            # Stream to CSV
            if self._csv is not None:
                self._csv.write_row(self._metrics_to_dict(m))

            self._total_records += 1

        # Periodic console summary (every 10 timesteps)
        if metrics and metrics[0].timestep % 10 == 0:
            self._log_summary(metrics)

    # ── queries (used by meta-scheduler / SLA checker) ────────────────

    def moving_average_latency(self, slice_name: str) -> float:
        """Return the mean latency over the recent window for *slice_name*."""
        hist = self._history.get(slice_name, [])
        if not hist:
            return 0.0
        return sum(m.latency for m in hist) / len(hist)

    def moving_average_throughput(self, slice_name: str) -> float:
        """Return the mean throughput over the recent window."""
        hist = self._history.get(slice_name, [])
        if not hist:
            return 0.0
        return sum(m.throughput for m in hist) / len(hist)

    def latest(self, slice_name: str) -> Optional[SliceMetrics]:
        """Return the most recent metrics for *slice_name*, or None."""
        hist = self._history.get(slice_name, [])
        return hist[-1] if hist else None

    # ── final report ──────────────────────────────────────────────────

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Produce a per-slice summary dict suitable for pretty-printing.

        Returns
        -------
        dict[str, dict]
            ``{slice_name: {avg_latency, avg_throughput, total_dropped, …}}``
        """
        result: Dict[str, Dict[str, Any]] = {}
        for name, hist in self._history.items():
            n = len(hist)
            if n == 0:
                continue
            result[name] = {
                "window_size": n,
                "avg_latency": sum(m.latency for m in hist) / n,
                "avg_throughput": sum(m.throughput for m in hist) / n,
                "avg_queue": sum(m.queue_length for m in hist) / n,
                "total_dropped": sum(m.dropped for m in hist),
                "sla_violations": sum(1 for m in hist if not m.sla_satisfied),
            }
        return result

    # ── lifecycle ─────────────────────────────────────────────────────

    def flush(self) -> None:
        if self._csv is not None:
            self._csv.flush()

    def close(self) -> None:
        if self._csv is not None:
            self._csv.close()

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _metrics_to_dict(m: SliceMetrics) -> Dict[str, Any]:
        return {
            "timestep": m.timestep,
            "slice_name": m.slice_name,
            "slice_type": m.slice_type,
            "allocated_prbs": m.allocated_prbs,
            "arrivals": m.arrivals,
            "served": m.served,
            "queue_length": m.queue_length,
            "dropped": m.dropped,
            "throughput": m.throughput,
            "latency": round(m.latency, 4),
            "sla_type": m.sla_type,
            "sla_threshold": m.sla_threshold,
            "sla_satisfied": m.sla_satisfied,
        }

    @staticmethod
    def _log_summary(metrics: Sequence[SliceMetrics]) -> None:
        t = metrics[0].timestep
        parts = []
        for m in metrics:
            flag = "✓" if m.sla_satisfied else "✗"
            parts.append(
                f"{m.slice_name}: prb={m.allocated_prbs} q={m.queue_length} "
                f"lat={m.latency:.2f} tput={m.throughput:.0f} sla={flag}"
            )
        logger.info("t=%-4d │ %s", t, "  │  ".join(parts))
