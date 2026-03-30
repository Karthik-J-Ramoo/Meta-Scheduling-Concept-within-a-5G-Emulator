"""
du_scheduler.py
===============
Distributed-Unit (DU) level scheduler.

Design rationale
----------------
* In a real O-RAN system the DU scheduler maps transport-block sizes to
  allocated PRBs on a per-TTI basis.  Here we abstract this as:
      ``served = min(queue, floor(allocated_prbs × service_rate))``
  which captures the essential *capacity-limited draining* behaviour without
  modelling modulation, CQI, or HARQ.
* The scheduler is **stateless** – it applies the same policy every timestep.
  This makes it easy to later inject a DRL-based intra-slice scheduler.
* After serving, the slice's ``compute_metrics()`` is called so that
  throughput / latency are always consistent with the latest queue state.

Extensibility hook
------------------
Subclass ``DUScheduler`` and override ``schedule_slice`` to implement
proportional-fair, round-robin, or ML-driven intra-slice policies.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

from core.slice import NetworkSlice


logger = logging.getLogger(__name__)


class DUScheduler:
    """
    Simple capacity-limited DU scheduler.

    For each slice the scheduler computes the maximum number of packets
    that can be served given the PRB allocation and service rate, then
    drains the queue accordingly.
    """

    def schedule(
        self,
        slices: Sequence[NetworkSlice],
        timestep: int,
    ) -> None:
        """
        Run one scheduling round across all slices.

        Parameters
        ----------
        slices : sequence of NetworkSlice
            The slices to schedule.
        timestep : int
            Current simulation timestep (used only for logging).
        """
        for s in slices:
            self.schedule_slice(s, timestep)

    def schedule_slice(self, s: NetworkSlice, timestep: int) -> None:
        """
        Serve as many queued packets as the PRB allocation allows.

        Scheduling formula
        ------------------
        ``max_serve = floor(allocated_prbs × service_rate)``
        ``served    = min(queue, max_serve)``

        After dequeuing, ``s.compute_metrics()`` is invoked so that
        throughput and latency reflect the new queue state.
        """
        max_serve = math.floor(s.allocated_prbs * s.service_rate)
        served = s.dequeue(max_serve)
        s.compute_metrics()

        logger.debug(
            "t=%d │ DU[%s] │ prbs=%d  svc_rate=%.2f → max_serve=%d  "
            "served=%d  queue_after=%d",
            timestep, s.name, s.allocated_prbs, s.service_rate,
            max_serve, served, s.queue,
        )
