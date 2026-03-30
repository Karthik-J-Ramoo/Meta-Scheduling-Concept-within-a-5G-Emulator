"""
meta_scheduler.py
=================
Cross-slice meta-scheduler (inter-slice resource orchestrator).

Design rationale
----------------
* The meta-scheduler sits *above* the DU scheduler in the O-RAN hierarchy.
  It decides **how many PRBs each slice gets** at every timestep, while the
  DU scheduler decides how to *use* those PRBs within a slice.
* This initial implementation is **rule-based**:
    1. Check which slices have SLA violations (fed by ``SLAChecker``).
    2. Increase the allocation for violated slices by ``reallocation_step``.
    3. Decrease the allocation for non-violated slices to compensate,
       respecting ``min_prbs_per_slice``.
    4. Apply a gentle *fairness decay* that nudges allocations back toward
       their initial (default) shares when all SLAs are satisfied.
    5. Clamp the total to ``total_prbs`` (hard budget constraint).
* The class exposes a clean ``allocate(slices, violations) -> dict`` interface
  so it can be **swapped out** for an Active-Inference agent or any other
  decision-making module without touching the rest of the simulator.

Extensibility hook
------------------
Subclass ``MetaScheduler`` and override ``allocate()`` to plug in:
  - Active Inference (free-energy minimisation)
  - Deep RL (PPO / SAC)
  - Model-predictive control
  - ChaosNet-driven allocation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence

from config.simulation_config import META_SCHEDULER, SIMULATION
from core.slice import NetworkSlice


logger = logging.getLogger(__name__)


class MetaScheduler:
    """
    Rule-based cross-slice PRB allocator.

    Parameters
    ----------
    total_prbs : int
        System-wide PRB budget (default from config).
    reallocation_step : int
        PRBs added/removed per adjustment action.
    fairness_decay : float
        Rate at which allocations drift back to default shares.
    min_prbs : int
        Minimum PRBs any single slice can hold.
    """

    def __init__(
        self,
        total_prbs: int | None = None,
        reallocation_step: int | None = None,
        fairness_decay: float | None = None,
        min_prbs: int | None = None,
    ) -> None:
        self.total_prbs = total_prbs or SIMULATION["total_prbs"]
        self.realloc_step = reallocation_step or META_SCHEDULER["reallocation_step"]
        self.fairness_decay = fairness_decay or META_SCHEDULER["fairness_decay"]
        self.min_prbs = min_prbs or META_SCHEDULER["min_prbs_per_slice"]

        logger.info(
            "MetaScheduler │ total=%d  step=%d  decay=%.3f  floor=%d",
            self.total_prbs, self.realloc_step, self.fairness_decay, self.min_prbs,
        )

    # ── public interface ──────────────────────────────────────────────

    def allocate(
        self,
        slices: Sequence[NetworkSlice],
        violations: Dict[str, bool],
        timestep: int,
    ) -> Dict[str, int]:
        """
        Decide PRB allocation for the next timestep.

        Parameters
        ----------
        slices : sequence of NetworkSlice
            All active slices (mutable – allocations are written back).
        violations : dict[str, bool]
            ``{slice_name: True}`` if the slice's SLA was violated this step.
        timestep : int
            Current timestep (for logging).

        Returns
        -------
        dict[str, int]
            Mapping ``slice_name → allocated_prbs`` after adjustment.

        Algorithm
        ---------
        1.  **Boost violated slices** – each violated slice receives up to
            ``realloc_step`` extra PRBs.
        2.  **Shrink donors** – non-violated slices donate PRBs proportionally,
            respecting ``min_prbs``.
        3.  **Fairness rebalance** – when *no* SLA is violated, gently push
            every allocation toward the initial share.
        4.  **Budget clamp** – ensure the total equals ``total_prbs`` exactly
            via a greedy residual distribution.
        """
        n = len(slices)
        allocs = {s.name: s.allocated_prbs for s in slices}
        violated_names = [name for name, v in violations.items() if v]
        safe_names = [name for name, v in violations.items() if not v]

        # ── Step 1 & 2: boost violated, shrink donors ────────────────
        if violated_names:
            total_needed = len(violated_names) * self.realloc_step

            # How much can donors actually give?
            donor_budget = 0
            for name in safe_names:
                donor_budget += max(0, allocs[name] - self.min_prbs)

            # Scale down if donors can't cover the full demand.
            grant = min(total_needed, donor_budget)
            per_violation = grant // max(len(violated_names), 1)

            # Boost violated slices
            for name in violated_names:
                allocs[name] += per_violation

            # Shrink donors proportionally
            if grant > 0 and donor_budget > 0:
                for name in safe_names:
                    donatable = max(0, allocs[name] - self.min_prbs)
                    reduction = int(donatable / donor_budget * grant)
                    allocs[name] -= reduction

            logger.info(
                "t=%d │ MetaSched │ VIOLATIONS %s │ boost=%d/slice  "
                "donor_budget=%d",
                timestep, violated_names, per_violation, donor_budget,
            )
        else:
            # ── Step 3: fairness decay toward initial shares ──────────
            initial = {s.name: s.initial_prbs for s in slices}
            for s in slices:
                diff = initial[s.name] - allocs[s.name]
                nudge = int(round(diff * self.fairness_decay))
                allocs[s.name] += nudge

        # ── Step 4: hard budget clamp ─────────────────────────────────
        allocs = self._clamp_budget(allocs, slices)

        # Write back
        for s in slices:
            s.allocated_prbs = allocs[s.name]

        logger.debug(
            "t=%d │ MetaSched │ final allocs: %s  (sum=%d)",
            timestep, allocs, sum(allocs.values()),
        )
        return allocs

    # ── internal helpers ──────────────────────────────────────────────

    def _clamp_budget(
        self,
        allocs: Dict[str, int],
        slices: Sequence[NetworkSlice],
    ) -> Dict[str, int]:
        """
        Ensure allocations sum to exactly ``total_prbs`` and every slice
        has at least ``min_prbs``.

        Strategy: apply floor, then distribute the residual round-robin
        starting from the slice with the highest current demand (queue).
        """
        # Floor enforcement
        for name in allocs:
            allocs[name] = max(allocs[name], self.min_prbs)

        residual = self.total_prbs - sum(allocs.values())

        if residual == 0:
            return allocs

        # Sort slices by descending queue length (neediest first) for positive
        # residual, or ascending for negative residual (take from least needy).
        sorted_slices = sorted(
            slices,
            key=lambda s: s.queue,
            reverse=(residual > 0),
        )

        step = 1 if residual > 0 else -1
        idx = 0
        while residual != 0:
            name = sorted_slices[idx % len(sorted_slices)].name
            if step < 0 and allocs[name] <= self.min_prbs:
                idx += 1
                # Safety: if we've gone through all slices without
                # being able to reduce, break to avoid infinite loop.
                if idx >= len(sorted_slices) * 2:
                    break
                continue
            allocs[name] += step
            residual -= step
            idx += 1

        return allocs
