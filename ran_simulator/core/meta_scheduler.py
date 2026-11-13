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
* ``mode="fixed"`` (default) replaces steps 2-5 with an exact allocator:
    - every slice gets a *requirement* – the PRBs its SLA needs next step,
      estimated from its queue plus the last arrivals (plus ``need_margin``);
    - a violated slice is boosted by ``reallocation_step`` and never shrinks;
    - a donor is never cut below ``min(current, requirement)``;
    - non-violated slices move toward their initial shares every step by
      ``max(1, ceil(decay * |diff|))`` PRBs (the legacy ``int(round(...))``
      is 0 for |diff| < 25, so allocations ratcheted);
    - the budget is met exactly by integer largest-remainder apportioning,
      cutting donor slack first, then boost, then requirement slack;
    - optional early warning: a latency slice whose latency exceeds
      ``early_warning * threshold`` is boosted before it violates.
  ``mode="legacy"`` keeps the original algorithm bit-for-bit.
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
import math
from typing import Any, Dict, Optional, Sequence

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
    mode : str
        ``"fixed"`` or ``"legacy"`` (see module docstring).
    early_warning : float or None
        Fixed mode: boost a latency slice once latency exceeds this fraction
        of its threshold.  ``None`` disables it.
    need_margin : int
        Fixed mode: PRBs added to a non-zero SLA requirement.
    throughput_reservation : str
        Fixed mode, throughput SLAs: ``"backlogged"`` requires
        ``ceil(min(T, offered)/r)`` PRBs; ``"always"`` reserves ``ceil(T/r)``
        regardless of demand (a guaranteed-bit-rate reservation).

    ``None`` means "take the value from config.META_SCHEDULER"; an explicit
    0 is honoured.
    """

    MODES = ("fixed", "legacy")

    def __init__(
        self,
        total_prbs: int | None = None,
        reallocation_step: int | None = None,
        fairness_decay: float | None = None,
        min_prbs: int | None = None,
        mode: str | None = None,
        early_warning: Optional[float] = None,
        need_margin: int | None = None,
        throughput_reservation: str | None = None,
    ) -> None:
        def pick(value, key):
            return META_SCHEDULER[key] if value is None else value

        self.total_prbs = SIMULATION["total_prbs"] if total_prbs is None else total_prbs
        self.realloc_step = pick(reallocation_step, "reallocation_step")
        self.fairness_decay = pick(fairness_decay, "fairness_decay")
        self.min_prbs = pick(min_prbs, "min_prbs_per_slice")
        self.mode = pick(mode, "mode")
        self.early_warning = early_warning
        self.need_margin = pick(need_margin, "need_margin")
        self.throughput_reservation = pick(throughput_reservation, "throughput_reservation")
        if self.throughput_reservation not in ("backlogged", "always"):
            raise ValueError(f"bad throughput_reservation {self.throughput_reservation!r}")
        if self.mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {self.mode!r}")

        logger.info(
            "MetaScheduler │ mode=%s  total=%d  step=%d  decay=%.3f  floor=%d  "
            "early_warning=%s  margin=%d  tput_reservation=%s",
            self.mode, self.total_prbs, self.realloc_step, self.fairness_decay,
            self.min_prbs, self.early_warning, self.need_margin,
            self.throughput_reservation,
        )

    # ── public interface ──────────────────────────────────────────────

    def allocate(
        self,
        slices: Sequence[NetworkSlice],
        violations: Dict[str, bool],
        timestep: int,
    ) -> Dict[str, int]:
        """
        Decide PRB allocation for the next timestep and write it back to
        ``slice.allocated_prbs``.  Nothing else on the slices is modified,
        and the scheduler itself keeps no state between calls.
        """
        if self.mode == "legacy":
            return self._allocate_legacy(slices, violations, timestep)

        allocs = self.plan(slices, violations)
        for s in slices:
            s.allocated_prbs = allocs[s.name]
        logger.debug(
            "t=%d │ MetaSched │ final allocs: %s  (sum=%d)",
            timestep, allocs, sum(allocs.values()),
        )
        return allocs

    # ── fixed allocator ───────────────────────────────────────────────

    def requirement(self, s: NetworkSlice) -> int:
        """
        PRBs slice *s* needs next step to meet its SLA, assuming next-step
        arrivals equal this step's (``offered = queue + arrivals``).

        * latency  L: backlog after service ≤ L·P·r  ⇔  P ≥ offered / (r(1+L))
        * throughput T: serve min(T, offered)        ⇔  P ≥ min(T, offered) / r
          (``throughput_reservation="always"``: P ≥ T / r)
        * relaxed: 0
        A non-zero requirement gets ``need_margin`` extra PRBs.
        """
        offered = s.queue + s.arrivals
        r = s.service_rate
        if s.sla_type == "latency":
            need = math.ceil(offered / (r * (1.0 + s.sla_threshold)) - 1e-9)
        elif s.sla_type == "throughput":
            rate = s.sla_threshold
            if self.throughput_reservation == "backlogged":
                rate = min(rate, offered)
            need = math.ceil(rate / r - 1e-9)
        else:
            need = 0
        return need + self.need_margin if need > 0 else 0

    def triggered(self, s: NetworkSlice, violated: bool) -> bool:
        """Violated, or (early warning) latency above the warning level."""
        if violated:
            return True
        return (self.early_warning is not None and s.sla_type == "latency"
                and s.latency > self.early_warning * s.sla_threshold)

    def plan(
        self,
        slices: Sequence[NetworkSlice],
        violations: Dict[str, bool],
    ) -> Dict[str, int]:
        """Pure function of the slice state: the next allocation (fixed mode)."""
        names = [s.name for s in slices]
        old = {s.name: s.allocated_prbs for s in slices}
        target = {s.name: s.initial_prbs for s in slices}
        protect = {s.name: max(self.min_prbs, self.requirement(s)) for s in slices}
        trig = {s.name: self.triggered(s, violations.get(s.name, False)) for s in slices}

        desired: Dict[str, int] = {}
        lower: Dict[str, int] = {}
        for n in names:
            if violations.get(n, False):
                lower[n] = old[n]                       # never shrink a violated slice
            else:
                lower[n] = max(self.min_prbs, min(old[n], protect[n]))
            if trig[n]:
                desired[n] = max(old[n] + self.realloc_step, protect[n])
            else:
                desired[n] = max(protect[n], self._toward(old[n], target[n]))
            desired[n] = max(desired[n], lower[n])

        excess = sum(desired.values()) - self.total_prbs
        if excess > 0:
            # Cut in tiers: donor slack above requirement, then boost above
            # requirement, then anything above the lower bound.
            keep = {n: max(lower[n], protect[n]) for n in names}
            tiers = [
                {n: desired[n] - keep[n] if not trig[n] else 0 for n in names},
                {n: desired[n] - keep[n] if trig[n] else 0 for n in names},
            ]
            for caps in tiers:
                excess -= self._take(desired, caps, excess, names)
            caps = {n: desired[n] - lower[n] for n in names}
            excess -= self._take(desired, caps, excess, names)
            if excess > 0:  # unreachable: sum(lower) <= sum(old) == total
                raise RuntimeError(f"PRB budget infeasible by {excess}")
        elif excess < 0:
            surplus = -excess
            # First fill slices that sit below their initial share, then
            # spread the rest by initial shares.
            deficit = {n: max(0, target[n] - desired[n]) for n in names}
            surplus -= self._give(desired, deficit, surplus, names)
            if surplus > 0:
                self._give(desired, {n: target[n] for n in names}, surplus, names,
                           capped=False)
        return desired

    def _toward(self, current: int, target: int) -> int:
        """Move *current* toward *target* by max(1, ceil(decay·|diff|))."""
        diff = target - current
        if diff == 0 or self.fairness_decay <= 0:
            return current
        step = max(1, math.ceil(round(abs(diff) * self.fairness_decay, 9)))
        return current + int(math.copysign(min(step, abs(diff)), diff))

    @staticmethod
    def _apportion(amount: int, weights: Dict[str, int], names: Sequence[str]) -> Dict[str, int]:
        """Split integer *amount* ∝ *weights* (largest remainder, exact)."""
        total = sum(weights[n] for n in names)
        if amount <= 0 or total <= 0:
            return {n: 0 for n in names}
        share = {n: amount * weights[n] // total for n in names}
        order = sorted(names, key=lambda n: (-(amount * weights[n] % total), names.index(n)))
        for n in order[: amount - sum(share.values())]:
            share[n] += 1
        return share

    def _take(self, allocs, caps, amount, names) -> int:
        """Remove up to *amount* from *allocs*, at most caps[n] each."""
        amount = min(max(amount, 0), sum(caps.values()))
        cut = self._apportion(amount, caps, names)
        for n in names:
            allocs[n] -= cut[n]
        return amount

    def _give(self, allocs, weights, amount, names, capped=True) -> int:
        """Add up to *amount* to *allocs* ∝ weights (capped at weights[n])."""
        if capped:
            amount = min(amount, sum(weights.values()))
        add = self._apportion(amount, weights, names)
        for n in names:
            allocs[n] += add[n]
        return sum(add.values())

    # ── legacy allocator (initial commit, kept bit-for-bit) ───────────

    def _allocate_legacy(
        self,
        slices: Sequence[NetworkSlice],
        violations: Dict[str, bool],
        timestep: int,
    ) -> Dict[str, int]:
        """
        Decide PRB allocation for the next timestep (legacy algorithm).

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
