"""
traffic_generator.py
====================
Stochastic traffic source for network slices.

Design rationale
----------------
* Uses a **Poisson** base arrival process with optional **burst overlay**.
  At each timestep a Bernoulli trial decides whether a "burst event" occurs;
  if so, the Poisson λ is temporarily multiplied by ``burst_mult``.
* The generator is **per-slice** – each ``NetworkSlice`` gets its own instance
  constructed from ``slice.traffic_params``.
* The class is deliberately stateless across timesteps (no memory / correlation)
  so it can be trivially swapped for a Markov-modulated or trace-driven source
  later.
* Accepts a ``numpy.random.Generator`` for reproducibility.  Give each slice
  its own stream (``SeedSequence(seed).spawn``) to make arrivals independent
  of the policy and of the other slices; the legacy setup shares one RNG.
* Optional ``phases`` in ``traffic_params`` – a list of ``[start_step, mult]``
  – scale ``mean_arrival`` from ``start_step`` on (regime shifts).
* ``TraceTraffic`` replays recorded arrivals (e.g. a metrics.csv column).

Extensibility hook
------------------
Replace this class with a file-driven or GAN-driven traffic model by
implementing the same ``generate(timestep) -> int`` interface.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from core.slice import NetworkSlice


logger = logging.getLogger(__name__)


class TrafficGenerator:
    """
    Generate packet arrivals for a single network slice.

    Parameters
    ----------
    nslice : NetworkSlice
        The slice this generator feeds.
    rng : numpy.random.Generator
        Shared RNG for reproducibility.
    """

    def __init__(self, nslice: NetworkSlice, rng: np.random.Generator) -> None:
        self.slice = nslice
        self.rng = rng

        params: dict[str, Any] = nslice.traffic_params
        self.mean_arrival: float = params["mean_arrival"]
        self.burst_prob: float = params["burst_prob"]
        self.burst_mult: float = params["burst_mult"]
        self.phases: list = sorted(params.get("phases", []))

        logger.debug(
            "TrafficGen[%s] │ λ=%.1f  burst_p=%.2f  burst_x=%.1f",
            nslice.name, self.mean_arrival, self.burst_prob, self.burst_mult,
        )

    # ── public interface ──────────────────────────────────────────────

    def generate(self, timestep: int) -> int:
        """
        Sample the number of packet arrivals for *timestep*.

        Returns
        -------
        int
            Non-negative number of arriving packets.
        """
        # Decide whether this timestep experiences a traffic burst.
        is_burst: bool = self.rng.random() < self.burst_prob
        base_lambda = self.mean_arrival * self.phase_multiplier(timestep)
        effective_lambda = base_lambda * self.burst_mult if is_burst else base_lambda

        arrivals: int = int(self.rng.poisson(lam=effective_lambda))

        if is_burst:
            logger.debug(
                "t=%d │ TrafficGen[%s] │ BURST │ λ_eff=%.1f → arrivals=%d",
                timestep, self.slice.name, effective_lambda, arrivals,
            )

        return arrivals

    def phase_multiplier(self, timestep: int) -> float:
        """Multiplier on ``mean_arrival`` of the phase active at *timestep*."""
        mult = 1.0
        for start, m in self.phases:
            if timestep >= start:
                mult = m
        return mult


class TraceTraffic:
    """Replay a recorded arrival sequence; same interface as TrafficGenerator."""

    def __init__(self, arrivals) -> None:
        self.arrivals = [int(a) for a in arrivals]

    def generate(self, timestep: int) -> int:
        return self.arrivals[timestep]
