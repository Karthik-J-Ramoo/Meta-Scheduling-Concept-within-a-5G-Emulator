"""
comparison.py
=============
Multi-seed policy comparison (T5/T6) shared by tests and
``scripts/run_experiments.py``.

Policies, all on identical per-slice arrival traces for each seed:

    static            initial shares (40/40/20)
    meta_legacy       the meta-scheduler as shipped (LEGACY_META)
    hindsight_static  best grid split chosen on the evaluation trace itself
                      (an oracle; a lower bound for any fixed split)
    tuned_static      best grid split on the *tuning* seeds, evaluated on the
                      evaluation seeds (the non-oracle baseline)
    meta_fixed        the fixed meta-scheduler (cfg["meta"])
"""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence

import numpy as np

from config.experiments import LEGACY_META
from core.baselines import best_split, grid_evaluate, split_grid
from core.episode_metrics import summarize
from core.simulation import arrival_traces, run_episode

POLICY_VARIANTS = ("static", "meta_legacy", "hindsight_static", "tuned_static", "meta_fixed")

EVAL_SEEDS = tuple(range(400, 420))
TUNE_SEEDS = tuple(range(0, 20))


def _with_meta(cfg: dict, meta: dict) -> dict:
    out = copy.deepcopy(cfg)
    out["meta"] = dict(meta)
    return out


def tuned_split(cfg: dict, tune_seeds: Sequence[int] = TUNE_SEEDS, grid=None):
    grid = grid or split_grid(total=cfg["simulation"]["total_prbs"])
    return best_split([grid_evaluate(cfg, arrival_traces(cfg, s), grid) for s in tune_seeds])


def compare_policies(
    cfg: dict,
    eval_seeds: Sequence[int] = EVAL_SEEDS,
    tune_seeds: Sequence[int] = TUNE_SEEDS,
    policies: Sequence[str] = POLICY_VARIANTS,
    grid=None,
) -> List[dict]:
    """One row per (policy, seed): workload, policy, seed, split + KPIs."""
    grid = grid or split_grid(total=cfg["simulation"]["total_prbs"])
    tuned = tuned_split(cfg, tune_seeds, grid) if "tuned_static" in policies else None
    legacy_cfg = _with_meta(cfg, LEGACY_META)
    rows = []
    for seed in eval_seeds:
        traces = arrival_traces(cfg, seed)
        for policy in policies:
            split = None
            if policy == "static":
                hist = run_episode(cfg, seed, "static", traces=traces)
            elif policy == "meta_legacy":
                hist = run_episode(legacy_cfg, seed, "meta", traces=traces)
            elif policy == "meta_fixed":
                hist = run_episode(cfg, seed, "meta", traces=traces)
            else:
                split = tuned if policy == "tuned_static" else \
                    best_split([grid_evaluate(cfg, traces, grid)])
                hist = run_episode(cfg, seed, "static", static_split=split, traces=traces)
            if split is None:
                split = tuple(m.allocated_prbs for m in hist[0])
            rows.append({"workload": cfg["name"], "policy": policy, "seed": seed,
                         "split": "/".join(map(str, split)), **summarize(hist)})
    return rows


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile bootstrap (1 − alpha) CI; seeded, deterministic."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def aggregate(rows: Sequence[dict], metrics: Sequence[str]) -> Dict[str, Dict[str, tuple]]:
    """``{policy: {metric: (mean, lo, hi)}}`` over seeds."""
    out: Dict[str, Dict[str, tuple]] = {}
    for policy in dict.fromkeys(r["policy"] for r in rows):
        sel = [r for r in rows if r["policy"] == policy]
        out[policy] = {m: bootstrap_ci([r[m] for r in sel]) for m in metrics}
    return out
