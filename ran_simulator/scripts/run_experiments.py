"""
run_experiments.py
==================
Regenerate every number and figure for the paper from the code.

    cd ran_simulator
    python scripts/run_experiments.py            # ~3 min

Writes into ``results/``:
    summary.csv          one row per (workload, policy, seed)
    paper_tables.md      Tables 2, 3, 5, 6, 7 + T1/T2/T4/T5/T6 tables,
                         and deviations from the replica's expected values
    figures/*.png        SLA timeline, PRB dynamics, queue length and
                         cumulative violations, static vs meta, same trace
    configs/*.json       the exact config of every workload run

All runs are seeded; re-running gives identical files.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config.experiments import FIXED_META, LEGACY_META, get_config  # noqa: E402
from core.baselines import (grid_evaluate, phase_requirements,  # noqa: E402
                            split_grid, static_safe_reservation)
from core.comparison import (EVAL_SEEDS, TUNE_SEEDS, aggregate,  # noqa: E402
                             compare_policies)
from core.episode_metrics import (prb_demand, saturation, summarize,  # noqa: E402
                                  violation_events)
from core.simulation import run_episode  # noqa: E402

OUT = ROOT / "results"
FIG = OUT / "figures"
CFG_DIR = OUT / "configs"
PAPER_CSV = ROOT / "tests" / "data" / "metrics_csv_original.csv"

T5_WORKLOADS = ["csv_fitted", "feasible", "regime", "anticorr", "checked_in_ps",
                "csv_repro", "t6_reliability"]
T5_METRICS = ["viol_total", "viol_URLLC", "viol_eMBB", "viol_IoT", "drops_total",
              "drops_IoT", "served_per_ts_IoT", "avg_latency_IoT", "prb_changes"]
POLICY_LABEL = {
    "static": "static 40/40/20",
    "meta_legacy": "current meta (legacy)",
    "hindsight_static": "hindsight best static",
    "tuned_static": "tuned static (seeds 0-19)",
    "meta_fixed": "new meta (fixed)",
}

# Expected values reported by the external replica (mean violations/run).
EXPECTED = {
    ("csv_fitted", "static"): 8.05, ("csv_fitted", "meta_legacy"): 3.40,
    ("csv_fitted", "hindsight_static"): 0.05,
    ("feasible", "static"): 9.55, ("feasible", "meta_legacy"): 3.00,
    ("feasible", "hindsight_static"): 0.0,
    ("regime", "static"): 27.45, ("regime", "meta_legacy"): 3.15,
    ("regime", "hindsight_static"): 0.0,
    ("anticorr", "static"): 73.1, ("anticorr", "meta_legacy"): 3.3,
    ("anticorr", "hindsight_static"): 0.0,
    ("checked_in_ps", "static"): 1.03, ("checked_in_ps", "meta_legacy"): 10.70,
}
EXPECTED_IOT_DROPS = {("csv_fitted", "static"): 775, ("csv_fitted", "meta_legacy"): 1506,
                      ("csv_fitted", "hindsight_static"): 2563}
RNG_REASON = ("per-slice SeedSequence streams draw a different sample of the same "
              "workload than the replica's generator")

# Categorical slots 1-3 of the reference palette (fixed order, one per policy).
COLORS = {"static": "#2a78d6", "meta_legacy": "#eb6834", "meta_fixed": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


# ── helpers ───────────────────────────────────────────────────────────

def recorded_traces():
    with open(PAPER_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = list(dict.fromkeys(r["slice_name"] for r in rows))
    return [[int(r["arrivals"]) for r in rows if r["slice_name"] == n] for n in names], rows


def with_meta(cfg, meta):
    cfg = copy.deepcopy(cfg)
    cfg["meta"] = dict(meta)
    return cfg


def save_config(cfg, suffix=""):
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = CFG_DIR / f"{cfg['name']}{suffix}.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def fmt_ci(v, digits=2):
    mean, lo, hi = v
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def series(history, name, attr):
    return [getattr(m, attr) for step in history for m in step if m.slice_name == name]


# ── sections ──────────────────────────────────────────────────────────

def section_t0_t1(md, paper):
    traces, rows = recorded_traces()
    cfg = get_config("csv_original")
    md.append("## T0 / T1 – paper trace\n")
    md.append(f"Paper CSV: `{PAPER_CSV.relative_to(ROOT).as_posix()}` ({len(rows)} rows), produced by "
              "config `csv_original` (recovered from `Meta_Scheduling_Test/`: URLLC "
              "λ=32 p_b=0.18 m=3.5, eMBB 48/0.20/2.8, IoT 18/0.12/2.4, SLA 2.0 ms / 38 pkts, "
              "legacy meta floor 5, seed 42). `tests/test_replay.py` replays all 600 rows "
              "with 0 mismatches and regenerates the CSV byte-for-byte.\n")
    out = []
    for policy, label in [("static", "static"), ("meta_legacy", "meta_legacy"),
                          ("meta_fixed", "meta_fixed")]:
        c = with_meta(cfg, LEGACY_META if policy == "meta_legacy" else FIXED_META)
        hist = run_episode(c, policy="static" if policy == "static" else "meta",
                           traces=traces)
        paper[policy] = hist
        ev = violation_events(hist)
        out.append([POLICY_LABEL[policy], sum(map(len, ev.values())),
                    ev["URLLC"] or "–", ev["eMBB"] or "–"])
    md.append("Counterfactual on the recorded arrivals (identical traffic for all policies):\n")
    md.append(md_table(["policy", "violations", "URLLC at t", "eMBB at t"], out))
    cfg = get_config("csv_repro")
    fresh = [[p, sum(map(len, violation_events(run_episode(cfg, 42, p)).values()))]
             for p in ("static", "meta")]
    md.append("\nFresh per-slice sample of the same workload (`csv_repro`, seed 42, "
              "new meta): " + ", ".join(f"{p} = {v}" for p, v in fresh) + ".\n")


def section_t2(md):
    traces, _ = recorded_traces()
    cfg = get_config("csv_original")
    grid = split_grid()
    extra = [(46, 44, 10), (49, 42, 9), (45, 40, 15), (50, 40, 10), (40, 40, 20)]
    ev = grid_evaluate(cfg, traces, grid + [s for s in extra if s not in grid])
    zero = sorted(s for s, (v, _) in ev.items() if v == 0)
    off_grid = [s for s in zero if s not in grid]
    md.append("## T2 – fixed-split sweep on the paper trace (static policy)\n")
    md.append(f"Grid u, e ∈ 20..70 step 3, i = 100−u−e ≥ 5: {len(grid)} splits; "
              f"{sum(1 for s in grid if ev[s][0] == 0)} of them reach 0 violations.\n")
    rows = [["/".join(map(str, s)), ev[s][0], ev[s][1]] for s in extra]
    best = sorted(ev, key=lambda s: (ev[s][0], ev[s][1], s))[:10]
    rows += [["/".join(map(str, s)) + " (grid best)", ev[s][0], ev[s][1]] for s in best]
    md.append(md_table(["split u/e/i", "violations", "drops"], rows))
    md.append("\nThe paper's 57.14 % = (7 − 3)/7 compares the meta-scheduler with the "
              f"40/40/20 split only; {len(zero) - len(off_grid)} grid splits and "
              f"{len(off_grid)} listed off-grid splits give 0 violations on the same trace.\n")


def section_t4(md):
    md.append("## T4 – load audit\n")
    md.append("Mean PRB demand = λ(1 − p_b + p_b·m)/r (phase-weighted); saturation = "
              "share of steps with overflow drops, seed 42, new meta.\n")
    rows = []
    for name in ["checked_in_ps", "csv_repro", "csv_fitted", "feasible", "regime",
                 "anticorr", "t6_reliability"]:
        cfg = get_config(name)
        d = prb_demand(cfg)
        sat = saturation(run_episode(cfg, 42, "meta"))
        flag = "OVERLOADED" if d["mean_sum"] > 100 or max(sat.values()) > 0.2 else "ok"
        rows.append([name] + [f"{d['mean'][n]:.1f}" for n in ("URLLC", "eMBB", "IoT")]
                    + [f"{d['mean_sum']:.1f}", f"{d['peak_sum']:.1f}",
                       " / ".join(f"{sat[n]:.0%}" for n in ("URLLC", "eMBB", "IoT")), flag])
    md.append(md_table(["workload", "URLLC", "eMBB", "IoT", "sum", "peak sum",
                        "saturation U/E/I", "audit"], rows))
    md.append("")


def section_reservations(md):
    md.append("## Static-safe reservations\n")
    md.append("PRBs at which an SLA cannot be violated under any traffic "
              "(`core.baselines.static_safe_reservation`). URLLC: the queue cap C = 200 "
              "bounds the backlog, so latency ≤ (C − ⌊P·r⌋)/(P·r) ≤ 2.0 for P ≥ 45 "
              "(P = 44 gives 2.03). eMBB demand-aware: ⌊P·r⌋ ≥ 38. When these sum to "
              "≤ 100 a fixed split is violation-free on every trace, which is why the "
              "hindsight and tuned static baselines reach ~0 in W1–W4; the tuned split "
              "47/38/15 is the first lattice point above (45, 38).\n")
    rows = []
    for name in ["feasible", "csv_fitted", "t6_reliability"]:
        cfg = get_config(name)
        r = [static_safe_reservation(c, cfg["sla_mode"]) for c in cfg["slices"]]
        rows.append([name, cfg["sla_mode"]] + ["none (load-dependent)" if x is None else x
                                               for x in r])
    md.append(md_table(["workload", "SLA mode", "URLLC", "eMBB", "IoT"], rows))
    md.append("")


def section_reservation_choice(md):
    md.append("## Choice of `throughput_reservation` (tuning seeds 0..19, W2–W4)\n")
    md.append("Adopted after the first T5 pass showed the fixed allocator cutting eMBB "
              "to its last-step demand. Chosen on tuning seeds only.\n")
    rows, totals = [], {}
    for mode in ("backlogged", "always"):
        for name in ("feasible", "regime", "anticorr"):
            cfg = get_config(name)
            cfg["meta"]["throughput_reservation"] = mode
            a = aggregate(compare_policies(cfg, eval_seeds=TUNE_SEEDS,
                                           policies=["meta_fixed"]), ["viol_total"])
            mean = a["meta_fixed"]["viol_total"][0]
            totals[mode] = totals.get(mode, 0) + mean
            rows.append([mode, name, f"{mean:.2f}"])
    md.append(md_table(["reservation", "workload", "mean violations"], rows))
    md.append("\nSummed: " + ", ".join(f"{k} {v:.2f}" for k, v in totals.items())
              + f" → `{FIXED_META['throughput_reservation']}`.\n")


def section_t5(md, summary_rows):
    md.append("## T5 / T6 – multi-seed comparison\n")
    md.append(f"Seeds {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]} (20 × 200 steps), identical "
              "per-slice traces for every policy. Mean [95 % bootstrap CI]. "
              "Tuned static is tuned on seeds 0..19 only.\n")
    aggs, deviations = {}, []
    for name in T5_WORKLOADS:
        cfg = get_config(name)
        save_config(cfg)
        save_config(with_meta(cfg, LEGACY_META), "__meta_legacy")
        rows = compare_policies(cfg)
        summary_rows += rows
        agg = aggregate(rows, T5_METRICS)
        aggs[name] = agg
        splits = {p: sorted({r["split"] for r in rows if r["policy"] == p}) for p in agg}
        md.append(f"### {name} – {cfg['description']}\n")
        table = []
        for p, a in agg.items():
            if p.startswith("meta"):
                sp = "dynamic"
            else:
                sp = splits[p][0] if len(splits[p]) == 1 else f"{len(splits[p])} splits"
            table.append([POLICY_LABEL[p], sp, fmt_ci(a["viol_total"]),
                          f"{a['viol_URLLC'][0]:.2f} / {a['viol_eMBB'][0]:.2f} / "
                          f"{a['viol_IoT'][0]:.2f}",
                          f"{a['drops_total'][0]:.0f}", f"{a['drops_IoT'][0]:.0f}",
                          f"{a['served_per_ts_IoT'][0]:.2f}",
                          f"{a['avg_latency_IoT'][0]:.2f}", f"{a['prb_changes'][0]:.1f}"])
            exp = EXPECTED.get((name, p))
            if exp is not None and abs(a["viol_total"][0] - exp) > 0.005:
                inside = a["viol_total"][1] <= exp <= a["viol_total"][2]
                deviations.append(f"{name} / {p}: {fmt_ci(a['viol_total'])} violations vs "
                                  f"expected {exp} ({'inside' if inside else 'OUTSIDE'} "
                                  f"our CI) – {RNG_REASON}.")
            exp = EXPECTED_IOT_DROPS.get((name, p))
            if exp is not None:
                deviations.append(f"{name} / {p}: IoT drops {a['drops_IoT'][0]:.0f} vs "
                                  f"expected {exp} – {RNG_REASON}.")
        md.append(md_table(["policy", "split", "violations", "U / E / I", "drops",
                            "IoT drops", "IoT served/ts", "IoT lat", "PRB changes"], table))
        md.append("")
    return aggs, deviations


def section_t6(md, aggs):
    cfg = get_config("t6_reliability")
    md.append("### T6 – per-phase SLA requirements (`t6_reliability`)\n")
    ph = phase_requirements(cfg)
    rows = [[f"{p['start']}–{p['end'] - 1}"] + [f"{p['req'][n]:.1f}" for n in p["req"]]
            + [f"{p['sum']:.1f}"] for p in ph]
    peak = {n: max(p["req"][n] for p in ph) for n in ph[0]["req"]}
    rows.append(["per-slice peak"] + [f"{v:.1f}" for v in peak.values()]
                + [f"{sum(peak.values()):.1f}"])
    md.append(md_table(["steps", "URLLC", "eMBB", "IoT", "sum"], rows))
    a = aggs["t6_reliability"]
    cond_i = all(a[p]["viol_total"][1] > 0 for p in ("hindsight_static", "tuned_static"))
    cond_ii = a["meta_fixed"]["viol_total"][2] < min(
        a[p]["viol_total"][1] for p in ("hindsight_static", "tuned_static"))
    md.append(f"\nPre-registered criteria: (i) best static CI lower bound > 0 → "
              f"**{'met' if cond_i else 'not met'}**; (ii) meta CI upper bound below both "
              f"static lower bounds → **{'met' if cond_ii else 'not met'}** "
              f"(new meta {fmt_ci(a['meta_fixed']['viol_total'])} vs best static "
              f"{fmt_ci(a['tuned_static']['viol_total'])}).\n")


def section_paper_tables(md, paper):
    cfg = get_config("csv_original")
    md.append("## Table 2 – traffic and SLA parameters\n")
    rows = []
    for name in ["csv_original", "csv_fitted", "feasible", "regime", "anticorr",
                 "t6_reliability", "checked_in"]:
        c = get_config(name)
        for s in c["slices"]:
            tp = s["traffic_params"]
            sla = f"{s['sla_type']} {s['sla_threshold']:g}"
            if s.get("sla_drop_cap") is not None:
                sla += f", drops ≤ {s['sla_drop_cap']:g}"
            ph = ", ".join(f"t≥{a}: ×{m:g}" for a, m in tp.get("phases", [])) or "–"
            rows.append([name, s["name"], tp["mean_arrival"], tp["burst_prob"],
                         tp["burst_mult"], ph, s["service_rate"], s["queue_capacity"],
                         s["initial_prbs"], sla, c["sla_mode"]])
    md.append(md_table(["config", "slice", "λ", "p_b", "m", "phases", "r", "C",
                        "init PRBs", "SLA", "SLA mode"], rows))

    md.append("\n## Table 3 – meta-scheduler parameters (as in the code)\n")
    keys = ["mode", "reallocation_step", "fairness_decay", "min_prbs_per_slice",
            "need_margin", "early_warning", "throughput_reservation"]
    note = {"fairness_decay": "fixed: max(1, ⌈0.02·|diff|⌉) PRBs/step; legacy: "
                              "int(round(0.02·diff)), 0 for |diff| < 25",
            "min_prbs_per_slice": "paper Table 3 states 10; the paper CSV run used 5 "
                                  "(IoT sits at 9 PRBs from t=43)"}
    md.append(md_table(["parameter", "new meta (config.META_SCHEDULER)",
                        "legacy (paper CSV run)", "note"],
                       [[k, FIXED_META.get(k, "–"), LEGACY_META.get(k, "n/a"),
                         note.get(k, "")] for k in keys]))

    md.append("\n## Table 5 – per-slice performance, paper trace (seed 42)\n")
    rows = []
    for policy in ("meta_legacy", "meta_fixed"):
        hist = paper[policy]
        ev = violation_events(hist)
        for n in ("URLLC", "eMBB", "IoT"):
            lat, tput = series(hist, n, "latency"), series(hist, n, "throughput")
            q, d = series(hist, n, "queue_length"), series(hist, n, "dropped")
            rows.append([POLICY_LABEL[policy], n, f"{sum(lat) / len(lat):.3f}",
                         f"{sum(tput) / len(tput):.2f}", f"{sum(q) / len(q):.1f}",
                         sum(d), len(ev[n])])
    md.append(md_table(["policy", "slice", "avg latency", "avg throughput", "avg queue",
                        "drops", "violations"], rows))

    md.append("\n## Table 6 – static vs meta on the same trace (paper trace, seed 42)\n")
    kpis = {p: summarize(paper[p]) for p in ("static", "meta_legacy", "meta_fixed")}
    base = kpis["static"]["viol_total"]
    rows = []
    for label, key in [("violations (total)", "viol_total"), ("URLLC violations", "viol_URLLC"),
                       ("eMBB violations", "viol_eMBB"), ("total drops", "drops_total"),
                       ("IoT drops", "drops_IoT"), ("IoT served/ts", "served_per_ts_IoT"),
                       ("IoT avg latency", "avg_latency_IoT"), ("PRB changes", "prb_changes")]:
        rows.append([label] + [f"{kpis[p][key]:.2f}" if isinstance(kpis[p][key], float)
                               else kpis[p][key] for p in kpis])
    rows.append(["violation reduction vs static"] + ["–"] + [
        f"{(base - kpis[p]['viol_total']) / base:.2%}" if base else "n/a"
        for p in ("meta_legacy", "meta_fixed")])
    md.append(md_table(["metric", "static 40/40/20", "current meta (legacy)",
                        "new meta (fixed)"], rows))
    md.append("\nThe static column is the same-trace counterfactual. 40/40/20 is one of "
              "184 fixed splits; T2 shows many give 0 violations on this trace.\n")

    md.append("## Table 7 – SLA violation events (paper trace)\n")
    th = {s["name"]: s["sla_threshold"] for s in cfg["slices"]}
    rows = []
    for policy in ("static", "meta_legacy", "meta_fixed"):
        for step in paper[policy]:
            for m in step:
                if not m.sla_satisfied:
                    val = m.latency if m.sla_type == "latency" else m.throughput
                    rows.append([POLICY_LABEL[policy], m.timestep, m.slice_name,
                                 f"{val:.4g}", th[m.slice_name], m.allocated_prbs,
                                 m.queue_length])
    md.append(md_table(["policy", "t", "slice", "value", "threshold", "PRBs", "queue"],
                       rows or [["–"] * 7]))
    md.append("")


# ── figures ───────────────────────────────────────────────────────────

def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _legend_above(fig, axes):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=8)


def make_figures(runs, tag, cfg, labels=None):
    """runs: {policy: history}; one PNG per figure type."""
    FIG.mkdir(parents=True, exist_ok=True)
    label = {**POLICY_LABEL, **(labels or {})}
    names = [s["name"] for s in cfg["slices"]]
    th = {s["name"]: (s["sla_type"], s["sla_threshold"]) for s in cfg["slices"]}

    def lines(ax, attr, name):
        for p, hist in runs.items():
            ax.plot(series(hist, name, attr), color=COLORS[p], linewidth=1.6,
                    label=label[p])

    # SLA timeline: one panel per slice with an SLA
    sla_slices = [n for n in names if th[n][0] != "relaxed"]
    fig, axes = plt.subplots(len(sla_slices), 1, figsize=(7, 2.4 * len(sla_slices)),
                             sharex=True)
    for ax, n in zip(axes, sla_slices):
        kind, thr = th[n]
        lines(ax, "latency" if kind == "latency" else "served", n)
        ax.axhline(thr, color=MUTED, linestyle="--", linewidth=1)
        ax.text(1.0, thr, f" SLA {thr:g}", color=MUTED, fontsize=8, va="bottom",
                transform=ax.get_yaxis_transform(), ha="right")
        _style(ax, f"{n} {'latency' if kind == 'latency' else 'served pkts/ts'}",
               "ms" if kind == "latency" else "pkts/ts")
    _legend_above(fig, axes)
    axes[-1].set_xlabel("timestep", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / f"{tag}_sla_timeline.png", dpi=200)
    plt.close(fig)

    for attr, fname, ylabel in [("allocated_prbs", "prb_dynamics", "PRBs"),
                                ("queue_length", "queue_length", "packets")]:
        fig, axes = plt.subplots(len(names), 1, figsize=(7, 2.2 * len(names)), sharex=True)
        for ax, n in zip(axes, names):
            lines(ax, attr, n)
            _style(ax, n, ylabel)
        _legend_above(fig, axes)
        axes[-1].set_xlabel("timestep", fontsize=9, color=MUTED)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(FIG / f"{tag}_{fname}.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 2.8))
    for p, hist in runs.items():
        cum, total = [], 0
        for step in hist:
            total += sum(1 for m in step if not m.sla_satisfied)
            cum.append(total)
        ax.step(range(len(cum)), cum, where="post", color=COLORS[p], linewidth=1.8,
                label=label[p])
        ax.annotate(f"{cum[-1]}", (len(cum) - 1, cum[-1]), color=INK, fontsize=8,
                    xytext=(4, 0), textcoords="offset points", va="center")
    _style(ax, "Cumulative SLA violations (all slices)", "violations")
    ax.set_xlabel("timestep", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"{tag}_cumulative_violations.png", dpi=200)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────

def main():
    logging.disable(logging.CRITICAL)   # per-violation warnings dominate run time
    OUT.mkdir(exist_ok=True)
    md = ["# Paper tables (generated by scripts/run_experiments.py)\n",
          "Do not edit by hand; re-run the script.\n"]
    paper, summary_rows = {}, []

    section_t0_t1(md, paper)
    section_t2(md)
    section_t4(md)
    section_reservations(md)
    section_reservation_choice(md)
    aggs, deviations = section_t5(md, summary_rows)
    section_t6(md, aggs)
    section_paper_tables(md, paper)

    md.append("## Deviations from the replica's expected values\n")
    md.append("\n".join(f"- {d}" for d in deviations) or "None.")
    md.append("")

    # summary.csv: paper-trace runs + all multi-seed runs
    for p, hist in paper.items():
        summary_rows.insert(0, {"workload": "csv_original_trace", "policy": p, "seed": 42,
                                "split": "/".join(str(m.allocated_prbs) for m in hist[0]),
                                **summarize(hist)})
    fields = list(summary_rows[0])
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    for name in ("csv_original", "checked_in"):
        save_config(get_config(name))
    save_config(with_meta(get_config("csv_original"), FIXED_META), "__meta_fixed")
    make_figures(paper, "paper_trace", get_config("csv_original"))
    t6 = get_config("t6_reliability")
    make_figures({"static": run_episode(t6, 400, "static", static_split=(47, 38, 15)),
                  "meta_legacy": run_episode(with_meta(t6, LEGACY_META), 400, "meta"),
                  "meta_fixed": run_episode(t6, 400, "meta")},
                 "t6_seed400", t6, labels={"static": "tuned static 47/38/15"})

    (OUT / "paper_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'paper_tables.md'}, {OUT / 'summary.csv'} "
          f"({len(summary_rows)} rows), {len(list(FIG.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
