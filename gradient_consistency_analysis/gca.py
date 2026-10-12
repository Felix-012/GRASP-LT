#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCA CSV Analyzer + Figure Generator (aligned to the smallest step count across runs)

What it does:
  1) Loads multiple GCA CSVs (one per run/baseline).
  2) Selects the "expert-active" rows (or sensible fallbacks for legacy/no-router CSVs).
  3) Aligns all runs to a common step range:
        common_max_step = min(max_step_per_run)
     and keeps only step <= common_max_step for cross-run comparisons.
  4) Generates:
      - Run-level plots (distributions + timeseries means)
      - Per-expert plots (mean±std over steps)
      - Per-expert purity↔gradient correlation plot (requested)

Outputs:
  - Figures saved as PNG and PDF into --outdir
  - Tables:
      * per_run_step_stats.csv
      * per_expert_summary.csv
      * per_expert_purity_corr.csv
      * summary_A_metrics.csv
      * summary_B_correlations_per_expert.csv

Usage:
  python analyze_gca_csvs_full.py \
    --csvs ours.csv mocle.csv fixed.csv norouter.csv \
    --names ours mocle fixed no_router \
    --outdir gca_figs \
    --min_active_size 8 \
    --only_routed_forward

Notes:
  - For per-expert plots, runs that don't have experts (e.g., no-router) will be skipped automatically.
  - Purity correlation requires a varying purity across steps for each expert; otherwise correlation is NaN.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def analyzed_expert_series(df: pd.DataFrame) -> pd.Series:
    if "analyzed_expert" in df.columns:
        return df["analyzed_expert"].astype("Int64").astype(float).fillna(-1).astype(int)
    if "expert" in df.columns:
        return df["expert"].astype("Int64").astype(float).fillna(-1).astype(int)
    return pd.Series([-1] * len(df), index=df.index, dtype=int)


def select_active_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Pick rows used for expert-level comparisons.

    Preference order:
      1) subset_definition=="active" & analyzed_param_group=="expert_params"      (new script)
      2) subset_definition=="routed" & analyzed_param_group=="expert_params"      (legacy)
      3) subset_definition=="all"   & analyzed_param_group=="trainable_params"   (no-router)
      4) analyzed_param_group=="expert_params"                                    (fallback)
    """
    subset_def = safe_col(df, "subset_definition", "").astype(str)
    param_group = safe_col(df, "analyzed_param_group", "").astype(str)

    m1 = (subset_def == "active") & (param_group == "expert_params")
    if m1.any():
        return df.loc[m1].copy(), "expert_active"

    m2 = (subset_def == "routed") & (param_group == "expert_params")
    if m2.any():
        return df.loc[m2].copy(), "expert_routed_legacy"

    m3 = (subset_def == "all") & (param_group == "trainable_params")
    if m3.any():
        return df.loc[m3].copy(), "no_router_all"

    m4 = (param_group == "expert_params")
    if m4.any():
        return df.loc[m4].copy(), "expert_params_fallback"

    raise ValueError("Could not identify suitable expert-level rows in this CSV.")


def savefig(outpath_no_ext: Path) -> None:
    plt.tight_layout()
    plt.savefig(outpath_no_ext.with_suffix(".png"), dpi=200)
    plt.savefig(outpath_no_ext.with_suffix(".pdf"))
    plt.close()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    if np.allclose(np.nanstd(x), 0.0) or np.allclose(np.nanstd(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return pearson(xr, yr)


def pretty(metric: str):
    if metric == "intra_cos":
        return "Cosine Similarity"
    if metric == "evr_all":
        return "EVR"
    if metric == "conflict_all":
        return "Conflict"
    return metric


# -----------------------------
# Plot helpers
# -----------------------------
def plot_box_by_run(df: pd.DataFrame, metric: str, outdir: Path, title: str, fname: str) -> None:
    if metric not in df.columns:
        return
    runs = list(df["run"].unique())
    data, labels = [], []
    for r in runs:
        vals = df.loc[df["run"] == r, metric].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(r)

    if not data:
        return

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(title)
    plt.ylabel(pretty(metric))
    savefig(outdir / fname)


def plot_timeseries_mean(df: pd.DataFrame, metric: str, outdir: Path, title: str, fname: str) -> None:
    if metric not in df.columns or "step" not in df.columns:
        return

    plt.figure()
    for run, dfr in df.groupby("run"):
        g = (
            dfr.dropna(subset=[metric])
               .groupby("step", as_index=False)[metric]
               .mean()
               .sort_values("step")
        )
        if len(g) == 0:
            continue
        plt.plot(g["step"].to_numpy(), g[metric].to_numpy(), label=str(run))

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(metric)
    plt.legend()
    savefig(outdir / fname)


def plot_per_expert_errorbar(per_expert: pd.DataFrame, mean_col: str, std_col: str,
                             outdir: Path, title: str, fname: str) -> None:
    if mean_col not in per_expert.columns:
        return

    plt.figure()
    for run, dfr in per_expert.groupby("run"):
        dfr = dfr.sort_values("analyzed_expert")
        x = dfr["analyzed_expert"].to_numpy()
        y = dfr[mean_col].to_numpy(dtype=float)
        yerr = dfr[std_col].to_numpy(dtype=float) if std_col in dfr.columns else None
        plt.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", label=str(run))

    plt.title(title)
    plt.xlabel("expert id")
    plt.ylabel(mean_col)
    plt.legend()
    savefig(outdir / fname)


def plot_scatter_per_expert(per_expert: pd.DataFrame, xcol: str, ycol: str,
                            outdir: Path, title: str, fname: str) -> pd.DataFrame:
    if xcol not in per_expert.columns or ycol not in per_expert.columns:
        return pd.DataFrame()

    corr_rows = []
    plt.figure()
    for run, dfr in per_expert.groupby("run"):
        x = dfr[xcol].to_numpy(dtype=float)
        y = dfr[ycol].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            continue
        plt.scatter(x[m], y[m], s=40, alpha=0.8, label=str(run))
        corr_rows.append(dict(
            run=str(run),
            x=xcol,
            y=ycol,
            n=int(m.sum()),
            pearson=pearson(x[m], y[m]),
            spearman=spearman(x[m], y[m]),
        ))

    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.legend()
    savefig(outdir / fname)

    return pd.DataFrame(corr_rows)


def plot_corr_per_expert_line(per_expert_corr: pd.DataFrame, corr_col: str,
                              outdir: Path, title: str, fname: str) -> None:
    """
    Requested plot:
      For each run and expert, show correlation (across steps) between purity and a gradient metric.
      x-axis: expert id
      y-axis: correlation coefficient
    """
    if corr_col not in per_expert_corr.columns:
        return
    plt.figure()
    for run, dfr in per_expert_corr.groupby("run"):
        dfr = dfr.sort_values("analyzed_expert")
        x = dfr["analyzed_expert"].to_numpy(dtype=int)
        y = dfr[corr_col].to_numpy(dtype=float)
        plt.plot(x, y, marker="o", linestyle="-", label=str(run))

    plt.title(title)
    plt.xlabel("expert id")
    plt.ylabel(corr_col)
    plt.legend()
    savefig(outdir / fname)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True, help="One or more GCA CSV files.")
    ap.add_argument("--names", nargs="+", required=True, help="Run names (same length as --csvs).")
    ap.add_argument("--outdir", required=True, help="Output directory for figures and summaries.")
    ap.add_argument("--min_active_size", type=int, default=2, help="Filter rows with active_size < this.")
    ap.add_argument("--only_routed_forward", action="store_true",
                    help="Keep only forward_definition=='routed' rows when available.")
    ap.add_argument("--purity_corr_metric", default="intra_cos",
                    choices=["intra_cos", "conflict_all", "evr_all"],
                    help="Metric to correlate with purity per expert across steps (requested plot).")
    ap.add_argument("--min_points_for_corr", type=int, default=5,
                    help="Minimum number of (step) points per expert to compute correlation.")
    args = ap.parse_args()

    if len(args.csvs) != len(args.names):
        raise SystemExit("ERROR: --csvs and --names must have the same length.")

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    active_by_run: Dict[str, pd.DataFrame] = {}
    modes: List[Dict[str, str]] = []

    # ---- Load & select active rows per run ----
    for csv_path_str, run_name in zip(args.csvs, args.names):
        csv_path = Path(csv_path_str)
        df = read_csv(csv_path)
        df["run"] = str(run_name)

        df = coerce_numeric(df, [
            "step", "batch_size", "active_size", "subset_size",
            "evr_all", "conflict_all", "evr_sub", "conflict_sub",
            "gc_paper", "mean_grad_cos", "intra_cos", "inter_cos",
            "purity", "label_entropy"
        ])

        if args.only_routed_forward and "forward_definition" in df.columns:
            df = df[df["forward_definition"].astype(str) == "routed"].copy()

        df_active, mode = select_active_rows(df)
        df_active["analyzed_expert"] = analyzed_expert_series(df_active)
        df_active["expert"] = df_active["analyzed_expert"]  # enforce consistent meaning

        if "active_size" in df_active.columns:
            df_active = df_active[df_active["active_size"].fillna(0).astype(int) >= int(args.min_active_size)].copy()

        if "step" not in df_active.columns:
            raise SystemExit(f"ERROR: selected active rows for run '{run_name}' have no 'step' column.")

        active_by_run[str(run_name)] = df_active
        modes.append(dict(run=str(run_name), mode=mode, source=str(csv_path)))

    # ---- Align to smallest step coverage across runs ----
    stats = []
    for run, dfr in active_by_run.items():
        steps = dfr["step"].dropna().astype(int)
        if steps.empty:
            raise SystemExit(f"ERROR: run '{run}' has no valid step values.")
        stats.append(dict(run=run, max_step=int(steps.max()), n_steps=int(steps.nunique())))
    stats_df = pd.DataFrame(stats).sort_values("max_step")
    stats_df.to_csv(outdir / "per_run_step_stats.csv", index=False)

    common_max_step = int(stats_df["max_step"].min())
    for run in list(active_by_run.keys()):
        dfr = active_by_run[run]
        active_by_run[run] = dfr[dfr["step"].astype(int) <= common_max_step].copy()

    modes_df = pd.DataFrame(modes)
    modes_df["common_max_step"] = common_max_step
    modes_df.to_csv(outdir / "run_modes.csv", index=False)

    # ---- Combine ----
    dfA = pd.concat(list(active_by_run.values()), ignore_index=True)
    dfA.to_csv(outdir / "aligned_active_rows.csv", index=False)

    # ---- Summary A: run-level pooled metrics ----
    summaryA = []
    for run, dfr in dfA.groupby("run"):
        for metric in ["intra_cos", "conflict_all", "evr_all"]:
            if metric not in dfr.columns:
                continue
            vals = dfr[metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            summaryA.append(dict(
                run=str(run),
                metric=metric,
                n=int(len(vals)),
                mean=float(np.mean(vals)),
                std=float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                median=float(np.median(vals)),
            ))
    pd.DataFrame(summaryA).to_csv(outdir / "summary_A_metrics.csv", index=False)

    # -----------------------------
    # Plot set A: run-level comparisons
    # -----------------------------
    plot_box_by_run(
        dfA, "intra_cos", outdir,
        title=f"Gradient Cosine Distribution",
        fname="A1_box_intra_cos_fig.pdf"
    )
    plot_box_by_run(
        dfA, "conflict_all", outdir,
        title=f"Conflict Fraction Distribution",
        fname="A2_box_conflict_all_fig.pdf"
    )
    plot_box_by_run(
        dfA, "evr_all", outdir,
        title=f"EVR Distribution",
        fname="A3_box_evr_all_fig.pdf"
    )

    plot_timeseries_mean(
        dfA, "intra_cos", outdir,
        title=f"(A4) Mean intra_cos over steps (mean across experts; aligned <= {common_max_step})",
        fname="A4_ts_mean_intra_cos"
    )
    plot_timeseries_mean(
        dfA, "conflict_all", outdir,
        title=f"(A5) Mean conflict_all over steps (mean across experts; aligned <= {common_max_step})",
        fname="A5_ts_mean_conflict_all"
    )
    plot_timeseries_mean(
        dfA, "evr_all", outdir,
        title=f"(A6) Mean evr_all over steps (mean across experts; aligned <= {common_max_step})",
        fname="A6_ts_mean_evr_all"
    )

    # -----------------------------
    # Per-expert summary (mean±std over steps)
    # -----------------------------
    # Only keep true experts for per-expert views
    dfE = dfA[dfA["analyzed_expert"].astype(int) >= 0].copy()

    agg_spec = {
        "n_rows": ("step", "count"),
        "n_steps": ("step", "nunique"),
    }
    if "active_size" in dfE.columns:
        agg_spec["active_size_mean"] = ("active_size", "mean")
        agg_spec["active_size_std"] = ("active_size", "std")
    if "purity" in dfE.columns:
        agg_spec["purity_mean"] = ("purity", "mean")
        agg_spec["purity_std"] = ("purity", "std")

    for m in ["intra_cos", "conflict_all", "evr_all"]:
        if m in dfE.columns:
            agg_spec[f"{m}_mean"] = (m, "mean")
            agg_spec[f"{m}_std"] = (m, "std")

    per_expert = (
        dfE.groupby(["run", "analyzed_expert"], as_index=False)
           .agg(**agg_spec)
           .sort_values(["run", "analyzed_expert"])
    )
    per_expert.to_csv(outdir / "per_expert_summary.csv", index=False)

    # -----------------------------
    # Plot set A (expert profiles)
    # -----------------------------
    plot_per_expert_errorbar(
        per_expert, "intra_cos_mean", "intra_cos_std",
        outdir,
        title="(A7) intra_cos per expert (mean±std over steps)",
        fname="A7_per_expert_intra_cos"
    )
    plot_per_expert_errorbar(
        per_expert, "conflict_all_mean", "conflict_all_std",
        outdir,
        title="(A8) conflict_all per expert (mean±std over steps)",
        fname="A8_per_expert_conflict_all"
    )
    plot_per_expert_errorbar(
        per_expert, "evr_all_mean", "evr_all_std",
        outdir,
        title="(A9) evr_all per expert (mean±std over steps)",
        fname="A9_per_expert_evr_all"
    )

    # -----------------------------
    # Plot set B: purity ↔ gradient relations (per expert, aggregated over steps)
    # -----------------------------
    corr_summary_rows = []
    if "purity_mean" in per_expert.columns:
        c1 = plot_scatter_per_expert(
            per_expert, "purity_mean", "intra_cos_mean",
            outdir,
            title="(B1) purity vs intra_cos (each point = one expert; aggregated over steps)",
            fname="B1_scatter_purity_vs_intra_per_expert"
        )
        c2 = plot_scatter_per_expert(
            per_expert, "purity_mean", "conflict_all_mean",
            outdir,
            title="(B2) purity vs conflict_all (each point = one expert; aggregated over steps)",
            fname="B2_scatter_purity_vs_conflict_per_expert"
        )
        c3 = plot_scatter_per_expert(
            per_expert, "purity_mean", "evr_all_mean",
            outdir,
            title="(B3) purity vs evr_all (each point = one expert; aggregated over steps)",
            fname="B3_scatter_purity_vs_evr_per_expert"
        )
        for d in [c1, c2, c3]:
            if len(d) > 0:
                corr_summary_rows.append(d)
    if corr_summary_rows:
        pd.concat(corr_summary_rows, ignore_index=True).to_csv(outdir / "summary_B_correlations_per_expert.csv", index=False)

    # -----------------------------
    # Requested plot: per-expert correlation across steps
    # -----------------------------
    # For each (run, expert): correlate purity(step) with metric(step), across steps.
    metric = args.purity_corr_metric
    per_expert_corr_rows = []

    if ("purity" in dfE.columns) and (metric in dfE.columns):
        for (run, e), dfr in dfE.groupby(["run", "analyzed_expert"]):
            x = dfr["purity"].to_numpy(dtype=float)
            y = dfr[metric].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]
            y = y[m]
            n = int(len(x))
            if n < int(args.min_points_for_corr):
                per_expert_corr_rows.append(dict(
                    run=str(run),
                    analyzed_expert=int(e),
                    n_points=n,
                    pearson=float("nan"),
                    spearman=float("nan"),
                ))
                continue
            per_expert_corr_rows.append(dict(
                run=str(run),
                analyzed_expert=int(e),
                n_points=n,
                pearson=pearson(x, y),
                spearman=spearman(x, y),
            ))

    per_expert_corr = pd.DataFrame(per_expert_corr_rows).sort_values(["run", "analyzed_expert"]) \
        if per_expert_corr_rows else pd.DataFrame()
    per_expert_corr.to_csv(outdir / "per_expert_purity_corr.csv", index=False)

    if len(per_expert_corr) > 0:
        plot_corr_per_expert_line(
            per_expert_corr, "pearson",
            outdir,
            title=f"(B4) Per-expert corr across steps: purity ↔ {metric} (Pearson)",
            fname=f"B4_corr_per_expert_purity_vs_{metric}_pearson"
        )
        plot_corr_per_expert_line(
            per_expert_corr, "spearman",
            outdir,
            title=f"(B5) Per-expert corr across steps: purity ↔ {metric} (Spearman)",
            fname=f"B5_corr_per_expert_purity_vs_{metric}_spearman"
        )

    print(f"[OK] common_max_step={common_max_step}. Wrote outputs to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
