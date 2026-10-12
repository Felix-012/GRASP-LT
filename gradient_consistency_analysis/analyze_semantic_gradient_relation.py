#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _norm_label(x: str) -> str:
    return " ".join(str(x).strip().split())


def _canon_pair(a: str, b: str) -> Tuple[str, str]:
    a_n, b_n = _norm_label(a), _norm_label(b)
    return (a_n, b_n) if a_n <= b_n else (b_n, a_n)


def corr_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
    if x.size < 2:
        return float("nan")
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def corr_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return corr_pearson(xr, yr)


def moving_average(v: np.ndarray, window: int) -> np.ndarray:
    if v.size == 0:
        return v
    w = int(window)
    if w <= 1:
        return v
    # NaN-safe smoothing: rolling mean ignores missing values and keeps length.
    w = min(w, int(v.size))
    s = pd.Series(v, dtype="float64")
    return s.rolling(window=w, min_periods=1, center=True).mean().to_numpy(dtype=np.float64)


def align_to_shortest(x: np.ndarray, *ys: np.ndarray) -> Tuple[np.ndarray, ...]:
    if x.size == 0:
        return (x,) + ys
    min_len = int(x.size)
    for y in ys:
        min_len = min(min_len, int(y.size))
    x_trim = x[:min_len]
    ys_trim = tuple(y[:min_len] for y in ys)
    return (x_trim,) + ys_trim


def finite_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(y)
    return x[m], y[m]


def finite_envelope(
    x: np.ndarray, y_mu: np.ndarray, y_sd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = y_mu - y_sd
    hi = y_mu + y_sd
    m = np.isfinite(y_mu) & np.isfinite(lo) & np.isfinite(hi)
    return x[m], lo[m], hi[m]


def load_pairwise_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pairwise CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"step", "label_i", "label_j", "grad_cosine"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df[list(required)].copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["grad_cosine"] = pd.to_numeric(df["grad_cosine"], errors="coerce")
    df["label_i"] = df["label_i"].astype(str).map(_norm_label)
    df["label_j"] = df["label_j"].astype(str).map(_norm_label)
    df = df.dropna(subset=["step", "grad_cosine"])
    df["step"] = df["step"].astype(int)

    # Canonicalize pair orientation.
    swap_mask = df["label_i"] > df["label_j"]
    li = df.loc[swap_mask, "label_i"].copy()
    df.loc[swap_mask, "label_i"] = df.loc[swap_mask, "label_j"].to_numpy()
    df.loc[swap_mask, "label_j"] = li.to_numpy()
    return df.sort_values(["step", "label_i", "label_j"]).reset_index(drop=True)


def _char_ngrams(text: str, n: int = 3) -> List[str]:
    s = f"  {text.lower()}  "
    if len(s) < n:
        return [s]
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def lexical_distance_matrix(labels: Sequence[str]) -> np.ndarray:
    labels = [_norm_label(x) for x in labels]
    n = len(labels)
    grams_per_label = [_char_ngrams(x, n=3) for x in labels]
    df_counter = Counter()
    for grams in grams_per_label:
        for g in set(grams):
            df_counter[g] += 1

    vocab = sorted(df_counter.keys())
    vid = {g: i for i, g in enumerate(vocab)}
    X = np.zeros((n, len(vocab)), dtype=np.float64)
    for i, grams in enumerate(grams_per_label):
        tf = Counter(grams)
        for g, c in tf.items():
            j = vid[g]
            X[i, j] = float(c)
    idf = np.array(
        [np.log((1.0 + n) / (1.0 + float(df_counter[g]))) + 1.0 for g in vocab],
        dtype=np.float64,
    )
    X *= idf[None, :]

    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    sim = np.clip(Xn @ Xn.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def load_distance_overrides(path: Path) -> Dict[Tuple[str, str], float]:
    if not path.exists():
        raise FileNotFoundError(f"Semantic distance CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"label_i", "label_j", "distance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out: Dict[Tuple[str, str], float] = {}
    for row in df.itertuples(index=False):
        li = _norm_label(getattr(row, "label_i"))
        lj = _norm_label(getattr(row, "label_j"))
        try:
            d = float(getattr(row, "distance"))
        except Exception:
            continue
        if not np.isfinite(d):
            continue
        out[_canon_pair(li, lj)] = d
    return out


def build_semantic_distance_matrix(
    labels: Sequence[str],
    semantic_distance_csv: Path | None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    labels_sorted = sorted({_norm_label(x) for x in labels})
    label_to_idx = {lab: i for i, lab in enumerate(labels_sorted)}
    dist = lexical_distance_matrix(labels_sorted)

    if semantic_distance_csv is not None:
        overrides = load_distance_overrides(semantic_distance_csv)
        for (li, lj), d in overrides.items():
            if li not in label_to_idx or lj not in label_to_idx:
                continue
            i, j = label_to_idx[li], label_to_idx[lj]
            if i == j:
                dist[i, j] = 0.0
            else:
                dist[i, j] = float(d)
                dist[j, i] = float(d)
    np.fill_diagonal(dist, 0.0)
    return dist, label_to_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 1 + 2: analyze semantic->gradient relationship from "
            "label-pair projected cosine CSVs."
        )
    )
    parser.add_argument("--csvs", nargs="+", required=True, help="Label-pair cosine CSVs.")
    parser.add_argument("--names", nargs="*", default=None, help="Optional names for each CSV.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--semantic-distance-csv",
        type=str,
        default=None,
        help=(
            "Optional CSV with columns: label_i,label_j,distance. "
            "If omitted, lexical char-ngram distance is used."
        ),
    )
    parser.add_argument("--num-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average window for plotted curves.")
    parser.add_argument(
        "--prefix",
        type=str,
        default="semantic_gradient",
        help="Filename prefix for outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(x) for x in args.csvs]
    if args.names is not None and len(args.names) not in (0, len(csv_paths)):
        raise ValueError("If --names is provided, it must match --csvs length.")
    run_names = args.names if args.names else [p.stem for p in csv_paths]

    runs_df: List[pd.DataFrame] = []
    all_labels: List[str] = []
    for name, path in zip(run_names, csv_paths):
        df = load_pairwise_csv(path)
        df["run"] = str(name)
        runs_df.append(df)
        all_labels.extend(df["label_i"].tolist())
        all_labels.extend(df["label_j"].tolist())

    if not runs_df:
        raise RuntimeError("No pairwise CSVs loaded.")
    merged = pd.concat(runs_df, axis=0, ignore_index=True)
    raw_steps_by_run = (
        merged.groupby("run", as_index=False)["step"]
        .nunique()
        .rename(columns={"step": "raw_steps"})
    )

    semantic_csv = Path(args.semantic_distance_csv) if args.semantic_distance_csv else None
    dist_matrix, label_to_idx = build_semantic_distance_matrix(all_labels, semantic_csv)
    n_labels = len(label_to_idx)
    if n_labels < 2:
        raise RuntimeError("Need at least two labels for semantic-distance analysis.")

    rng = np.random.default_rng(int(args.seed))
    n_perm = max(0, int(args.num_permutations))
    permutations = [rng.permutation(n_labels) for _ in range(n_perm)]

    step_rows: List[Dict[str, object]] = []
    for run, df_run in merged.groupby("run"):
        for step, df_step in df_run.groupby("step"):
            idx_i = np.array([label_to_idx[x] for x in df_step["label_i"]], dtype=np.int64)
            idx_j = np.array([label_to_idx[x] for x in df_step["label_j"]], dtype=np.int64)
            grad_cos = df_step["grad_cosine"].to_numpy(dtype=np.float64)
            m = np.isfinite(grad_cos)
            if int(m.sum()) < int(args.min_pairs):
                continue
            idx_i, idx_j, grad_cos = idx_i[m], idx_j[m], grad_cos[m]
            sem_dist_true = dist_matrix[idx_i, idx_j]

            spearman_true = corr_spearman(sem_dist_true, grad_cos)
            pearson_true = corr_pearson(sem_dist_true, grad_cos)

            spearman_perm: List[float] = []
            pearson_perm: List[float] = []
            for perm in permutations:
                sem_dist_perm = dist_matrix[perm[idx_i], perm[idx_j]]
                spearman_perm.append(corr_spearman(sem_dist_perm, grad_cos))
                pearson_perm.append(corr_pearson(sem_dist_perm, grad_cos))

            spearman_perm_arr = np.asarray(spearman_perm, dtype=np.float64)
            pearson_perm_arr = np.asarray(pearson_perm, dtype=np.float64)
            valid_sp = spearman_perm_arr[np.isfinite(spearman_perm_arr)]
            valid_pe = pearson_perm_arr[np.isfinite(pearson_perm_arr)]
            sp_mean = float(np.mean(valid_sp)) if valid_sp.size else float("nan")
            sp_std = float(np.std(valid_sp)) if valid_sp.size else float("nan")
            pe_mean = float(np.mean(valid_pe)) if valid_pe.size else float("nan")
            pe_std = float(np.std(valid_pe)) if valid_pe.size else float("nan")

            p_sp = (
                float((np.sum(np.abs(valid_sp) >= abs(spearman_true)) + 1) / (valid_sp.size + 1))
                if valid_sp.size
                else float("nan")
            )
            p_pe = (
                float((np.sum(np.abs(valid_pe) >= abs(pearson_true)) + 1) / (valid_pe.size + 1))
                if valid_pe.size
                else float("nan")
            )

            step_rows.append(
                dict(
                    run=str(run),
                    step=int(step),
                    n_pairs=int(grad_cos.size),
                    spearman_true=float(spearman_true),
                    pearson_true=float(pearson_true),
                    spearman_perm_mean=sp_mean,
                    spearman_perm_std=sp_std,
                    spearman_perm_pvalue=p_sp,
                    pearson_perm_mean=pe_mean,
                    pearson_perm_std=pe_std,
                    pearson_perm_pvalue=p_pe,
                )
            )

    if not step_rows:
        raise RuntimeError("No valid step-level rows were produced.")

    step_df = pd.DataFrame(step_rows).sort_values(["run", "step"]).reset_index(drop=True)
    step_path = outdir / f"{args.prefix}_step_correlations.csv"
    step_df.to_csv(step_path, index=False)

    summary = (
        step_df.groupby("run", as_index=False)
        .agg(
            steps=("step", "nunique"),
            mean_pairs=("n_pairs", "mean"),
            spearman_true_mean=("spearman_true", "mean"),
            spearman_perm_mean=("spearman_perm_mean", "mean"),
            spearman_perm_p_lt_05=("spearman_perm_pvalue", lambda s: float(np.nanmean(np.array(s) < 0.05))),
            pearson_true_mean=("pearson_true", "mean"),
            pearson_perm_mean=("pearson_perm_mean", "mean"),
            pearson_perm_p_lt_05=("pearson_perm_pvalue", lambda s: float(np.nanmean(np.array(s) < 0.05))),
        )
        .sort_values("run")
    )
    summary = summary.merge(raw_steps_by_run, on="run", how="left")
    summary["kept_step_fraction"] = summary["steps"] / summary["raw_steps"].replace(0, np.nan)
    summary_path = outdir / f"{args.prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Plot 1: per-run true vs permutation-null envelope (both Spearman and Pearson).
    plot_specs = [
        (
            "spearman",
            "spearman_true",
            "spearman_perm_mean",
            "spearman_perm_std",
            "Spearman(semantic distance, gradient cosine)",
        ),
        (
            "pearson",
            "pearson_true",
            "pearson_perm_mean",
            "pearson_perm_std",
            "Pearson(semantic distance, gradient cosine)",
        ),
    ]
    for metric_tag, true_col, mu_col, sd_col, ylab in plot_specs:
        for run, df_run in step_df.groupby("run"):
            d = df_run.sort_values("step")
            x = d["step"].to_numpy(dtype=np.int64)
            y_true = moving_average(d[true_col].to_numpy(dtype=np.float64), int(args.smooth))
            y_mu = moving_average(d[mu_col].to_numpy(dtype=np.float64), int(args.smooth))
            y_sd = moving_average(d[sd_col].to_numpy(dtype=np.float64), int(args.smooth))
            x, y_true, y_mu, y_sd = align_to_shortest(x, y_true, y_mu, y_sd)

            x_true, y_true_f = finite_xy(x, y_true)
            x_mu, y_mu_f = finite_xy(x, y_mu)
            x_env, y_lo, y_hi = finite_envelope(x, y_mu, y_sd)
            if x_true.size == 0 and x_mu.size == 0:
                print(f"[warn] no finite {metric_tag} points to plot for run={run!r}")
                continue

            plt.figure(figsize=(8.5, 4.2))
            if x_true.size > 0:
                plt.plot(x_true, y_true_f, linewidth=2.2, label="True semantics")
            if x_mu.size > 0:
                plt.plot(x_mu, y_mu_f, linewidth=1.8, linestyle="--", label="Permutation mean")
            if x_env.size > 0:
                plt.fill_between(x_env, y_lo, y_hi, alpha=0.2, label="Permutation ±1 std")
            plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
            plt.xlabel("Training step")
            plt.ylabel(ylab)
            plt.title(f"Semantic -> gradient relationship ({run})")
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            out_png = outdir / f"{args.prefix}_{run}_{metric_tag}_perm_control.png"
            plt.savefig(out_png, dpi=200)
            plt.savefig(out_png.with_suffix(".pdf"))
            plt.close()

    # Plot 2: multi-run true-correlation curves (Spearman + Pearson).
    compare_specs = [
        ("spearman", "spearman_true", "Spearman(semantic distance, gradient cosine)"),
        ("pearson", "pearson_true", "Pearson(semantic distance, gradient cosine)"),
    ]
    for metric_tag, true_col, ylab in compare_specs:
        plt.figure(figsize=(9.0, 4.5))
        plotted = 0
        for run, df_run in step_df.groupby("run"):
            d = df_run.sort_values("step")
            x = d["step"].to_numpy(dtype=np.int64)
            y_true = moving_average(d[true_col].to_numpy(dtype=np.float64), int(args.smooth))
            x, y_true = align_to_shortest(x, y_true)
            x_f, y_f = finite_xy(x, y_true)
            if x_f.size == 0:
                print(f"[warn] no finite {metric_tag} curve in compare plot for run={run!r}")
                continue
            plt.plot(x_f, y_f, linewidth=2.0, label=str(run))
            plotted += 1

        if plotted == 0:
            plt.close()
            raise RuntimeError(f"No finite {metric_tag} curves found for compare plot.")
        plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
        plt.xlabel("Training step")
        plt.ylabel(ylab)
        plt.title("Semantic-distance vs gradient-similarity over training")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        combined_png = outdir / f"{args.prefix}_compare_{metric_tag}_true.png"
        plt.savefig(combined_png, dpi=200)
        plt.savefig(combined_png.with_suffix(".pdf"))
        plt.close()

    print(f"Wrote: {step_path}")
    print(f"Wrote: {summary_path}")
    for row in summary.itertuples(index=False):
        frac = float(getattr(row, "kept_step_fraction"))
        if np.isfinite(frac) and frac < 0.5:
            print(
                f"[warn] sparse curve for run={getattr(row, 'run')!r}: "
                f"kept {int(getattr(row, 'steps'))}/{int(getattr(row, 'raw_steps'))} steps "
                f"(fraction={frac:.3f}). Consider lowering --min-pairs."
            )
    print(f"Wrote plots under: {outdir}")


if __name__ == "__main__":
    main()
