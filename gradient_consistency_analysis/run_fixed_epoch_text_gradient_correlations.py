#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute expert-level text-vs-gradient correlations from fixed-epoch GCA outputs.

Input expectation:
  <fixed-root>/checkpoint-<STEP>/gradient_consistency.csv

For each requested profile (subset_definition:metric), this script:
  1) Aggregates each checkpoint CSV to one row per (step, expert_id).
  2) Computes per-anchor correlations across experts:
       - Pearson(text_sim, grad_sim)
       - Spearman(text_sim, grad_sim)
  3) Writes per-profile CSV outputs and a compact cross-profile summary.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_PROFILES = [
    "active:intra_cos",
    "dominant_label:intra_cos",
    "dominant_label:mean_grad_cos",
    "dominant_label:gc_paper",
]

STEP_PATTERNS = [
    re.compile(r"checkpoint-(\d+)"),
    re.compile(r"step[_-]?(\d+)"),
    re.compile(r"ckpt[_-]?(\d+)"),
    re.compile(r"^(\d+)$"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute fixed-epoch expert-level text-vs-gradient correlation CSVs."
    )
    ap.add_argument("--fixed-root", required=True, help="Directory containing checkpoint-*/gradient_consistency.csv.")
    ap.add_argument(
        "--text-sim",
        required=True,
        help=(
            "Path to text-sim file. Supported: "
            "CSV with expert_id/text_sim columns (or equivalent), "
            "or txt lines like 'expert 0: ... avg_cos=0.50789'."
        ),
    )
    ap.add_argument("--outdir", required=True, help="Output directory for generated CSVs.")
    ap.add_argument("--csv-name", default="gradient_consistency.csv", help="Filename under each checkpoint directory.")
    ap.add_argument(
        "--profiles",
        nargs="+",
        default=DEFAULT_PROFILES,
        help="Space-separated subset:metric entries (e.g. active:intra_cos dominant_label:mean_grad_cos).",
    )
    ap.add_argument("--param-group", default="expert_params", help="Filter on analyzed_param_group if column exists.")
    ap.add_argument("--forward-definition", default="routed", help="Filter on forward_definition if column exists.")
    ap.add_argument("--window-size", type=int, default=1, help="Trailing window size over checkpoint steps.")
    ap.add_argument("--min-experts", type=int, default=2, help="Minimum experts required for correlation.")
    ap.add_argument("--step-stride", type=int, default=1, help="Keep every Nth anchor step.")
    ap.add_argument("--min-step", type=int, default=None, help="Optional minimum checkpoint step.")
    ap.add_argument("--max-step", type=int, default=None, help="Optional maximum checkpoint step.")
    ap.add_argument("--prefix", default="fixed_epoch_text_vs_gradient", help="Prefix for summary outputs.")
    return ap.parse_args()


def _extract_step_from_path(path: Path) -> Optional[int]:
    parts = [path.name] + [p.name for p in path.parents]
    for part in parts:
        for pat in STEP_PATTERNS:
            m = pat.search(part)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    continue
    return None


def discover_checkpoint_csvs(root: Path, csv_name: str) -> List[Tuple[int, Path]]:
    pairs: List[Tuple[int, Path]] = []
    for csv_path in root.rglob(csv_name):
        if not csv_path.is_file():
            continue
        step = _extract_step_from_path(csv_path)
        if step is None:
            continue
        pairs.append((int(step), csv_path))
    pairs.sort(key=lambda x: (x[0], str(x[1])))
    return pairs


def _safe_profile_tag(profile: str) -> str:
    subset, metric = profile.split(":", 1)
    subset = re.sub(r"[^a-zA-Z0-9_]+", "_", subset.strip())
    metric = re.sub(r"[^a-zA-Z0-9_]+", "_", metric.strip())
    return f"{subset}__{metric}"


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_text_similarity(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"text-sim file not found: {path}")

    if path.suffix.lower() == ".txt":
        # Expected lines like: expert 0: n=... avg_cos=0.507890 ...
        rows: List[Dict[str, float]] = []
        pat = re.compile(r"expert\s+(\d+)\s*:.*?avg_cos\s*=\s*([\-+0-9.eE]+)")
        for ln in path.read_text(encoding="utf-8").splitlines():
            m = pat.search(ln)
            if not m:
                continue
            rows.append(dict(expert_id=int(m.group(1)), text_sim=float(m.group(2))))
        if not rows:
            raise ValueError(f"{path}: could not parse any 'expert ... avg_cos=...' lines.")
        df = pd.DataFrame(rows)
        return df.groupby("expert_id", as_index=False)["text_sim"].mean()

    df = pd.read_csv(path)
    expert_col = None
    for c in ("expert_id", "analyzed_expert", "expert"):
        if c in df.columns:
            expert_col = c
            break
    sim_col = None
    for c in ("text_sim", "avg_cos", "mean_cos"):
        if c in df.columns:
            sim_col = c
            break

    if expert_col is None or sim_col is None:
        raise KeyError(
            f"{path}: need expert and text similarity columns. "
            f"Supported expert cols: expert_id/analyzed_expert/expert, "
            f"supported sim cols: text_sim/avg_cos/mean_cos."
        )

    out = df.loc[:, [expert_col, sim_col]].copy()
    out.columns = ["expert_id", "text_sim"]
    out = _coerce_numeric(out, ["expert_id", "text_sim"])
    out = out.dropna(subset=["expert_id", "text_sim"]).copy()
    out["expert_id"] = out["expert_id"].astype(int)
    out = out.groupby("expert_id", as_index=False)["text_sim"].mean()
    return out.sort_values("expert_id").reset_index(drop=True)


def summarize_profile_one_csv(
    *,
    csv_path: Path,
    step: int,
    subset_definition: str,
    metric: str,
    param_group: str,
    forward_definition: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "analyzed_expert" not in df.columns:
        raise KeyError(f"{csv_path}: missing required column 'analyzed_expert'")
    if metric not in df.columns:
        raise KeyError(f"{csv_path}: missing required metric column {metric!r}")

    if subset_definition and "subset_definition" in df.columns:
        df = df.loc[df["subset_definition"].astype(str) == str(subset_definition)].copy()
    if param_group and "analyzed_param_group" in df.columns:
        df = df.loc[df["analyzed_param_group"].astype(str) == str(param_group)].copy()
    if forward_definition and "forward_definition" in df.columns:
        df = df.loc[df["forward_definition"].astype(str) == str(forward_definition)].copy()

    df = _coerce_numeric(df, ["analyzed_expert", metric])
    df = df.dropna(subset=["analyzed_expert", metric]).copy()
    if df.empty:
        return pd.DataFrame()

    df["analyzed_expert"] = df["analyzed_expert"].astype(int)
    df = df.loc[df["analyzed_expert"] >= 0].copy()
    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby("analyzed_expert", as_index=False)
        .agg(
            grad_sim=(metric, "mean"),
            grad_sim_std=(metric, "std"),
            n_grad_rows=(metric, "count"),
        )
        .rename(columns={"analyzed_expert": "expert_id"})
    )
    agg["step"] = int(step)
    agg["source_csv"] = str(csv_path)
    return agg


def summarize_profile(
    *,
    csv_pairs: Sequence[Tuple[int, Path]],
    subset_definition: str,
    metric: str,
    param_group: str,
    forward_definition: str,
    min_step: Optional[int],
    max_step: Optional[int],
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for step, csv_path in csv_pairs:
        if min_step is not None and step < int(min_step):
            continue
        if max_step is not None and step > int(max_step):
            continue
        try:
            part = summarize_profile_one_csv(
                csv_path=csv_path,
                step=int(step),
                subset_definition=subset_definition,
                metric=metric,
                param_group=param_group,
                forward_definition=forward_definition,
            )
        except Exception as exc:
            print(f"[warn] skipped {csv_path}: {exc}")
            continue
        if not part.empty:
            rows.append(part)

    if not rows:
        return pd.DataFrame(
            columns=["step", "expert_id", "grad_sim", "grad_sim_std", "n_grad_rows", "source_csv"]
        )
    out = pd.concat(rows, axis=0, ignore_index=True)
    out = out.sort_values(["step", "expert_id"]).reset_index(drop=True)
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson(xr, yr)


def build_anchor_correlations(
    *,
    profile_df: pd.DataFrame,
    text_df: pd.DataFrame,
    window_size: int,
    min_experts: int,
    step_stride: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if profile_df.empty:
        empty_corr = pd.DataFrame(
            columns=[
                "step_anchor",
                "window_start",
                "window_end",
                "n_experts",
                "pearson",
                "spearman",
                "text_mean",
                "grad_mean",
            ]
        )
        empty_merged = pd.DataFrame(
            columns=[
                "step_anchor",
                "window_start",
                "window_end",
                "expert_id",
                "grad_sim",
                "grad_sim_std",
                "n_grad_rows",
                "text_sim",
            ]
        )
        return empty_corr, empty_merged

    steps = sorted(profile_df["step"].astype(int).unique().tolist())
    anchors = steps[:: max(1, int(step_stride))]

    merged_rows: List[pd.DataFrame] = []
    corr_rows: List[Dict[str, object]] = []
    for anchor in anchors:
        lo = int(anchor) - int(window_size) + 1
        hi = int(anchor)
        sub = profile_df.loc[(profile_df["step"] >= lo) & (profile_df["step"] <= hi)].copy()
        if sub.empty:
            continue

        agg = (
            sub.groupby("expert_id", as_index=False)
            .agg(
                grad_sim=("grad_sim", "mean"),
                grad_sim_std=("grad_sim", "std"),
                n_grad_rows=("n_grad_rows", "sum"),
            )
        )
        m = agg.merge(text_df, on="expert_id", how="left")
        m["step_anchor"] = int(anchor)
        m["window_start"] = int(lo)
        m["window_end"] = int(hi)
        merged_rows.append(m)

        x = pd.to_numeric(m["text_sim"], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(m["grad_sim"], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        n = int(x.size)

        corr_rows.append(
            dict(
                step_anchor=int(anchor),
                window_start=int(lo),
                window_end=int(hi),
                n_experts=n,
                pearson=(float(_pearson(x, y)) if n >= int(min_experts) else float("nan")),
                spearman=(float(_spearman(x, y)) if n >= int(min_experts) else float("nan")),
                text_mean=(float(np.mean(x)) if n > 0 else float("nan")),
                grad_mean=(float(np.mean(y)) if n > 0 else float("nan")),
            )
        )

    merged_df = (
        pd.concat(merged_rows, axis=0, ignore_index=True)
        if merged_rows
        else pd.DataFrame(
            columns=[
                "step_anchor",
                "window_start",
                "window_end",
                "expert_id",
                "grad_sim",
                "grad_sim_std",
                "n_grad_rows",
                "text_sim",
            ]
        )
    )
    corr_df = (
        pd.DataFrame(corr_rows).sort_values("step_anchor").reset_index(drop=True)
        if corr_rows
        else pd.DataFrame(
            columns=[
                "step_anchor",
                "window_start",
                "window_end",
                "n_experts",
                "pearson",
                "spearman",
                "text_mean",
                "grad_mean",
            ]
        )
    )
    return corr_df, merged_df


def parse_profile(profile: str) -> Tuple[str, str]:
    if ":" not in profile:
        raise ValueError(f"Invalid profile {profile!r}; expected subset:metric")
    subset, metric = profile.split(":", 1)
    subset = subset.strip()
    metric = metric.strip()
    if not subset or not metric:
        raise ValueError(f"Invalid profile {profile!r}; expected subset:metric")
    return subset, metric


def main() -> None:
    args = parse_args()
    fixed_root = Path(args.fixed_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_pairs = discover_checkpoint_csvs(fixed_root, csv_name=str(args.csv_name))
    if not csv_pairs:
        raise RuntimeError(f"No '{args.csv_name}' files found under: {fixed_root}")

    text_df = load_text_similarity(Path(args.text_sim))
    text_out = outdir / f"{args.prefix}_text_similarity_per_expert.csv"
    text_df.to_csv(text_out, index=False)

    profile_summary_rows: List[Dict[str, object]] = []
    for profile in args.profiles:
        subset, metric = parse_profile(profile)
        tag = _safe_profile_tag(profile)
        print(f"[run] profile={profile}")

        step_expert_df = summarize_profile(
            csv_pairs=csv_pairs,
            subset_definition=subset,
            metric=metric,
            param_group=str(args.param_group),
            forward_definition=str(args.forward_definition),
            min_step=args.min_step,
            max_step=args.max_step,
        )
        step_expert_csv = outdir / f"{args.prefix}_{tag}_step_expert.csv"
        step_expert_df.to_csv(step_expert_csv, index=False)

        corr_df, merged_df = build_anchor_correlations(
            profile_df=step_expert_df,
            text_df=text_df,
            window_size=int(args.window_size),
            min_experts=int(args.min_experts),
            step_stride=int(args.step_stride),
        )
        merged_csv = outdir / f"{args.prefix}_{tag}_text_vs_gradient_per_anchor.csv"
        corr_csv = outdir / f"{args.prefix}_{tag}_correlations_per_anchor.csv"
        merged_df.to_csv(merged_csv, index=False)
        corr_df.to_csv(corr_csv, index=False)

        mean_pearson = float(pd.to_numeric(corr_df.get("pearson"), errors="coerce").mean(skipna=True)) if not corr_df.empty else float("nan")
        mean_spearman = float(pd.to_numeric(corr_df.get("spearman"), errors="coerce").mean(skipna=True)) if not corr_df.empty else float("nan")
        anchors = int(corr_df["step_anchor"].nunique()) if "step_anchor" in corr_df.columns else 0
        profile_summary_rows.append(
            dict(
                profile=profile,
                step_expert_csv=str(step_expert_csv),
                merged_csv=str(merged_csv),
                corr_csv=str(corr_csv),
                anchors=anchors,
                mean_pearson=mean_pearson,
                mean_spearman=mean_spearman,
            )
        )

    summary_df = pd.DataFrame(profile_summary_rows).sort_values("profile").reset_index(drop=True)
    summary_csv = outdir / f"{args.prefix}_profile_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"[ok] text similarity: {text_out}")
    print(f"[ok] profile summary: {summary_csv}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
