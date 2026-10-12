#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize fixed-weight GCA checkpoint outputs into a step-indexed CSV.

Expected input layout (default):
  <input-root>/checkpoint-<STEP>/gradient_consistency.csv

For each checkpoint step, this script:
  1) loads the GC CSV,
  2) filters rows (e.g. subset_definition=active, expert_params),
  3) averages selected metrics over the analyzed GC batches per expert,
  4) writes one row per (step, analyzed_expert).

The output is compatible with analyze_text_vs_gradient_over_steps.py,
which requires at least: step, analyzed_expert, <grad_metric>.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "intra_cos",
    "conflict_all",
    "evr_all",
    "gc_paper",
    "mean_grad_cos",
    "inter_cos",
    "intra_minus_inter",
    "purity",
    "dominant_frac",
    "label_entropy",
]


STEP_PATTERNS = [
    re.compile(r"checkpoint-(\d+)"),
    re.compile(r"step[_-]?(\d+)"),
    re.compile(r"ckpt[_-]?(\d+)"),
    re.compile(r"^(\d+)$"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Aggregate per-checkpoint fixed-weight GC CSVs into a step-indexed expert table."
    )
    ap.add_argument("--input-root", required=True, help="Root directory containing checkpoint subdirectories.")
    ap.add_argument(
        "--csv-name",
        default="gradient_consistency.csv",
        help="Expected CSV filename under each checkpoint directory.",
    )
    ap.add_argument("--out-csv", required=True, help="Output CSV path.")
    ap.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric columns to average per (step, analyzed_expert).",
    )
    ap.add_argument(
        "--subset-definition",
        default="active",
        help="Filter on subset_definition column if present (empty string disables).",
    )
    ap.add_argument(
        "--param-group",
        default="expert_params",
        help="Filter on analyzed_param_group column if present (empty string disables).",
    )
    ap.add_argument(
        "--forward-definition",
        default="",
        help="Optional filter for forward_definition column, e.g. routed or fixed:0.",
    )
    ap.add_argument("--min-step", type=int, default=None, help="Optional minimum checkpoint step.")
    ap.add_argument("--max-step", type=int, default=None, help="Optional maximum checkpoint step.")
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


def _normalize_metrics_arg(metrics_raw: str) -> List[str]:
    return [m.strip() for m in str(metrics_raw).split(",") if m.strip()]


def _filter_rows(
    df: pd.DataFrame,
    subset_definition: str,
    param_group: str,
    forward_definition: str,
) -> pd.DataFrame:
    out = df.copy()
    if subset_definition and "subset_definition" in out.columns:
        out = out.loc[out["subset_definition"].astype(str) == str(subset_definition)].copy()
    if param_group and "analyzed_param_group" in out.columns:
        out = out.loc[out["analyzed_param_group"].astype(str) == str(param_group)].copy()
    if forward_definition and "forward_definition" in out.columns:
        out = out.loc[out["forward_definition"].astype(str) == str(forward_definition)].copy()
    return out


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def summarize_one_csv(
    csv_path: Path,
    step: int,
    metrics: List[str],
    subset_definition: str,
    param_group: str,
    forward_definition: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "analyzed_expert" not in df.columns:
        raise KeyError(f"{csv_path}: missing required column 'analyzed_expert'")

    df = _filter_rows(
        df,
        subset_definition=subset_definition,
        param_group=param_group,
        forward_definition=forward_definition,
    )

    use_metrics = [m for m in metrics if m in df.columns]
    if not use_metrics:
        raise KeyError(
            f"{csv_path}: none of requested metrics found. Requested={metrics}, columns={list(df.columns)}"
        )

    df = _coerce_numeric(df, ["analyzed_expert"] + use_metrics)
    df = df.dropna(subset=["analyzed_expert"]).copy()
    df["analyzed_expert"] = df["analyzed_expert"].astype(int)
    df = df.loc[df["analyzed_expert"] >= 0].copy()
    if df.empty:
        return pd.DataFrame()

    group_cols = {m: "mean" for m in use_metrics}
    agg = df.groupby("analyzed_expert", as_index=False).agg(group_cols)
    counts = (
        df.groupby("analyzed_expert", as_index=False)
        .size()
        .rename(columns={"size": "n_gc_rows"})
    )
    agg = agg.merge(counts, on="analyzed_expert", how="left")
    agg["step"] = int(step)
    agg["source_csv"] = str(csv_path)
    return agg


def main() -> None:
    args = parse_args()

    root = Path(args.input_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics = _normalize_metrics_arg(args.metrics)

    pairs = discover_checkpoint_csvs(root, csv_name=str(args.csv_name))
    if not pairs:
        raise RuntimeError(f"No '{args.csv_name}' files found under: {root}")

    rows: List[pd.DataFrame] = []
    for step, csv_path in pairs:
        if args.min_step is not None and step < int(args.min_step):
            continue
        if args.max_step is not None and step > int(args.max_step):
            continue
        try:
            s = summarize_one_csv(
                csv_path=csv_path,
                step=int(step),
                metrics=metrics,
                subset_definition=str(args.subset_definition),
                param_group=str(args.param_group),
                forward_definition=str(args.forward_definition),
            )
        except Exception as exc:
            print(f"[warn] skipping {csv_path}: {exc}")
            continue
        if not s.empty:
            rows.append(s)

    if not rows:
        raise RuntimeError("No valid rows after filtering/aggregation.")

    out = pd.concat(rows, axis=0, ignore_index=True)
    out = out.sort_values(["step", "analyzed_expert"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)

    steps = sorted(out["step"].unique().tolist())
    experts = sorted(out["analyzed_expert"].unique().tolist())
    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] steps={steps}")
    print(f"[ok] experts={experts}")
    print(f"[ok] rows={len(out)}")


if __name__ == "__main__":
    main()
