#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select random ImageNet-LT classes from the bottom X percent by count.

Input CSV must have columns:
  - label  (class name / id)
  - count   (integer-like)

If labels appear multiple times, counts are summed per label.
Optionally filters classes by a minimum sample count before computing the bottom X%.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with columns: label,count")
    ap.add_argument("--bottom_pct", type=float, default=5.0, help="Bottom percentage to consider (default: 10)")
    ap.add_argument("--n_select", type=int, default=8, help="How many classes to sample (default: 5)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    ap.add_argument(
        "--min_num_samples",
        type=int,
        default=0,
        help="Only consider classes with count >= min_num_samples (default: 25)",
    )
    ap.add_argument("--out", default="", help="Optional output CSV path (selected labels only). If empty, print.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("label", "count"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Found columns: {list(df.columns)}")

    # Clean + aggregate
    df = df[["label", "count"]].copy()
    df["label"] = df["label"].astype(str)
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["count"])
    df["count"] = df["count"].astype(int)

    agg = df.groupby("label", as_index=False)["count"].sum()
    if agg.empty:
        raise ValueError("No valid rows found after parsing 'count'.")

    # Apply min filter BEFORE bottom-percent selection
    if args.min_num_samples > 0:
        agg = agg.loc[agg["count"] >= args.min_num_samples].copy()
        if agg.empty:
            raise ValueError(
                f"No classes remain after filtering with min_num_samples={args.min_num_samples}."
            )

    # Bottom X% by count (among remaining classes)
    agg = agg.sort_values("count", ascending=True, kind="mergesort").reset_index(drop=True)
    n_classes = len(agg)
    bottom_n = max(1, int(math.ceil((args.bottom_pct / 100.0) * n_classes)))
    bottom = agg.iloc[:bottom_n].copy()

    # Sample
    k = min(args.n_select, len(bottom))
    sampled = bottom.sample(n=k, replace=False, random_state=args.seed).sort_values("count", ascending=True)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sampled[["labels"]].to_csv(out_path, index=False)
    else:
        # Print with counts for convenience
        print(sampled.to_string(index=False))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
