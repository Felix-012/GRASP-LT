#!/usr/bin/env python3
"""
Add one-hot label name as 'Finding Labels' and merge 'impression' from CSV B by dicom_id.

Usage:
  python add_labels_and_impressions.py \
      --csv-a A.csv \
      --csv-b B.csv \
      --out A_with_labels_and_impressions.csv \
      --impression-col impression

Notes:
- Assumes CSV A has one-hot columns for labels (exactly one column has value 1 per row; if multiple, the first found is used; if none, leaves empty).
- Detects label columns heuristically: numeric columns whose set of non-null unique values is a subset of {0,1}.
- Excludes 'dicom_id' from label detection, of course.
- Merges impression text from CSV B using the column name provided via --impression-col (default 'impression').
"""

import argparse
import sys
from typing import List
import pandas as pd
import numpy as np


def infer_label_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Heuristically infer one-hot label columns: numeric with values in {0,1}."""
    candidate_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        # Skip obvious non-labels (string/object types)
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            continue
        # Accept only numeric columns with values subset of {0,1} (ignoring NaN)
        uniques = set(pd.unique(df[col].dropna()))
        if len(uniques) > 0 and uniques.issubset({0, 1, 0.0, 1.0, np.int64(0), np.int64(1)}):
            candidate_cols.append(col)
    return candidate_cols


def add_finding_labels(df_a: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
    """Create 'Finding Labels' column from one-hot label columns."""
    # For efficiency, we can argmax over the subset if strictly one-hot; but we’ll keep it simple & explicit.
    def find_label(row):
        for col in label_cols:
            val = row[col]
            # Robust check: count as positive if exactly 1 (int or float)
            if pd.notna(val) and float(val) == 1.0:
                return col
        return ""

    df_a = df_a.copy()
    if 'Finding Labels' in df_a.columns:
        # Overwrite to avoid stale values
        df_a.drop(columns=['Finding Labels'], inplace=True)
    df_a['Finding Labels'] = df_a.apply(find_label, axis=1)
    return df_a


def merge_impressions(df_a: pd.DataFrame, df_b: pd.DataFrame, impression_col: str) -> pd.DataFrame:
    """Left-merge impression text from B onto A using dicom_id."""
    if 'dicom_id' not in df_a.columns:
        raise ValueError("CSV A must contain a 'dicom_id' column.")
    if 'dicom_id' not in df_b.columns:
        raise ValueError("CSV B must contain a 'dicom_id' column.")
    if impression_col not in df_b.columns:
        raise ValueError(f"CSV B must contain the impression column '{impression_col}'.")

    # Ensure unique mapping per dicom_id by taking first non-null impression
    b_subset = df_b[['dicom_id', impression_col]].copy()
    # If there are duplicates, keep the first occurrence (you can change to .last/agg as needed)
    b_subset = b_subset.dropna(subset=['dicom_id']).drop_duplicates(subset=['dicom_id'], keep='first')

    merged = df_a.merge(b_subset, on='dicom_id', how='left')
    # Rename impression column in the merged result to 'impression'
    if impression_col != 'impression':
        merged.rename(columns={impression_col: 'impression'}, inplace=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Add 'Finding Labels' from one-hot columns and merge impressions by dicom_id.")
    parser.add_argument("--csv-a", required=True, help="Path to CSV A (has one-hot label columns and dicom_id).")
    parser.add_argument("--csv-b", required=False, help="Path to CSV B (has dicom_id and impression text).")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--impression-col", default="impression", help="Column name in B for the impression text (default: impression).")
    parser.add_argument("--label-cols", nargs="*", default=None,
                        help="Optional explicit list of label columns in A. If omitted, inferred automatically.")
    parser.add_argument("--index", action="store_true", help="Keep DataFrame index in the output CSV.")
    args = parser.parse_args()

    # Read inputs
    try:
        df_a = pd.read_csv(args.csv_a)
    except Exception as e:
        print(f"Failed to read CSV A from {args.csv_a}: {e}", file=sys.stderr)
        sys.exit(1)

    df_b = None
    if args.csv_b:
        try:
            df_b = pd.read_csv(args.csv_b)
        except Exception as e:
            print(f"Failed to read CSV B from {args.csv_b}: {e}", file=sys.stderr)
            sys.exit(1)

    # Infer label columns if not provided
    exclude = ['dicom_id']
    if args.label_cols is None or len(args.label_cols) == 0:
        label_cols = infer_label_columns(df_a, exclude=exclude)
        if not label_cols:
            print("No one-hot label columns inferred. "
                  "Provide them explicitly via --label-cols.", file=sys.stderr)
            sys.exit(2)
    else:
        missing = [c for c in args.label_cols if c not in df_a.columns]
        if missing:
            print(f"The following --label-cols are not present in A: {missing}", file=sys.stderr)
            sys.exit(2)
        label_cols = args.label_cols

    # Build Finding Labels
    df_a_labeled = add_finding_labels(df_a, label_cols)

    # Merge impressions from B
    if df_b is None:          # no impressions yet: keep an empty column, filled later by utils/add_impression.py
        merged = df_a_labeled.copy()
        if args.impression_col not in merged.columns:
            merged[args.impression_col] = None
    else:
        merged = merge_impressions(df_a_labeled, df_b, args.impression_col)

    # Ensure final columns include 'Finding Labels' and 'impression' near the end (optional)
    # Keep original order, append if needed
    cols = list(merged.columns)
    for needed in ['Finding Labels', 'impression']:
        if needed in cols:
            cols.remove(needed)
            cols.append(needed)
    merged = merged[cols]

    # Write output
    try:
        merged.to_csv(args.out, index=args.index)
    except Exception as e:
        print(f"Failed to write output CSV to {args.out}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Wrote: {args.out}")
    print(f"Inferred label columns: {label_cols}")


if __name__ == "__main__":
    main()
