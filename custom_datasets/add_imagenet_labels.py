#!/usr/bin/env python3
import argparse, ast, pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Append ImageNet label names as 'Finding Labels'.")
    ap.add_argument("csv_in", help="Path to input CSV")
    ap.add_argument("clsidx_txt", help="Path to imagenet1000_clsidx_to_labels.txt (a Python dict literal)")
    ap.add_argument("csv_out", help="Path to output CSV")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Select labels with value > threshold (default: 0.0). If none, use top-1.")
    args = ap.parse_args()

    # Load mapping {0: 'tench, Tinca tinca', 1: 'goldfish, ...', ...}
    with open(args.clsidx_txt, "r", encoding="utf-8") as f:
        mapping_raw = ast.literal_eval(f.read())
    mapping = {int(k): v for k, v in mapping_raw.items()}

    df = pd.read_csv(args.csv_in)

    # Collect available label columns in order
    label_cols = [f"label_{i}" for i in range(1000) if f"label_{i}" in df.columns]
    if not label_cols:
        raise ValueError("No columns named label_0 ... label_999 were found.")

    def row_to_names(row):
        # pick all labels above threshold; else fall back to top-1
        present_idxs = [
            int(col.split("_", 1)[1])
            for col in label_cols
            if pd.to_numeric(row[col], errors="coerce") > args.threshold
        ]
        if not present_idxs:
            # top-1 by value among existing label columns
            best_col = max(label_cols, key=lambda c: pd.to_numeric(row[c], errors="coerce"))
            present_idxs = [int(best_col.split("_", 1)[1])]
        names = [mapping.get(i, f"id_{i}") for i in present_idxs]
        return "; ".join(names)

    df["Finding Labels"] = df[label_cols].apply(row_to_names, axis=1)
    df.to_csv(args.csv_out, index=False)
    print(f"Wrote: {args.csv_out}")

if __name__ == "__main__":
    main()
