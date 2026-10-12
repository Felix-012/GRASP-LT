#!/usr/bin/env python3
"""Rebuild the GRASP label partition from scratch (balanced METIS partition).

This is the procedure that produced the paper partition: each sample is represented
by the set of its finding labels, a Jaccard k-NN graph is built over samples, and
METIS cuts it into K balanced parts (ufactor 500, i.e. ~1.5:1 imbalance tolerance).
"No Finding" samples are routed to a reserved expert K. Because MIMIC-CXR-LT is
single-label, the result is a balanced binning of whole labels.

METIS output depends on the METIS build, so use `build_label_files.py --map submitted`
to reproduce the paper exactly; this script documents and reruns the procedure.
Requires: pip install metis pyroaring (plus a system libmetis).
"""
import argparse
import collections
import re
import sys
from pathlib import Path

import pandas as pd
from pyroaring import FrozenBitMap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_graph.cluster_labels import single_kway_partition_from_bitmaps  # noqa: E402


def parse_cell(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s == "No Finding":
        return []
    return [p for p in (q.strip() for q in re.split(r"\s*\|\s*", s)) if p and p != "No Finding"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=3, help="number of pathology groups (healthy expert is added as id K)")
    ap.add_argument("--ufactor", type=int, default=500)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    col_idx = {c: i for i, c in enumerate(df.columns)}
    bitmaps, valid, empty = [], [], []
    for i, cell in enumerate(df["Finding Labels"]):
        labs = parse_cell(cell)
        if not labs:
            empty.append(i)
            continue
        bitmaps.append(FrozenBitMap(sorted({col_idx[n] for n in labs})))
        valid.append(i)

    parts, _ = single_kway_partition_from_bitmaps(bitmaps, jac_weights=None, K=args.K, ufactor=args.ufactor)
    labels = [None] * len(df)
    for j, i in enumerate(valid):
        labels[i] = parts[j]
    for i in empty:
        labels[i] = args.K
    Path(args.out).write_text("\n".join(map(str, labels)))
    print("samples per expert:", dict(sorted(collections.Counter(labels).items())))


if __name__ == "__main__":
    main()
