#!/usr/bin/env python3
"""Build the downstream evaluation CSV: NIH-CXR-LT test images whose (single) label is one of the
13 labels shared with MIMIC-CXR-LT. NIH 'Effusion' is MIMIC 'Pleural Effusion'. Yields 16,506 rows.

    python tools/make_nih_eval_csv.py --nih_csv nih-cxr-lt_single-label_test.csv \
        --image_root $NIH_IMAGES --out $GRASP_DATA/nih-cxr-lt/cxr14-lt_test.csv
"""
import argparse
import os

import pandas as pd

SHARED = ["Atelectasis", "Calcification of the Aorta", "Cardiomegaly", "Consolidation", "Edema", "No Finding",
          "Pleural Effusion", "Pneumomediastinum", "Pneumonia", "Pneumoperitoneum", "Pneumothorax",
          "Subcutaneous Emphysema", "Tortuous Aorta"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nih_csv", required=True, help="LongTailCXR nih-cxr-lt_single-label_test.csv")
    ap.add_argument("--image_root", required=True, help="folder with the ChestX-ray14 PNGs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.nih_csv).rename(columns={"Effusion": "Pleural Effusion"})
    lab_cols = [c for c in d.columns if c not in ("id", "subject_id")]
    if not (d[lab_cols].sum(axis=1) == 1).all():
        raise SystemExit("expected exactly one positive label per row (single-label split)")
    d["Finding Labels"] = d[lab_cols].idxmax(axis=1)
    d = d[d["Finding Labels"].isin(SHARED)].copy()
    d["path"] = [os.path.join(a.image_root, i) for i in d["id"]]
    cols = ["id", "subject_id"] + [c for c in SHARED if c in d.columns] + ["Finding Labels", "path"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    d[cols].to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(d)} rows (paper: 16,506)")


if __name__ == "__main__":
    main()
