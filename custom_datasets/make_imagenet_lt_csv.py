#!/usr/bin/env python3
"""Build imagenet_lt_train.csv / imagenet_lt_test.csv from the official ImageNet-LT split files (Liu et al., 2019).

    python custom_datasets/make_imagenet_lt_csv.py --txt_dir <dir with ImageNet_LT_{train,val,test}.txt> \
        --clsidx imagenet1000_clsidx_to_labels.txt --out_dir $GRASP_DATA/imagenet-lt

Layout of the paper files: one running index over train, val and test (in that order), written twice (index and
"Unnamed: 0") and as "id"; one-hot columns label_0 ... label_999; "Split" (TRAIN / VAL / TEST); "Finding Labels" =
the class name used as the prompt. The test rows keep the order of ImageNet_LT_test.txt, which also fixes the noise
assigned to each prompt during generation. Check the result with tools/verify_data.py (checksums of the paper files).
"""
import argparse
import ast
import csv
from pathlib import Path

N_CLASSES = 1000


def read_split(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            p, c = line.rsplit(maxsplit=1)
            rows.append((p, int(c)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--txt_dir", required=True)
    ap.add_argument("--clsidx", required=True, help="imagenet1000_clsidx_to_labels.txt (a Python dict literal)")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    names = ast.literal_eval(Path(a.clsidx).read_text())
    header = ["", "Unnamed: 0", "path", "id"] + [f"label_{i}" for i in range(N_CLASSES)] + ["Split", "Finding Labels"]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writers = {}
    files = []
    for split, target in (("TRAIN", "imagenet_lt_train.csv"), ("VAL", None), ("TEST", "imagenet_lt_test.csv")):
        if target:
            f = open(out / target, "w", newline="")
            files.append(f)
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            writers[split] = w
    idx = 0
    for split, fname in (("TRAIN", "ImageNet_LT_train.txt"), ("VAL", "ImageNet_LT_val.txt"), ("TEST", "ImageNet_LT_test.txt")):
        for p, c in read_split(Path(a.txt_dir) / fname):
            if split in writers:
                onehot = ["0"] * N_CLASSES
                onehot[c] = "1"
                writers[split].writerow([idx, idx, p, idx] + onehot + [split, names[c]])
            idx += 1
    for f in files:
        f.close()
    print(f"wrote {out}/imagenet_lt_train.csv and {out}/imagenet_lt_test.csv ({idx} indexed rows)")


if __name__ == "__main__":
    main()
