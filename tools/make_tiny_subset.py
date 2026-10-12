#!/usr/bin/env python3
"""Cut a tiny, self-consistent copy of a prepared MIMIC-CXR-LT data root (for smoke tests and development).

Rows are chosen so that every finding (and hence every expert of every partition) appears at least once;
the remaining rows are the first rows of each split. All row-aligned files are subset consistently:
CSVs, latent / embedding HDF5s (item_<i> groups, renumbered), routing files, text-cluster labels, and the
real-image HDF5s; the NIH evaluation HDF5 is cut to its first --nih rows.

    python tools/make_tiny_subset.py --src $GRASP_DATA --dst /tmp/grasp_tiny --train 256 --test 128
"""
import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def pick(df, n, col="Finding Labels"):
    first = df.reset_index(drop=True).drop_duplicates(col).index.tolist()
    rest = [i for i in range(len(df)) if i not in set(first)][: max(0, n - len(first))]
    return sorted(first + rest)


def subset_items(src, dst, idx):
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        for new, old in enumerate(idx):
            fi.copy(fi[f"item_{old}"], fo, name=f"item_{new}")


def subset_rows_h5(src, dst, idx):
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        for k, v in fi.attrs.items():
            fo.attrs[k] = v
        for k in fi.keys():
            ds = fi[k]
            if ds.shape and ds.shape[0] == fi["images"].shape[0]:
                fo.create_dataset(k, data=ds[np.asarray(idx)], dtype=ds.dtype)
            else:
                fo.create_dataset(k, data=ds[()], dtype=ds.dtype)


def subset_lines(src, dst, idx, trailing=None):
    text = Path(src).read_text()
    lines = text.split()
    out = "\n".join(lines[i] for i in idx)
    if trailing is None:
        trailing = text.endswith("\n")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(out + ("\n" if trailing else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--train", type=int, default=256)
    ap.add_argument("--test", type=int, default=128)
    ap.add_argument("--nih", type=int, default=256)
    a = ap.parse_args()
    S, D = Path(a.src) / "mimic-lt", Path(a.dst) / "mimic-lt"
    (D / "test").mkdir(parents=True, exist_ok=True)

    tr, te = pd.read_csv(S / "mimic-cxr-lt_train.csv"), pd.read_csv(S / "mimic-cxr-lt_test.csv")
    itr, ite = pick(tr, a.train), pick(te, a.test)
    tr.iloc[itr].to_csv(D / "mimic-cxr-lt_train.csv", index=False)
    te.iloc[ite].to_csv(D / "mimic-cxr-lt_test.csv", index=False)

    for name in ("image_latents_flux-lt.h5", "embeddings_flux-lt.h5", "embeddings_mimic-lt_labels_train.h5",
                 "embeddings_mimic-lt_label_txt_train.h5"):
        if (S / name).exists():
            subset_items(S / name, D / name, itr)
    for name in ("embeddings_flux-lt_test.h5", "embeddings_mimic-lt_labels_test.h5",
                 "embeddings_mimic-lt_label_txt_test.h5"):
        if (S / name).exists():
            subset_items(S / name, D / name, ite)
    if (S / "null_embedding.h5").exists():
        shutil.copy(S / "null_embedding.h5", D / "null_embedding.h5")

    subset_lines(S / "labels_gt.txt", D / "labels_gt.txt", itr)
    subset_lines(S / "test" / "labels_gt.txt", D / "test" / "labels_gt.txt", ite)
    pr = S / "partitions_grasp_rebuttal"
    if pr.exists():
        for f in pr.glob("labels_*.txt"):
            subset_lines(f, D / "partitions_grasp_rebuttal" / f.name, itr)
        for f in (pr / "test").glob("labels_*.txt"):
            subset_lines(f, D / "partitions_grasp_rebuttal" / "test" / f.name, ite)
    tc = S / "text_cluster" / "train"
    if tc.exists():
        (D / "text_cluster" / "train").mkdir(parents=True, exist_ok=True)
        subset_lines(tc / "labels.txt", D / "text_cluster" / "train" / "labels.txt", itr)
        shutil.copy(tc / "centroids.npz", D / "text_cluster" / "train" / "centroids.npz")
    for split, idx in (("train", itr), ("test", ite)):
        f = S / f"mimic-lt_{split}.h5"
        if f.exists():
            subset_rows_h5(f, D / f"mimic-lt_{split}.h5", idx)
    nih = Path(a.src) / "nih-cxr-lt" / "classifier_eval.h5"
    if nih.exists():
        (Path(a.dst) / "nih-cxr-lt").mkdir(parents=True, exist_ok=True)
        with h5py.File(nih, "r") as fi:
            n = fi["images"].shape[0]
            targets = fi["targets"][()]
        first = sorted({int(np.where(targets == t)[0][0]) for t in np.unique(targets)})
        idx = sorted(set(first) | set(range(min(a.nih, n))))
        subset_rows_h5(nih, Path(a.dst) / "nih-cxr-lt" / "classifier_eval.h5", idx)
    print(f"tiny data root: {a.dst}  (train {len(itr)}, test {len(ite)} rows)")


if __name__ == "__main__":
    main()
