#!/usr/bin/env python3
"""Text-embedding partition ("Embed" in the paper; also used for ImageNet-LT and MoCLE).

Bisecting k-means (cosine) on the pooled FLUX text embeddings of the training prompts.
Writes `labels.txt` (one cluster id per training row) and `centroids.npz` (key
"centroids", unit-normalised, shape (K, D)) as expected by the textlabel and MoCLE routers.
Optionally assigns a second embedding file (e.g. the test prompts) to the nearest centroid.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom_datasets.embeddings import EmbeddingDataset  # noqa: E402
from knowledge_graph.cluster_textdiff import run_cluster_library  # noqa: E402


def pooled(path):
    x = EmbeddingDataset(path).load_ppe()
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_embeddings", required=True, help="embeddings HDF5 of the training prompts")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_init", type=int, default=100)
    ap.add_argument("--max_iter", type=int, default=1000)
    ap.add_argument("--assign_embeddings", help="optional: embeddings HDF5 to assign to the nearest centroid")
    ap.add_argument("--assign_out", help="output labels file for --assign_embeddings")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X = pooled(args.train_embeddings)
    labels, _, _, centers, meta = run_cluster_library(
        X, K=args.K, algorithm="bisect_kmeans", metric="cosine",
        n_init=args.n_init, max_iter=args.max_iter, seed=args.seed)
    (out / "labels.txt").write_text("\n".join(map(str, labels)))
    np.savez(out / "centroids.npz", centroids=centers.astype(np.float32))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("cluster sizes:", dict(sorted(collections.Counter(labels.tolist()).items())))

    if args.assign_embeddings:
        Y = pooled(args.assign_embeddings)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True).clip(min=1e-12)
        ids = (Y @ centers.T).argmax(1)
        dest = Path(args.assign_out or out / "assigned_labels.txt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(map(str, ids)))
        print(f"assigned {len(ids)} rows -> {dest}:", dict(sorted(collections.Counter(ids.tolist()).items())))


if __name__ == "__main__":
    main()
