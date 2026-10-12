import json
from pathlib import Path
from collections import Counter, defaultdict

import h5py
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans  # <-- library K-Means

from custom_datasets.embeddings import EmbeddingDataset


# ----------------------- helpers ----------------------- #

def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _unit_normalize(X):
    """Row-wise L2 normalize in-place-safe manner. Raises if zero-norm or NaNs exist."""
    X = X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(norms)):
        raise ValueError("Embeddings contain NaN/Inf.")
    if np.any(norms == 0):
        raise ValueError("Zero-norm rows found; cannot use cosine/spherical k-means.")
    return X / norms

def save_single_stacked_cluster_chart(counters, K, out_path, *, top_n=20):
    """
    Single figure with K bars (one per cluster), stacked by GT labels.
    Uses top_n labels globally; the rest are grouped into 'Other'.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) pick top_n labels globally
    total_per_label = Counter()
    for i in range(K):
        total_per_label.update(counters[i])
    top_labels = [lab for lab, _ in total_per_label.most_common(top_n)]

    # 2) build (K, top_n + 1) matrix with 'Other' as last column
    counts = np.zeros((K, len(top_labels) + 1), dtype=int)
    for i in range(K):
        total_i = sum(counters[i].values())
        other = total_i
        for j, lab in enumerate(top_labels):
            v = counters[i].get(lab, 0)
            counts[i, j] = v
            other -= v
        counts[i, -1] = max(0, other)  # 'Other'

    # 3) plot single stacked bar chart (no explicit colors)
    fig = plt.figure()
    x = np.arange(K)
    bottom = np.zeros(K, dtype=int)
    legend_labels = top_labels + ["Other"]
    for j, lab in enumerate(legend_labels):
        plt.bar(x, counts[:, j], bottom=bottom, label=lab)
        bottom += counts[:, j]

    plt.xticks(x, [f"Cluster {i}" for i in range(K)])
    plt.ylabel("Count")
    #plt.title("Ground-truth label composition by cluster")
    plt.tight_layout()
    fig.savefig(out_path, dpi=1000)
    plt.close(fig)
    print(f"Saved {out_path}")

def _nearest_reps_cosine(X_raw, X_fit, labels, centers):
    """Return rep_idx/rep_vecs using cosine similarity on normalized X_fit."""
    K = centers.shape[0]
    rep_idx, rep_vecs = {}, {}
    for k in range(K):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            continue
        sims = X_fit[idx] @ centers[k]
        j_local = int(np.argmax(sims))
        gidx = int(idx[j_local])
        rep_idx[k] = gidx
        rep_vecs[k] = X_raw[gidx].copy()
    return rep_idx, rep_vecs


# ----------------------- K-Means (library) ----------------------- #

def run_cluster_library(
    X,
    *,
    K,
    algorithm="vmf",       # "vmf" | "bisect_kmeans" | "kmeans"
    metric="cosine",       # "cosine" or "euclidean" (used for kmeans/bisect_kmeans)
    n_init=32,
    max_iter=500,
    seed=42,
    verbose=False
):
    """
    Returns:
      labels: (n,) int in 0..K-1
      rep_idx: {k -> index of nearest sample to center (cosine for vmf/cosine setups)}
      rep_vecs: {k -> original-space vector of that representative}
      centers: (K, d) ndarray (unit-normalized for cosine/vmf; means for euclidean)
      meta: dict
    """
    X = _as_numpy(X).astype(np.float32, copy=False)
    n, d = X.shape

    if algorithm == "vmf":
        # Requires: pip install spherecluster
        from spherecluster import VonMisesFisherMixture
        X_fit = _unit_normalize(X)
        vmf = VonMisesFisherMixture(
            n_clusters=K,
            posterior_type="hard",     # "soft" also available if you want probabilities
            n_init=n_init,
            max_iter=max_iter,
            random_state=seed,
        )
        labels = vmf.fit_predict(X_fit).astype(int)
        # centers attribute name differs across versions; try a few:
        centers = getattr(vmf, "cluster_centers_", None)
        if centers is None:
            centers = getattr(vmf, "means_", None)
        if centers is None:
            # fallback: empirical mean directions
            centers = np.stack([
                _unit_normalize(X_fit[labels == k].mean(axis=0, keepdims=True))[0]
                for k in range(K)
            ], axis=0)
        rep_idx, rep_vecs = _nearest_reps_cosine(X, X_fit, labels, centers)
        meta = {
            "algorithm": "vmf-mixture",
            "K": int(K),
            "n_init": int(n_init),
            "max_iter": int(max_iter),
            "seed": int(seed),
            # Optional extras if present:
            "concentrations": np.asarray(getattr(vmf, "concentrations_", []), dtype=np.float32).tolist()
        }
        return labels, rep_idx, rep_vecs, centers.astype(np.float32), meta

    elif algorithm == "bisect_kmeans":
        # scikit-learn’s hierarchical k-means; use cosine by L2-normalizing first
        from sklearn.cluster import BisectingKMeans
        if metric == "cosine":
            X_fit = _unit_normalize(X)
        else:
            X_fit = X
        bk = BisectingKMeans(
            n_clusters=K,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=seed,
            # splits the cluster with largest inertia (often best for long tails)
            bisecting_strategy="biggest_inertia",
        )
        labels = bk.fit_predict(X_fit).astype(int)
        centers = bk.cluster_centers_.astype(np.float32)
        if metric == "cosine":
            centers = _unit_normalize(centers)
            rep_idx, rep_vecs = _nearest_reps_cosine(X, X_fit, labels, centers)
        else:
            # Euclidean reps
            rep_idx, rep_vecs = {}, {}
            for k in range(K):
                idx = np.where(labels == k)[0]
                if idx.size == 0:
                    continue
                Xk = X_fit[idx]
                c = centers[k]
                x2 = np.einsum("ij,ij->i", Xk, Xk)
                c2 = float(np.dot(c, c))
                dots = Xk @ c
                d2 = x2 + c2 - 2.0 * dots
                j_local = int(np.argmin(d2))
                gidx = int(idx[j_local])
                rep_idx[k] = gidx
                rep_vecs[k] = X[gidx].copy()
        meta = {
            "algorithm": "bisecting-kmeans",
            "metric": metric,
            "K": int(K),
            "n_init": int(n_init),
            "max_iter": int(max_iter),
            "seed": int(seed),
            "inertia_": float(getattr(bk, "inertia_", float("nan"))),
        }
        return labels, rep_idx, rep_vecs, centers, meta
    elif algorithm == "kmedoids":
        if metric not in {"cosine", "euclidean"}:
            raise ValueError(f"k-medoids supports metric 'cosine' or 'euclidean', got: {metric}")

        try:
            # requires: pip install scikit-learn-extra
            from sklearn_extra.cluster import KMedoids
        except ImportError as e:
            raise ImportError(
                "k-medoids requires scikit-learn-extra. "
                "Please install it with `pip install scikit-learn-extra`."
            ) from e

        X_fit = _unit_normalize(X) if metric == "cosine" else X

        km = KMedoids(
            n_clusters=K,
            metric=metric,          # 'cosine' or 'euclidean'
            method="pam",
            max_iter=max_iter,
            random_state=seed,
        )
        labels = km.fit_predict(X_fit).astype(int)
        medoid_indices = np.asarray(km.medoid_indices_, dtype=int)

        # For cosine, keep your convention: unit-normalized "centers" (medoid vectors).
        centers = (
            _unit_normalize(X[medoid_indices]) if metric == "cosine"
            else X[medoid_indices]
        ).astype(np.float32)

        rep_idx  = {k: int(medoid_indices[k]) for k in range(K)}
        rep_vecs = {k: X[int(medoid_indices[k])].copy() for k in range(K)}

        meta = {
            "algorithm": "kmedoids",
            "impl": "sklearn-extra",
            "metric": metric,
            "K": int(K),
            "max_iter": int(max_iter),
            "seed": int(seed),
            "inertia_": float(getattr(km, "inertia_", float("nan"))),
            "medoid_indices": medoid_indices.tolist(),
        }
        return labels, rep_idx, rep_vecs, centers, meta
    else:
        # fallback to your standard k-means (as you already have)
        from sklearn.cluster import KMeans
        X_fit = _unit_normalize(X) if metric == "cosine" else X
        km = KMeans(
            n_clusters=K,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=seed,
            verbose=1 if verbose else 0,
        )
        labels = km.fit_predict(X_fit).astype(int)
        centers = km.cluster_centers_.astype(np.float32)
        if metric == "cosine":
            centers = _unit_normalize(centers)
            rep_idx, rep_vecs = _nearest_reps_cosine(X, X_fit, labels, centers)
        else:
            rep_idx, rep_vecs = {}, {}
            for k in range(K):
                idx = np.where(labels == k)[0]
                if idx.size == 0:
                    continue
                Xk = X_fit[idx]
                c = centers[k]
                x2 = np.einsum("ij,ij->i", Xk, Xk)
                c2 = float(np.dot(c, c))
                dots = Xk @ c
                d2 = x2 + c2 - 2.0 * dots
                j_local = int(np.argmin(d2))
                gidx = int(idx[j_local])
                rep_idx[k] = gidx
                rep_vecs[k] = X[gidx].copy()
        meta = {
            "algorithm": "kmeans (sklearn)",
            "metric": metric,
            "K": int(K),
            "n_init": int(n_init),
            "max_iter": int(max_iter),
            "seed": int(seed),
            "inertia_": float(getattr(km, "inertia_", float("nan"))),
        }
        return labels, rep_idx, rep_vecs, centers, meta
