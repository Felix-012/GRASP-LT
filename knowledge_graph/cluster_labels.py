import random
from typing import List, Tuple, Optional, Iterable, Dict, Any

import numpy as np
from pyroaring import BitMap
import array
import math
import matplotlib.pyplot as plt
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    confusion_matrix, f1_score, classification_report,
)

def compute_index_frequencies(bitmaps: List[BitMap]) -> Dict[int, int]:
    counter = Counter()
    for bm in bitmaps:
        counter.update(bm)
    return counter


def compute_rarity_weights(freqs: Dict[int, int], N: int) -> Dict[int, float]:
    return {i: math.log(N / (1 + f)) for i, f in freqs.items()}

def weighted_jaccard(a: BitMap, b: BitMap, weights: Dict[int, float]) -> float:
    inter = a & b
    union = a | b

    inter_weight = sum(weights.get(i, 0.0) for i in inter)
    union_weight = sum(weights.get(i, 0.0) for i in union)

    return inter_weight / union_weight if union_weight > 0 else 0.0


# -----------------------------
# Similarity and LSH kNN
# -----------------------------
def jaccard(a: BitMap, b: BitMap) -> float:
    ua, ub = len(a), len(b)
    if ua == 0 and ub == 0:
        return 1.0
    inter = len(a & b)
    union = ua + ub - inter
    return inter / union if union else 0.0

def build_knn_adjacency(
    bitmaps: List[BitMap],
    jac_weights = None,
    k: int = 24,
    num_perm: int = 64,
    lsh_type: str = "forest",     # "forest" or "lsh"
    lsh_params: Optional[dict] = None,
    bytes_per_int: int = 8        # 4 if your IDs fit in 32-bit
) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    Returns:
      adj       : undirected adjacency list (neighbors per node), sized to non-empty items
      idx_nonempty : mapping local->global index for non-empty bitmaps
      idx_empty    : list of global indices for empty bitmaps
    """
    from datasketch import MinHash, MinHashLSH, MinHashLSHForest

    idx_nonempty = [i for i,bm in enumerate(bitmaps) if len(bm) > 0]
    idx_empty    = [i for i,bm in enumerate(bitmaps) if len(bm) == 0]
    X = [bitmaps[i] for i in idx_nonempty]
    n = len(X)
    if n == 0:
        return [], idx_nonempty, idx_empty

    # MinHash sketches
    mhs = []
    for bm in X:
        mh = MinHash(num_perm=num_perm)
        for x in bm:
            mh.update(int(x).to_bytes(bytes_per_int, "little", signed=False))
        mhs.append(mh)

    # LSH index
    if lsh_type == "forest":
        forest = MinHashLSHForest(num_perm=num_perm)
        for i, mh in enumerate(mhs):
            forest.add(str(i), mh)
        forest.index()
        def query(i, topk):
            return list(map(int, forest.query(mhs[i], topk)))
    else:
        params = {"threshold": 0.5}
        if lsh_params: params.update(lsh_params)
        lsh = MinHashLSH(num_perm=num_perm, **params)
        for i, mh in enumerate(mhs):
            lsh.insert(str(i), mh)
        def query(i, topk):
            # LSH returns a set; keep up to topk arbitrary (we will re-rank)
            res = list(map(int, lsh.query(mhs[i])))
            return res[:max(topk, 2*k)]

    # Directed kNN by exact Jaccard re-ranking
    directed = [set() for _ in range(n)]
    for i in range(n):
        cand = [j for j in query(i, k*3) if j != i]
        if not cand:
            continue
        ai = X[i]
        if jac_weights is None:
            scored = [(jaccard(ai, X[j]), j) for j in cand]
        else:
            scored = [(weighted_jaccard(ai, X[j], jac_weights), j) for j in cand]
        scored.sort(reverse=True)
        for _, j in scored[:k]:
            directed[i].add(j)

    # Symmetrize
    for i in range(n):
        for j in list(directed[i]):
            directed[j].add(i)

    adj = [sorted(list(directed[i])) for i in range(n)]
    return adj, idx_nonempty, idx_empty

# ========= Single METIS partition =========

def _metis_partition(adj: List[List[int]], k: int, ufactor: int = 500) -> Tuple[int, List[int]]:
    """
    ufactor in [1..1000] controls allowed imbalance (~1 + ufactor/1000).
    Example: ufactor=500 -> ~1.5x load cap (loose balance).
    """
    import metis
    edgecuts, parts = metis.part_graph(
        adj,
        nparts=k,
        ufactor=ufactor,     # loosen balance to allow skewed cuts
        recursive=False,     # direct k-way (better for k>2)
        ncuts=1,             # one refinement round is fine for speed
        niter=10,            # small number of refinement iterations
    )
    return edgecuts, list(parts)

# ========= Public API =========

def single_kway_partition_from_bitmaps(
    bitmaps: List[BitMap],
    K: int = 3,
    jac_weights=None,
    *,
    knn: int = 24,
    num_perm: int = 64,
    ufactor: int = 500,                 # increase to allow more imbalance
    empties_strategy: str = "separate", # or "separate"
    bytes_per_int: int = 8,
) -> Tuple[List[int], dict]:
    """
    Returns:
      labels : list[int] of length len(bitmaps), values in {0..K-1} (or K if 'separate' empties)
      meta   : dict(edgecuts=..., sizes=..., K=..., ufactor=...)
    """
    from collections import Counter

    adj, idx_nonempty, idx_empty = build_knn_adjacency(
        bitmaps, jac_weights, k=knn, num_perm=num_perm, bytes_per_int=bytes_per_int
    )
    if not idx_nonempty:  # all empty
        labels = [ (K if empties_strategy == "separate" else 0) ] * len(bitmaps)
        return labels, {"note": "all empty", "K": K, "edgecuts": 0, "sizes": Counter(labels)}

    edgecuts, parts_local = _metis_partition(adj, k=K, ufactor=ufactor)

    # Map back to original order
    labels = [None] * len(bitmaps)
    for loc_i, glob_i in enumerate(idx_nonempty):
        labels[glob_i] = int(parts_local[loc_i])

    if empties_strategy == "separate":
        for gi in idx_empty:
            labels[gi] = K
    else:  # assign empties to current smallest part by node count
        from collections import Counter
        counts = Counter([labels[i] for i in idx_nonempty])
        smallest = min(range(K), key=lambda c: counts.get(c, 0))
        for gi in idx_empty:
            labels[gi] = smallest

    # Meta
    sizes = {c: labels.count(c) for c in set(labels)}
    return labels, {"edgecuts": edgecuts, "sizes": sizes, "K": K, "ufactor": ufactor}



from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
from pyroaring import FrozenBitMap
