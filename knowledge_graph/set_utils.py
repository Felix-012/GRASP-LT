from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Optional, Sequence, Set, Union

import numpy as np
from tqdm.auto import tqdm


def _require_pyroaring():
    try:
        from pyroaring import BitMap, FrozenBitMap
    except Exception as exc:
        raise ImportError(
            "Optional bitmap/set-router utilities require `pyroaring`. "
            "Install it if you use load_bitmaps(), to_bitmaps(), or k_medoids() "
            "on Roaring bitmap inputs."
        ) from exc
    return BitMap, FrozenBitMap


def load_mapping(path: str | Path) -> dict[str, str]:
    """Load a string-to-string JSON mapping."""
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def load_bitmap_strings(
    filepath: str | Path,
    *,
    start_id: int = 0,
    strip: bool = True,
    skip_blank: bool = False,
    return_type=str,
) -> dict[int, Any]:
    mapping: dict[int, Any] = {}
    with Path(filepath).expanduser().open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp, start=start_id):
            if strip:
                line = line.rstrip()
            if skip_blank and not line:
                continue
            mapping[idx] = return_type(line)
    return mapping


def to_bitmaps(list_of_sets: Iterable[Set[str]]):
    BitMap, _ = _require_pyroaring()
    tok2id: dict[str, int] = {}
    id2tok: list[str] = []
    bitmaps = []

    for tokens in list_of_sets:
        bitmap = BitMap()
        for token in tokens:
            idx = tok2id.get(token)
            if idx is None:
                idx = len(id2tok)
                tok2id[token] = idx
                id2tok.append(token)
            bitmap.add(idx)
        bitmaps.append(bitmap)

    return bitmaps, {i: token for i, token in enumerate(id2tok)}, tok2id


def write_records(
    bitmap_fh: BinaryIO,
    bit_sets: Sequence,
    cui_sets: Optional[Sequence[Set[str]]] = None,
    cui_fh: Optional[BinaryIO] = None,
) -> None:
    for bitmap_idx, bitmap in enumerate(bit_sets):
        raw = bitmap.serialize()
        bitmap_fh.write(len(raw).to_bytes(4, "little"))
        bitmap_fh.write(raw)
        if cui_fh is not None and cui_sets is not None:
            cui_fh.write(",".join(cui_sets[bitmap_idx]).encode("utf-8") + b"\n")


def load_bitmaps(path: str | Path) -> list:
    _, FrozenBitMap = _require_pyroaring()
    out = []
    buf = Path(path).expanduser().read_bytes()
    i = 0
    while i < len(buf):
        n = int.from_bytes(buf[i:i + 4], "little")
        out.append(FrozenBitMap.deserialize(buf[i + 4:i + 4 + n]))
        i += 4 + n
    return out


def jaccard_dist(a, b) -> float:
    dist = 1.0 - a.jaccard_index(b)
    if dist != dist:
        return 0.0
    return float(dist)


def make_dist_cached(bitmaps, dist_fn_to_cache: Callable, maxsize: int = 2_000_000):
    @lru_cache(maxsize=maxsize)
    def _dist(i: int, j: int) -> float:
        if j < i:
            i, j = j, i
        if i == j:
            return 0.0
        return float(dist_fn_to_cache(bitmaps[i], bitmaps[j]))

    return _dist


def _assign(bitmaps, centers, dist_fn):
    labels = [-1] * len(bitmaps)
    clusters = [[] for _ in centers]
    dists_all = [[dist_fn(i, c) for c in centers] for i in range(len(bitmaps))]
    order = np.argsort([min(row) for row in dists_all])

    for idx in order:
        costs = [dists_all[idx][j] for j in range(len(centers))]
        c = int(np.argmin(costs))
        clusters[c].append(int(idx))
        labels[int(idx)] = c

    return labels, clusters


def _farthest_first(bitmaps, k: int, dist_fn):
    centers = [0]
    dist = np.array([dist_fn(0, i) for i in range(len(bitmaps))], dtype=np.float32)
    for _ in range(k - 1):
        idx = int(dist.argmax())
        centers.append(idx)
        new = np.array([dist_fn(idx, i) for i in range(len(bitmaps))], dtype=np.float32)
        dist = np.minimum(dist, new)
    return centers


def k_medoids(
    bitmaps: Sequence,
    k: int,
    max_iter: int = 10,
    *,
    dist_fn: Callable[[Any, Any], float],
    cache_size: int = 2_000_000,
    seed: Optional[int] = None,
    enforce_min_sep: Optional[float] = None,
):
    """Small k-medoids implementation used by the optional set router."""
    del seed, enforce_min_sep
    if k <= 0:
        raise ValueError("k must be positive.")
    if len(bitmaps) == 0:
        raise ValueError("Cannot cluster an empty bitmap collection.")

    k = min(int(k), len(bitmaps))
    dist_cached = make_dist_cached(bitmaps, dist_fn, maxsize=cache_size)
    centers = _farthest_first(bitmaps, k, dist_cached)

    for _ in tqdm(range(max_iter), desc="k-medoids", position=0):
        labels, clusters = _assign(bitmaps, centers, dist_cached)
        new_centers = []
        for center, members in zip(centers, clusters):
            if not members:
                new_centers.append(center)
                continue
            scores = []
            for i in members:
                denom = max(1, len(members) - 1)
                score = sum(dist_cached(i, j) for j in members if j != i) / denom
                scores.append((score, i))
            new_centers.append(min(scores)[1])
        if new_centers == centers:
            break
        centers = new_centers

    labels, _ = _assign(bitmaps, centers, dist_cached)
    return labels, centers


def compute_distance_matrix(
    out_path: Union[Path, str],
    bitmaps: Sequence,
    dist_func: Callable = jaccard_dist,
):
    out_path = Path(out_path).with_suffix(".npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(bitmaps)
    distances = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            dist = float(dist_func(bitmaps[i], bitmaps[j]))
            distances[i, j] = dist
            distances[j, i] = dist
    np.save(out_path, distances)
    return distances
