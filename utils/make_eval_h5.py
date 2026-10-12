#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pack an evaluation set into a single HDF5 quickly.

Speed tricks:
  • Default mode stores RAW FILE BYTES (no PIL) -> massively faster.
  • Parallel reads/encodes with ThreadPool (raw) or ProcessPool (png).
  • Vectorized HDF5 writes in batches to avoid per-item overhead.

Datasets:
  - images   : vlen uint8 (raw bytes per file; PNG/JPEG/etc.)
  - targets  : int64
  - paths    : vlen utf-8 strings (absolute paths used)
  - ids      : (optional) vlen utf-8 strings if --id_col given
Attrs:
  - classes  : vlen utf-8 strings (stable order from label_col)
  - label_col, path_col, id_col
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import io
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import h5py
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

try:
    from PIL import Image
except Exception:
    Image = None  # only needed for --mode png

# ---------- helpers ----------

def _resolve_path(p: str, root_prefix: Optional[str], csv_dir: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    if root_prefix and Path(root_prefix).is_dir():
        return Path(root_prefix) / pp
    return csv_dir / pp

def _read_raw_bytes(path: Path) -> bytes:
    # Fast path: just read file bytes, no decoding/re-encoding
    with open(path, "rb") as f:
        return f.read()

def _encode_png_rgb(path: Path) -> bytes:
    # CPU-heavy path: decode -> RGB -> PNG encode
    if Image is None:
        raise RuntimeError("Pillow required for --mode png. Install pillow or pillow-simd.")
    with Image.open(path) as im:
        im = im.convert("RGB")
        bio = io.BytesIO()
        im.save(bio, format="PNG", optimize=False)  # optimize=False is faster
        return bio.getvalue()

def _build_class_map(labels: List[str]):
    classes, class_to_idx = [], {}
    targets = np.empty(len(labels), dtype=np.int64)
    for i, name in enumerate(labels):
        if name not in class_to_idx:
            class_to_idx[name] = len(classes)
            classes.append(name)
        targets[i] = class_to_idx[name]
    return classes, class_to_idx, targets

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with label/path columns.")
    ap.add_argument("--out", required=True, help="Output HDF5 file.")
    ap.add_argument("--root", default=None, help="Optional base dir for resolving relative CSV paths.")
    ap.add_argument("--label_col", default="Finding Labels")
    ap.add_argument("--path_col", default="path")
    ap.add_argument("--id_col", default=None, help="Optional id column to store.")
    ap.add_argument("--mode", choices=["raw", "png"], default="raw",
                    help="raw = store original file bytes (fastest). png = re-encode to RGB PNG (slower).")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8,
                    help="Parallel workers for I/O/encode.")
    ap.add_argument("--batch", type=int, default=2048, help="HDF5 write batch size.")
    ap.add_argument("--fail_on_missing", action="store_true", help="Error on missing images instead of skipping.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    if args.label_col not in df.columns:
        raise ValueError(f"CSV {csv_path} missing '{args.label_col}'")
    if args.path_col not in df.columns:
        raise ValueError(f"CSV {csv_path} missing '{args.path_col}'")

    labels_raw = df[args.label_col].astype(str).tolist()
    paths_raw  = df[args.path_col].astype(str).tolist()
    ids_raw    = df[args.id_col].astype(str).tolist() if args.id_col and args.id_col in df.columns else None

    # Build class map (stable order)
    classes, class_to_idx, targets_full = _build_class_map(labels_raw)

    # Resolve paths & filter missing early
    csv_dir = csv_path.parent
    resolved: List[Path] = []
    kept_targets: List[int] = []
    kept_ids: List[str] | None = ([] if ids_raw is not None else None)
    kept_paths_str: List[str] = []

    for i, p in enumerate(paths_raw):
        rp = _resolve_path(p, args.root, csv_dir)
        if not rp.is_file():
            if args.fail_on_missing:
                raise FileNotFoundError(f"Missing image: {rp}")
            continue
        resolved.append(rp)
        kept_targets.append(int(targets_full[i]))
        kept_paths_str.append(str(rp))
        if kept_ids is not None:
            kept_ids.append(ids_raw[i] if ids_raw is not None else "")

    n = len(resolved)
    if n == 0:
        raise RuntimeError("No images to write (after resolving and filtering).")

    # Choose encoder + pool
    if args.mode == "raw":
        encoder = _read_raw_bytes
        Pool = ThreadPoolExecutor  # I/O-bound
    else:
        encoder = _encode_png_rgb
        Pool = ProcessPoolExecutor  # CPU-bound

    # Prepare HDF5
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    str_dtype = h5py.string_dtype(encoding="utf-8")
    try:
        vlen_bytes = h5py.vlen_dtype(np.dtype("uint8"))
    except AttributeError:
        # older h5py fallback
        vlen_bytes = h5py.special_dtype(vlen=np.uint8)

    # Avoid libver="latest" so files remain readable by evaluation containers
    # built against older HDF5 libraries.
    with h5py.File(out_path, "w") as h5:
        # Precreate datasets
        d_images  = h5.create_dataset("images", shape=(n,), dtype=vlen_bytes)  # no compression -> faster
        d_targets = h5.create_dataset("targets", data=np.asarray(kept_targets, dtype=np.int64), dtype=np.int64)
        d_paths   = h5.create_dataset("paths", (n,), dtype=str_dtype)
        d_ids     = h5.create_dataset("ids", (n,), dtype=str_dtype) if kept_ids is not None else None

        # classes as dataset
        h5.create_dataset("classes", data=np.array(classes, dtype=object), dtype=str_dtype)
        h5.attrs["label_col"] = args.label_col
        h5.attrs["path_col"]  = args.path_col
        if args.id_col:
            h5.attrs["id_col"] = args.id_col

        # Write paths/ids in bulk (fast)
        d_paths[:] = np.asarray(kept_paths_str, dtype=object)
        if d_ids is not None:
            d_ids[:] = np.asarray(kept_ids, dtype=object)

        # Parallel read/encode and batched write (CORRECT vlen writes)
        batch = max(1, int(args.batch))
        workers = max(1, int(args.workers))
        with Pool(max_workers=workers) as pool:
            futures = [pool.submit(encoder, rp) for rp in resolved]

            pbar = tqdm(total=n, desc=f"Writing ({args.mode})")
            start = 0
            while start < n:
                end = min(start + batch, n)
                k = end - start

                # Build a chunk explicitly typed as vlen uint8.
                # Each element must be a NumPy uint8 array (not Python bytes).
                chunk = np.empty(k, dtype=vlen_bytes)
                for j in range(k):
                    b = futures[start + j].result()           # bytes
                    arr = np.frombuffer(b, dtype=np.uint8)    # view
                    chunk[j] = np.array(arr, copy=True)       # own memory; robust for h5py

                d_images[start:end] = chunk                   # correct vlen write
                start = end
                pbar.update(k)
            pbar.close()

    print(f"Done. Wrote {n} images, {len(classes)} classes -> {out_path}")
    if args.mode == "raw":
        print("Tip: use an H5 loader that decodes from raw bytes and converts to RGB on-the-fly.")

if __name__ == "__main__":
    main()
