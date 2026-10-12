#!/usr/bin/env python3
"""Consolidate classifier `best_metrics.json` files into one CSV.

Compared to `consolidate_class_res.py`, this version supports multiple
`no_resample`-style roots with potentially different intermediate directory
layouts. It walks each root recursively, finds every `best_metrics.json`, and
infers (dataset_seed, model, classifier_seed) by looking for path components
matching the patterns:

    dataset_seed:    matches `^seed\\d+$`  (e.g., seed0, seed5868, seed9090)
    classifier_seed: matches `^seed_\\d+$` (e.g., seed_000)

The model is taken to be the directory that sits between dataset_seed and
classifier_seed in the path. This works for both layouts the user has:

  Layout A (legacy):
    $WORKSPACE/classifier_results/no_resample/<dataset_seed>/<model>/<cls_seed>/best_metrics.json

  Layout B (moclegate / GRASP variants):
    $WORKSPACE/data/grasp_v2/classifier_results/no_resample/mimic-lt/<dataset_seed>/<model>/<cls_seed>/best_metrics.json

Real-data runs are picked up via a separate `--real-dir`, e.g.
    $WORKSPACE/classifier_results/real/mimic/<cls_seed>/best_metrics.json
and assigned model='real', dataset_seed='mimic'.

Usage (on the cluster):
    python utils/consolidate_class_res_v2.py \
        --root $WORKSPACE/classifier_results/no_resample \
        --root $WORKSPACE/data/grasp_v2/classifier_results/no_resample/mimic-lt \
        --real-dir $WORKSPACE/classifier_results/real \
        --out consolidated_classifier_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

DATASET_SEED_RE = re.compile(r"^seed\d+$")          # seed0, seed5868, ...
CLS_SEED_RE = re.compile(r"^seed_\d+$")             # seed_000, seed_001, ...


def norm_label(lbl: str) -> str:
    x = lbl.strip().lower()
    x = re.sub(r"\s+", "_", x)
    x = x.replace("/", "-")
    return x


def find_jsons(roots: List[Path], real_dir: Path | None) -> List[Tuple[Path, str]]:
    """Return a list of (json_path, layout_kind) where layout_kind is
    'no_resample' or 'real'. Handles non-existent roots silently."""
    out: List[Tuple[Path, str]] = []
    for root in roots:
        if root and root.exists():
            for p in root.glob("**/best_metrics.json"):
                out.append((p, "no_resample"))
    if real_dir and real_dir.exists():
        for p in real_dir.glob("**/best_metrics.json"):
            out.append((p, "real"))
    return out


def parse_no_resample(p: Path) -> Dict[str, str] | None:
    """Find the (dataset_seed, model, cls_seed) triple in the path by pattern.

    Looks for a `seedX` path component followed by some `model` directory and
    a `seed_NNN` cls-seed directory. Returns None if no such triple is found.
    """
    parts = p.parts
    for i, comp in enumerate(parts):
        if not DATASET_SEED_RE.match(comp):
            continue
        # Need at least: seedX / model / seed_NNN / best_metrics.json
        if i + 2 >= len(parts):
            continue
        cls_seed = parts[i + 2]
        if not CLS_SEED_RE.match(cls_seed):
            continue
        model = parts[i + 1]
        return {
            "root": "no_resample",
            "group": comp,        # dataset_seed
            "model": model,
            "seed": cls_seed,     # classifier_seed
            "dataset": "",
            "run_id": f"no_resample/{comp}/{model}/{cls_seed}",
        }
    return None


def parse_real(p: Path, real_dir: Path) -> Dict[str, str] | None:
    parts = p.parts
    try:
        idx = parts.index(real_dir.name)
    except ValueError:
        return None
    if idx + 2 >= len(parts):
        return None
    dataset = parts[idx + 1]
    cls_seed = parts[idx + 2]
    if not CLS_SEED_RE.match(cls_seed):
        return None
    return {
        "root": "real",
        "group": dataset,
        "model": "real",
        "seed": cls_seed,
        "dataset": dataset,
        "run_id": f"real/{dataset}/{cls_seed}",
    }


def collect(items: List[Tuple[Path, str]], real_dir: Path | None):
    rows = []
    label_order_set: set[str] = set()
    parsed_cache = []

    for p, kind in items:
        try:
            with p.open("r") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if kind == "no_resample":
            meta = parse_no_resample(p)
        else:
            meta = parse_real(p, real_dir) if real_dir else None
        if meta is None:
            continue
        per_class = (data or {}).get("per_class", {}) or {}
        label_order_set.update(per_class.keys())
        parsed_cache.append((p, data, meta))

    label_order = sorted(label_order_set, key=lambda s: norm_label(s))
    return parsed_cache, label_order


def write_csv(parsed_cache, label_order: List[str], out_csv: Path):
    base_cols = [
        "root", "group", "dataset", "model", "seed", "run_id",
        "macro_f1", "balanced_acc", "json_path",
    ]
    metric_cols = []
    for lbl in label_order:
        n = norm_label(lbl)
        metric_cols.extend([f"{n}_support", f"{n}_pred", f"{n}_f1"])
    header = base_cols + metric_cols

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for p, data, meta in parsed_cache:
            aggregates = (data or {}).get("aggregates", {}) or {}
            per_class = (data or {}).get("per_class", {}) or {}
            row = {
                **meta,
                "macro_f1": aggregates.get("macro_f1", ""),
                "balanced_acc": aggregates.get("balanced_acc", ""),
                "json_path": str(p),
            }
            for lbl in label_order:
                n = norm_label(lbl)
                stats = per_class.get(lbl, {}) or {}
                row[f"{n}_support"] = stats.get("support", 0)
                row[f"{n}_pred"] = stats.get("pred", 0)
                row[f"{n}_f1"] = stats.get("f1", 0.0)
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, action="append", default=[],
                    help="A directory to walk for best_metrics.json (no_resample-style). "
                         "May be passed multiple times.")
    ap.add_argument("--real-dir", type=Path, default=None,
                    help="Optional 'real' directory containing real-data runs "
                         "as <real_dir>/<dataset>/<cls_seed>/best_metrics.json.")
    ap.add_argument("--out", type=Path, default=Path("consolidated_classifier_results.csv"))
    args = ap.parse_args()

    if not args.root and not args.real_dir:
        ap.error("Pass at least one --root or --real-dir.")

    items = find_jsons(args.root, args.real_dir)
    if not items:
        print("No best_metrics.json files found in the given roots.")
        return

    parsed_cache, label_order = collect(items, args.real_dir)
    if not parsed_cache:
        print("Found JSONs but none could be parsed into the expected layout.")
        return

    write_csv(parsed_cache, label_order, args.out)

    # Summary
    from collections import defaultdict
    counts = defaultdict(int)
    for _, _, meta in parsed_cache:
        counts[(meta["model"], meta["group"])] += 1
    print(f"Wrote {len(parsed_cache)} runs to {args.out}")
    print(f"Distinct labels ({len(label_order)}): {', '.join(label_order[:6])}{' ...' if len(label_order) > 6 else ''}")
    print(f"\n{'model':40s}  {'dataset_seed':14s}  {'n_runs':>6s}")
    for (model, ds_seed), n in sorted(counts.items()):
        print(f"{model:40s}  {ds_seed:14s}  {n:6d}")


if __name__ == "__main__":
    main()
