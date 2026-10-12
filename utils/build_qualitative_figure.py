#!/usr/bin/env python
"""Build a publication-ready qualitative figure for the GRASP paper.

Each input class folder contains files of the form:
  <CLASS>__<rowid>__row<num>__real.png
  <CLASS>__<rowid>__row<num>__synth_finetune__idx*.png
  <CLASS>__<rowid>__row<num>__synth_mocle__idx*.png
  <CLASS>__<rowid>__row<num>__synth_adapter__idx*.png
  <CLASS>__<rowid>__row<num>__synth_grasp__idx*.png

For each chosen class we display one real image plus four GRASP samples by
default. For the compact diversity layout, use FT, MoCLE, and three GRASP
samples without Real. The adapter baseline is intentionally excluded.
Output is a vector PDF with embedded rasters.

Usage:
  python utils/build_qualitative_figure.py \
      --root rebuttal_images_ms \
      --classes Pneumomediastinum Edema Pleural_Effusion \
      --rows-per-class 1 \
      --out grasp_paper_v3/figures/qualitative_mimic.pdf

  python utils/build_qualitative_figure.py \
      --root rebuttal_images_imagenet \
      --classes Dutch_oven Turdus_migratorius chainsaw \
      --rows-per-class 1 \
      --out grasp_paper_v3/figures/qualitative_imagenet.pdf
"""
from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DEFAULT_SAMPLES_PER_CLASS = 4


ROW_RE = re.compile(r"^(?P<cls>.+?)__(?P<rowid>.+?)__(row|h5idx)(?P<num>\d+)__(?P<kind>[a-z_]+?)(?:__idx\d+)?\.(?:png|jpg|jpeg)$",
                    re.IGNORECASE)


def index_class_dir(class_dir: Path):
    """Return dict[rowprefix -> dict[col_key -> Path]]."""
    rows = defaultdict(dict)
    for p in sorted(class_dir.iterdir()):
        if not p.is_file():
            continue
        m = ROW_RE.match(p.name)
        if not m:
            continue
        prefix = f"{m.group('cls')}__{m.group('rowid')}__{m.group('num')}"
        kind_raw = m.group("kind").rstrip("_").lower()
        # Normalize kind to one of the COL_KEYS
        if kind_raw == "real":
            kind = "real"
        elif kind_raw.startswith("synth_"):
            kind = kind_raw  # synth_finetune/synth_mocle/synth_adapter/synth_grasp
        else:
            continue
        rows[prefix][kind] = p
    return dict(rows)


def pretty_class_name(cls: str) -> str:
    s = cls.replace("_", " ").strip()
    # Special-cases: ImageNet uses scientific names; keep one capital
    return s


def select_candidates(
    candidates: list[Path],
    n: int,
    selection_order: str,
    rng: random.Random,
) -> list[Path]:
    if selection_order == "first":
        return candidates[:n]
    if selection_order == "last":
        return list(reversed(candidates))[:n]
    if selection_order == "random":
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return shuffled[:n]
    raise ValueError(f"Unknown selection order: {selection_order}")


def choose_class_examples(
    class_dir: Path,
    samples_per_class: int,
    selection_order: str,
    rng: random.Random,
):
    """Return (real_path, [sample_paths]) for one class.

    The rebuttal extraction stores one generated sample per method and source
    row. For this qualitative figure we want a class-level panel, so we use the
    first real image as the anchor and collect several GRASP samples from
    deterministic source rows. Adapter images are never selected.
    """
    indexed = index_class_dir(class_dir)
    if not indexed:
        return None

    real_candidates = [
        cells["real"]
        for _, cells in sorted(indexed.items())
        if "real" in cells
    ]
    sample_candidates = [
        cells["synth_grasp"]
        for _, cells in sorted(indexed.items())
        if "synth_grasp" in cells
    ]
    if not real_candidates:
        return None
    if len(sample_candidates) < samples_per_class:
        print(
            f"WARNING: {class_dir} only has {len(sample_candidates)} GRASP "
            f"samples; expected {samples_per_class}"
        )
    real_path = select_candidates(real_candidates, 1, selection_order, rng)[0]
    samples = select_candidates(sample_candidates, samples_per_class, selection_order, rng)
    if len(samples) < samples_per_class:
        return None
    return real_path, samples


def choose_diversity_examples(
    class_dir: Path,
    grasp_samples_per_class: int,
    selection_order: str,
    rng: random.Random,
):
    """Return [finetune, mocle, grasp...] for a compact diversity panel."""
    indexed = index_class_dir(class_dir)
    if not indexed:
        return None

    finetune_candidates = [
        cells["synth_finetune"]
        for _, cells in sorted(indexed.items())
        if "synth_finetune" in cells
    ]
    mocle_candidates = [
        cells["synth_mocle"]
        for _, cells in sorted(indexed.items())
        if "synth_mocle" in cells
    ]
    grasp_candidates = [
        cells["synth_grasp"]
        for _, cells in sorted(indexed.items())
        if "synth_grasp" in cells
    ]
    if not finetune_candidates or not mocle_candidates:
        return None
    if len(grasp_candidates) < grasp_samples_per_class:
        print(
            f"WARNING: {class_dir} only has {len(grasp_candidates)} GRASP "
            f"samples; expected {grasp_samples_per_class}"
        )
        return None
    return [
        select_candidates(finetune_candidates, 1, selection_order, rng)[0],
        select_candidates(mocle_candidates, 1, selection_order, rng)[0],
        *select_candidates(grasp_candidates, grasp_samples_per_class, selection_order, rng),
    ]


def build_figure(root: Path, classes, rows_per_class: int, out: Path,
                 figwidth_in: float, dpi: int, samples_per_class: int, layout: str,
                 selection_order: str, seed: int):
    # First gather all class panels. rows_per_class is retained for backward
    # compatibility with existing commands; each class now occupies one row.
    if rows_per_class != 1:
        print("WARNING: --rows-per-class is ignored by the class-panel layout")

    rng = random.Random(seed)
    layout_rows = []   # list of (class_name, [files])
    for cls in classes:
        cdir = root / cls
        if not cdir.exists():
            print(f"WARNING: {cdir} missing, skipping")
            continue
        if layout == "real-grasp":
            examples = choose_class_examples(cdir, samples_per_class, selection_order, rng)
            if examples is None:
                print(f"WARNING: {cdir} has no usable real + GRASP sample set, skipping")
                continue
            real_path, sample_paths = examples
            files = [real_path] + sample_paths
        elif layout == "diversity":
            files = choose_diversity_examples(cdir, samples_per_class, selection_order, rng)
            if files is None:
                print(f"WARNING: {cdir} has no usable FT + MoCLE + GRASP sample set, skipping")
                continue
        else:
            raise ValueError(f"Unknown layout: {layout}")
        layout_rows.append((cls, files))

    if not layout_rows:
        raise SystemExit("No images to plot.")

    n_rows_total = len(layout_rows)
    n_cols = len(layout_rows[0][1])
    if layout == "real-grasp":
        col_labels = ["Real"] + [f"GRASP {i}" for i in range(1, samples_per_class + 1)]
    else:
        col_labels = ["FT", "MoCLE"] + [f"GRASP {i}" for i in range(1, samples_per_class + 1)]

    # Each cell square; figure height proportional to row count.
    cell_in = figwidth_in / (n_cols + 0.6)  # leave room for left class label
    fig_w = cell_in * n_cols + 0.7  # +inch for left labels
    fig_h = cell_in * n_rows_total + 0.45  # +inch for top column headers

    fig, axes = plt.subplots(
        n_rows_total, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=False,
    )

    # Plot
    prev_class = None
    for ri, (cls, files) in enumerate(layout_rows):
        for ci, image_path in enumerate(files):
            ax = axes[ri][ci]
            img = mpimg.imread(image_path)
            ax.imshow(img, interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            # Light box around each cell
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#888")
            # Column header on first row
            if ri == 0:
                ax.set_title(col_labels[ci], fontsize=8, pad=3)
        # Class label on the left of the first cell of each new class
        if cls != prev_class:
            label = pretty_class_name(cls)
            axes[ri][0].set_ylabel(label, fontsize=8, rotation=90, labelpad=3)
            prev_class = cls

    plt.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.01,
                        wspace=0.04, hspace=0.04)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {out} ({n_rows_total} rows x {n_cols} cols, {fig_w:.1f}\"x{fig_h:.1f}\")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--rows-per-class", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS,
                    help="Number of GRASP samples to show.")
    ap.add_argument("--layout", choices=("real-grasp", "diversity"), default="real-grasp",
                    help="real-grasp: Real + GRASP samples. diversity: FT + MoCLE + GRASP samples.")
    ap.add_argument("--selection-order", choices=("first", "last", "random"), default="first",
                    help="Choose samples from first, last, or seeded-random matching source rows.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed used when --selection-order random is selected.")
    ap.add_argument("--figwidth-in", type=float, default=6.7,
                    help="NeurIPS text width is ~5.5in; use 6.7 for full-page figures.")
    ap.add_argument("--dpi", type=int, default=200,
                    help="Raster DPI for embedded images. 200 is typical for paper.")
    args = ap.parse_args()
    build_figure(args.root, args.classes, args.rows_per_class, args.out,
                 args.figwidth_in, args.dpi, args.samples_per_class, args.layout,
                 args.selection_order, args.seed)


if __name__ == "__main__":
    main()
