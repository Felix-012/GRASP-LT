#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze expert-level text similarity vs gradient similarity over training.

This script combines:
1) Static text similarity per expert from pooled text embeddings, and
2) Step-windowed gradient similarity per expert from projected_alignment CSV.

Outputs:
- text_similarity_per_expert.csv
- text_vs_gradient_per_expert_per_anchor.csv
- text_vs_gradient_correlations_per_anchor.csv
- correlation_over_anchors.(png|pdf)
- scatter_grid_over_anchors.(png|pdf)
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Matches the set-label mapping used in GCA helpers.
DEFAULT_LABEL_TO_EXPERT: Dict[str, int] = {
    "Lung Lesion": 0,
    "Pneumonia": 0,
    "Pneumothorax": 0,
    "Edema": 0,
    "Enlarged Cardiomediastinum": 0,
    "Support Devices": 0,
    "Fracture": 0,
    "Lung Opacity": 1,
    "Calcification of the Aorta": 1,
    "Consolidation": 1,
    "Tortuous Aorta": 1,
    "Pleural Other": 1,
    "Pneumoperitoneum": 1,
    "Subcutaneous Emphysema": 1,
    "Pneumomediastinum": 1,
    "Atelectasis": 2,
    "Cardiomegaly": 2,
    "Pleural Effusion": 2,
    "No Finding": 3,
}


def _norm_label(x: object) -> str:
    return " ".join(str(x).strip().split())


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
    if x.size < 2:
        return float("nan")
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson(xr, yr)


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if (not np.isfinite(n)) or n < eps:
        return np.zeros_like(v)
    return v / n


def _pooled_to_vec(ppe: np.ndarray, reduce_tokens: str) -> np.ndarray:
    if ppe.ndim == 1:
        return ppe
    if ppe.ndim == 2:
        if ppe.shape[0] == 1:
            return ppe[0]
        if reduce_tokens == "mean":
            return ppe.mean(axis=0)
        if reduce_tokens == "first":
            return ppe[0]
        if reduce_tokens == "last":
            return ppe[-1]
        raise ValueError(f"Unknown reduce_tokens={reduce_tokens}")
    raise ValueError(f"Unexpected pooled_prompt_embeds shape={ppe.shape}")


def _avg_pairwise_cos_from_unit_sum(sum_unit: np.ndarray, n: int) -> float:
    if n < 2:
        return float("nan")
    ss = float(np.dot(sum_unit, sum_unit))
    return (ss - n) / (n * (n - 1))


def _parse_int_lines(path: Path) -> np.ndarray:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    return np.asarray([int(x) for x in lines], dtype=np.int64)


def _parse_label(x: str, multi_label_mode: str, sep: str) -> str:
    raw = _norm_label(x)
    toks = [_norm_label(t) for t in raw.split(sep) if _norm_label(t)]
    if len(toks) <= 1:
        return toks[0] if toks else ""
    if multi_label_mode == "first":
        return toks[0]
    raise ValueError(
        f"Encountered multi-label entry {raw!r}. "
        "Use --multi-label-mode first to force first token."
    )


def _load_label_to_expert(
    mapping_csv: Optional[Path],
    mapping_label_col: str,
    mapping_expert_col: str,
) -> Dict[str, int]:
    if mapping_csv is None:
        return {_norm_label(k): int(v) for k, v in DEFAULT_LABEL_TO_EXPERT.items()}

    df = pd.read_csv(mapping_csv)
    if mapping_label_col not in df.columns or mapping_expert_col not in df.columns:
        raise KeyError(
            f"Mapping CSV must contain columns {mapping_label_col!r} and {mapping_expert_col!r}."
        )
    out: Dict[str, int] = {}
    for row in df.itertuples(index=False):
        lab = _norm_label(getattr(row, mapping_label_col))
        exp = int(getattr(row, mapping_expert_col))
        out[lab] = exp
    return out


def load_text_similarity_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"expert_id", "text_sim"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"text-sim CSV missing required columns: {sorted(missing)}")
    out = df.loc[:, ["expert_id", "text_sim"]].copy()
    out["expert_id"] = pd.to_numeric(out["expert_id"], errors="coerce")
    out["text_sim"] = pd.to_numeric(out["text_sim"], errors="coerce")
    out = out.dropna(subset=["expert_id"]).copy()
    out["expert_id"] = out["expert_id"].astype(int)
    return out.groupby("expert_id", as_index=False)["text_sim"].mean()


def _sorted_group_keys(hf: h5py.File, key_prefix: str) -> List[str]:
    keys = [k for k in hf.keys() if k.startswith(key_prefix)]
    if not keys:
        raise ValueError(f"No groups found with prefix {key_prefix!r}.")

    def _key_idx(k: str) -> int:
        rest = k[len(key_prefix):]
        return int(rest)

    return sorted(keys, key=_key_idx)


def _expert_ids_from_metadata(
    *,
    metadata_csv: Path,
    label_col: str,
    split_col: Optional[str],
    split_value: Optional[str],
    label_to_expert: Dict[str, int],
    multi_label_mode: str,
    multi_label_sep: str,
) -> np.ndarray:
    df = pd.read_csv(metadata_csv)
    if label_col not in df.columns:
        raise KeyError(f"metadata CSV missing label column {label_col!r}")

    if split_col and split_value is not None and split_col in df.columns:
        s = df[split_col].astype(str).str.strip().str.upper()
        df = df.loc[s == str(split_value).strip().upper()].copy()

    labels = [
        _parse_label(str(x), multi_label_mode=multi_label_mode, sep=multi_label_sep)
        for x in df[label_col].astype(str).tolist()
    ]
    unknown = sorted({lab for lab in labels if lab not in label_to_expert})
    if unknown:
        preview = unknown[:8]
        raise KeyError(
            f"Found labels not in label->expert mapping (showing up to 8): {preview}"
        )
    return np.asarray([label_to_expert[lab] for lab in labels], dtype=np.int64)


def compute_text_similarity_per_expert(
    *,
    emb_h5: Path,
    reduce_tokens: str,
    key_prefix: str,
    expert_ids: np.ndarray,
) -> pd.DataFrame:
    with h5py.File(emb_h5, "r") as hf:
        keys = _sorted_group_keys(hf, key_prefix=key_prefix)
        if len(keys) != int(expert_ids.shape[0]):
            raise ValueError(
                f"Length mismatch: embeddings groups={len(keys)} vs expert_ids={expert_ids.shape[0]}"
            )

        sum_unit: Dict[int, np.ndarray] = defaultdict(lambda: None)  # type: ignore[assignment]
        count: Dict[int, int] = defaultdict(int)
        n_zero: Dict[int, int] = defaultdict(int)

        for i, k in enumerate(keys):
            grp = hf[k]
            if "pooled_prompt_embeds" not in grp:
                raise KeyError(f"Group {k!r} missing dataset 'pooled_prompt_embeds'")
            ppe = grp["pooled_prompt_embeds"][...]
            vec = _pooled_to_vec(np.asarray(ppe, dtype=np.float64), reduce_tokens=reduce_tokens)
            u = _l2_normalize(vec)

            expert = int(expert_ids[i])
            if sum_unit[expert] is None:
                sum_unit[expert] = np.zeros_like(u, dtype=np.float64)
            if np.all(u == 0):
                n_zero[expert] += 1
            sum_unit[expert] += u
            count[expert] += 1

    rows: List[Dict[str, object]] = []
    for expert in sorted(count.keys()):
        n = int(count[expert])
        avg = _avg_pairwise_cos_from_unit_sum(sum_unit[expert], n)  # type: ignore[arg-type]
        rows.append(
            dict(
                expert_id=expert,
                n_text=n,
                n_pairs_text=(n * (n - 1)) // 2 if n >= 2 else 0,
                text_sim=float(avg),
                n_zero_norm=int(n_zero[expert]),
                zero_norm_frac=(float(n_zero[expert]) / float(n)) if n > 0 else float("nan"),
            )
        )
    return pd.DataFrame(rows)


def _anchors_from_checkpoint_dir(
    checkpoint_dir: Path,
    start: int,
    end: Optional[int],
) -> List[int]:
    steps: List[int] = []
    if not checkpoint_dir.exists():
        return steps
    for p in checkpoint_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except Exception:
            continue
        if step < start:
            continue
        if end is not None and step > end:
            continue
        steps.append(step)
    return sorted(set(steps))


def _anchors_from_stride(start: int, end: int, stride: int, include_end: bool) -> List[int]:
    if end < start:
        return []
    out = list(range(start, end + 1, stride))
    if include_end and (not out or out[-1] != end):
        out.append(end)
    return out


def _make_anchors(
    *,
    projected_df: pd.DataFrame,
    anchors_raw: Optional[str],
    per_step: bool,
    step_stride: int,
    checkpoint_dir: Optional[Path],
    anchor_start: int,
    anchor_end: Optional[int],
    anchor_stride: int,
    include_end: bool,
) -> List[int]:
    if anchors_raw:
        vals = [x.strip() for x in str(anchors_raw).split(",") if x.strip()]
        return sorted({int(x) for x in vals})

    if per_step:
        steps = sorted({int(x) for x in pd.to_numeric(projected_df["step"], errors="coerce").dropna().astype(int).tolist()})
        lo = int(anchor_start)
        hi = int(anchor_end) if anchor_end is not None else (steps[-1] if steps else lo)
        st = max(1, int(step_stride))
        return [s for s in steps if (s >= lo and s <= hi and ((s - lo) % st == 0))]

    if checkpoint_dir is not None:
        ckpt_anchors = _anchors_from_checkpoint_dir(checkpoint_dir, start=anchor_start, end=anchor_end)
        if ckpt_anchors:
            return ckpt_anchors

    max_step = int(pd.to_numeric(projected_df["step"], errors="coerce").dropna().max())
    end = anchor_end if anchor_end is not None else max_step
    return _anchors_from_stride(anchor_start, end, anchor_stride, include_end=include_end)


def aggregate_gradient_similarity(
    *,
    projected_csv: Path,
    grad_metric: str,
    anchors: Sequence[int],
    window_size: int,
) -> pd.DataFrame:
    df = pd.read_csv(projected_csv)
    required = {"step", "analyzed_expert", grad_metric}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"projected CSV missing required columns: {sorted(missing)}")

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["analyzed_expert"] = pd.to_numeric(df["analyzed_expert"], errors="coerce")
    df[grad_metric] = pd.to_numeric(df[grad_metric], errors="coerce")
    df = df.dropna(subset=["step", "analyzed_expert"])
    df["step"] = df["step"].astype(int)
    df["analyzed_expert"] = df["analyzed_expert"].astype(int)

    rows: List[Dict[str, object]] = []
    for anchor in anchors:
        lo = int(anchor) - int(window_size) + 1
        hi = int(anchor)
        sub = df.loc[(df["step"] >= lo) & (df["step"] <= hi)].copy()
        if sub.empty:
            continue
        agg = (
            sub.groupby("analyzed_expert", as_index=False)
            .agg(
                grad_sim=(grad_metric, "mean"),
                grad_sim_std=(grad_metric, "std"),
                n_grad_rows=(grad_metric, lambda s: int(np.isfinite(np.asarray(s, dtype=np.float64)).sum())),
                min_step=("step", "min"),
                max_step=("step", "max"),
            )
            .rename(columns={"analyzed_expert": "expert_id"})
        )
        agg["step_anchor"] = int(anchor)
        agg["window_start"] = int(lo)
        agg["window_end"] = int(hi)
        rows.append(agg)

    if not rows:
        return pd.DataFrame(
            columns=[
                "step_anchor",
                "window_start",
                "window_end",
                "expert_id",
                "grad_sim",
                "grad_sim_std",
                "n_grad_rows",
                "min_step",
                "max_step",
            ]
        )
    return pd.concat(rows, axis=0, ignore_index=True)


def build_anchor_correlations(
    merged: pd.DataFrame,
    min_experts: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for anchor, d in merged.groupby("step_anchor"):
        x = d["text_sim"].to_numpy(dtype=np.float64)
        y = d["grad_sim"].to_numpy(dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        n = int(x.size)
        rows.append(
            dict(
                step_anchor=int(anchor),
                n_experts=n,
                pearson=(float(_pearson(x, y)) if n >= min_experts else float("nan")),
                spearman=(float(_spearman(x, y)) if n >= min_experts else float("nan")),
                text_mean=(float(np.mean(x)) if n > 0 else float("nan")),
                grad_mean=(float(np.mean(y)) if n > 0 else float("nan")),
            )
        )
    return pd.DataFrame(rows).sort_values("step_anchor").reset_index(drop=True)


def plot_correlations(corr_df: pd.DataFrame, outdir: Path, prefix: str, plot_type: str) -> None:
    if corr_df.empty:
        return
    x = corr_df["step_anchor"].to_numpy(dtype=np.int64)
    yp = corr_df["pearson"].to_numpy(dtype=np.float64)
    ys = corr_df["spearman"].to_numpy(dtype=np.float64)

    plt.figure(figsize=(8.8, 4.5))
    if plot_type == "bar":
        width = max(1.0, 0.4 * float(np.min(np.diff(x))) if x.size >= 2 else 1.0)
        off = 0.28 * width
        if np.isfinite(yp).any():
            plt.bar(x - off, yp, width=0.55 * width, alpha=0.75, label="Pearson")
        if np.isfinite(ys).any():
            plt.bar(x + off, ys, width=0.55 * width, alpha=0.75, label="Spearman")
    else:
        if np.isfinite(yp).any():
            plt.plot(x, yp, marker="o", linewidth=1.8, label="Pearson")
        if np.isfinite(ys).any():
            plt.plot(x, ys, marker="o", linewidth=1.8, label="Spearman")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    plt.xlabel("Step anchor")
    plt.ylabel("Correlation across experts")
    plt.title("Text similarity vs gradient similarity over training")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_png = outdir / f"{prefix}_correlation_over_anchors.png"
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_png.with_suffix(".pdf"))
    plt.close()


def plot_scatter_grid(
    merged: pd.DataFrame,
    corr_df: pd.DataFrame,
    outdir: Path,
    prefix: str,
    scatter_steps_raw: Optional[str],
    scatter_max_panels: int,
) -> None:
    if merged.empty:
        return
    all_anchors = sorted(merged["step_anchor"].unique().tolist())
    if scatter_steps_raw:
        wanted = {int(x.strip()) for x in str(scatter_steps_raw).split(",") if x.strip()}
        anchors = [a for a in all_anchors if int(a) in wanted]
    else:
        anchors = all_anchors
    max_panels = max(1, int(scatter_max_panels))
    if len(anchors) > max_panels:
        idx = np.linspace(0, len(anchors) - 1, num=max_panels, dtype=int)
        anchors = [anchors[i] for i in idx]
    if not anchors:
        return

    n = len(anchors)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows), squeeze=False)
    ax_list = axes.ravel()

    corr_map = {
        int(r.step_anchor): (float(r.pearson), float(r.spearman), int(r.n_experts))
        for r in corr_df.itertuples(index=False)
    }
    for i, anchor in enumerate(anchors):
        ax = ax_list[i]
        d = merged.loc[merged["step_anchor"] == anchor]
        x = d["text_sim"].to_numpy(dtype=np.float64)
        y = d["grad_sim"].to_numpy(dtype=np.float64)
        eids = d["expert_id"].to_numpy(dtype=np.int64)
        m = np.isfinite(x) & np.isfinite(y)
        x, y, eids = x[m], y[m], eids[m]
        ax.scatter(x, y, s=40)
        for xx, yy, ee in zip(x, y, eids):
            ax.text(float(xx), float(yy), str(int(ee)), fontsize=8, alpha=0.9)
        pp, sp, nn = corr_map.get(int(anchor), (float("nan"), float("nan"), 0))
        ax.set_title(f"step {anchor} (n={nn}, r={pp:.2f}, rho={sp:.2f})")
        ax.set_xlabel("Text similarity")
        ax.set_ylabel("Gradient similarity")
        ax.grid(alpha=0.25)

    for j in range(len(anchors), len(ax_list)):
        ax_list[j].axis("off")

    fig.tight_layout()
    out_png = outdir / f"{prefix}_scatter_grid_over_anchors.png"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Analyze per-expert text similarity vs gradient similarity over step anchors."
    )
    ap.add_argument("--projected-csv", required=True, help="Projected alignment CSV from gc_train_mode.")
    ap.add_argument("--emb-h5", default=None, help="Embedding H5 with item_* groups and pooled_prompt_embeds.")

    ap.add_argument(
        "--expert-txt",
        default=None,
        help="Optional per-item expert id txt (one int per line, aligned to H5 item index).",
    )
    ap.add_argument("--metadata-csv", default=None, help="Metadata CSV (used if --expert-txt is omitted).")
    ap.add_argument("--label-col", default="Finding Labels")
    ap.add_argument("--split-col", default=None, help="Optional split column for metadata filter.")
    ap.add_argument("--split-value", default=None, help="Optional split value for metadata filter.")
    ap.add_argument("--multi-label-mode", default="error", choices=["error", "first"])
    ap.add_argument("--multi-label-sep", default="|")

    ap.add_argument("--mapping-csv", default=None, help="Optional label->expert mapping CSV.")
    ap.add_argument("--mapping-label-col", default="label")
    ap.add_argument("--mapping-expert-col", default="expert")

    ap.add_argument("--reduce-tokens", default="mean", choices=["mean", "first", "last"])
    ap.add_argument("--key-prefix", default="item_")

    ap.add_argument("--grad-metric", default="intra_cos")
    ap.add_argument("--window-size", type=int, default=100, help="Trailing window size for each anchor.")
    ap.add_argument("--min-experts", type=int, default=2, help="Min experts needed for correlation at an anchor.")
    ap.add_argument("--text-sim-csv", default=None, help="Optional precomputed text similarity CSV with expert_id,text_sim.")
    ap.add_argument("--per-step", action="store_true", help="Use all available steps (optionally strided) as anchors.")
    ap.add_argument("--step-stride", type=int, default=1, help="Used with --per-step: keep every Nth step.")
    ap.add_argument("--correlation-plot-type", default="bar", choices=["bar", "line"])
    ap.add_argument("--scatter-steps", default=None, help="Optional comma list of steps to show in scatter grid.")
    ap.add_argument("--scatter-max-panels", type=int, default=12, help="Max scatter subplots if many anchors.")

    ap.add_argument("--anchors", default=None, help="Optional comma-separated anchor steps, e.g. 100,200,300")
    ap.add_argument("--checkpoint-dir", default=None, help="Optional checkpoint directory to infer anchor steps.")
    ap.add_argument("--anchor-start", type=int, default=100)
    ap.add_argument("--anchor-end", type=int, default=None)
    ap.add_argument("--anchor-stride", type=int, default=100)
    ap.add_argument("--include-end", action="store_true")

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="text_vs_gradient")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.text_sim_csv:
        text_df = load_text_similarity_csv(Path(args.text_sim_csv))
        text_df["n_text"] = np.nan
        text_df["n_pairs_text"] = np.nan
    else:
        if not args.emb_h5:
            raise ValueError("--emb-h5 is required when --text-sim-csv is not provided.")
        label_to_expert = _load_label_to_expert(
            mapping_csv=Path(args.mapping_csv) if args.mapping_csv else None,
            mapping_label_col=args.mapping_label_col,
            mapping_expert_col=args.mapping_expert_col,
        )
        if args.expert_txt:
            expert_ids = _parse_int_lines(Path(args.expert_txt))
        else:
            if not args.metadata_csv:
                raise ValueError("Provide either --text-sim-csv or (emb/meta args).")
            expert_ids = _expert_ids_from_metadata(
                metadata_csv=Path(args.metadata_csv),
                label_col=args.label_col,
                split_col=args.split_col,
                split_value=args.split_value,
                label_to_expert=label_to_expert,
                multi_label_mode=args.multi_label_mode,
                multi_label_sep=args.multi_label_sep,
            )

        text_df = compute_text_similarity_per_expert(
            emb_h5=Path(str(args.emb_h5)),
            reduce_tokens=args.reduce_tokens,
            key_prefix=args.key_prefix,
            expert_ids=expert_ids,
        )
    text_csv = outdir / f"{args.prefix}_text_similarity_per_expert.csv"
    text_df.to_csv(text_csv, index=False)

    projected_df = pd.read_csv(args.projected_csv)
    anchors = _make_anchors(
        projected_df=projected_df,
        anchors_raw=args.anchors,
        per_step=bool(args.per_step),
        step_stride=int(args.step_stride),
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        anchor_start=int(args.anchor_start),
        anchor_end=(int(args.anchor_end) if args.anchor_end is not None else None),
        anchor_stride=int(args.anchor_stride),
        include_end=bool(args.include_end),
    )
    if not anchors:
        raise RuntimeError("No anchors resolved. Check --anchors / --checkpoint-dir / stride settings.")

    grad_df = aggregate_gradient_similarity(
        projected_csv=Path(args.projected_csv),
        grad_metric=args.grad_metric,
        anchors=anchors,
        window_size=int(args.window_size),
    )
    if grad_df.empty:
        raise RuntimeError("No gradient rows after anchor/window aggregation.")

    merged = grad_df.merge(
        text_df[["expert_id", "text_sim", "n_text", "n_pairs_text"]],
        on="expert_id",
        how="left",
    )
    merged_csv = outdir / f"{args.prefix}_text_vs_gradient_per_expert_per_anchor.csv"
    merged.to_csv(merged_csv, index=False)

    corr_df = build_anchor_correlations(merged, min_experts=int(args.min_experts))
    corr_csv = outdir / f"{args.prefix}_correlations_per_anchor.csv"
    corr_df.to_csv(corr_csv, index=False)

    plot_correlations(
        corr_df,
        outdir=outdir,
        prefix=args.prefix,
        plot_type=str(args.correlation_plot_type),
    )
    plot_scatter_grid(
        merged,
        corr_df,
        outdir=outdir,
        prefix=args.prefix,
        scatter_steps_raw=args.scatter_steps,
        scatter_max_panels=int(args.scatter_max_panels),
    )

    print(f"Wrote: {text_csv}")
    print(f"Wrote: {merged_csv}")
    print(f"Wrote: {corr_csv}")
    print(f"Wrote plots under: {outdir}")
    print(f"Anchors used: {','.join(str(x) for x in anchors)}")


if __name__ == "__main__":
    main()
