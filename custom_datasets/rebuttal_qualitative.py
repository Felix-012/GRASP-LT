#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative exporter (REAL + multiple SYNTH variants) for ImageNet-style datasets,
with optional BioViL-T phrase-grounding heatmap overlay.

Inputs:
  - base CSV with columns:
      - an ID column (e.g. image_id / filename / index) used only for naming outputs
      - a label column (single label string or int-as-string)
      - optional caption/text column (e.g. "caption") if you want --vlp_use_caption
  - real H5:  images stored either as:
        (A) raw encoded file bytes per row (vlen bytes / uint8 vector), OR
        (B) arrays (H,W) or (H,W,C)
  - synth H5(s): images stored as arrays (H,W) or (H,W,C), length >= base_len (optionally looped)

Core:
  - For each requested class, sample N rows from the base CSV.
  - Export REAL PNG and one SYNTH per synth source per row (single-pick mode).
  - Append-friendly: skip files that already exist.

Optional:
  - BioViL-T overlay (hi-ml-multimodal) for a query phrase (per label, optionally augmented with caption).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import torch

# Optional dependency (only required when --loc_overlay/--vlp_overlay is used)
try:
    from health_multimodal.image.utils import get_image_inference, ImageModelType
    from health_multimodal.text.utils import get_bert_inference, BertEncoderType
    from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine

    _HIML_OK = True
except Exception:
    _HIML_OK = False


# -----------------------------
# Utilities: naming / sanitation
# -----------------------------

def pick_synth_index_single(base_idx: int, base_len: int, synth_len: int, mode: str) -> int:
    """
    Returns a single synth index for a given base index.
    - mode='index':  synth_idx = base_idx (requires synth_len > base_idx)
    - mode='first':  same as 'index'
    - mode='latest': synth_idx = base_idx + k*base_len where k is the largest such that < synth_len
    """
    if synth_len <= base_idx:
        raise ValueError(f"synth_len={synth_len} too small for base_idx={base_idx}")

    if mode in ("index", "first"):
        return int(base_idx)

    if mode == "latest":
        k_max = (synth_len - 1 - base_idx) // base_len
        return int(base_idx + k_max * base_len)

    raise ValueError(f"Unknown synth_pick mode: {mode}")


def _slug(s: str, maxlen: int = 80) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s[:maxlen] if len(s) > maxlen else s


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Heatmap normalization + colormap overlay
# -----------------------------

def normalize_heatmap(
    hm: np.ndarray,
    *,
    p_low: float = 5.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """
    Robust normalization to [0,1] that:
      - ignores NaNs (hi-ml pads outside crop with NaN)
      - contrast-stretches using percentiles for visibility
    """
    hm = np.asarray(hm, dtype=np.float32)

    finite = hm[np.isfinite(hm)]
    if finite.size == 0:
        return np.zeros_like(hm, dtype=np.float32)

    lo = np.percentile(finite, p_low)
    hi = np.percentile(finite, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        lo = np.nanmin(hm)
        hi = np.nanmax(hm)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
            out = np.zeros_like(hm, dtype=np.float32)
            out[~np.isfinite(out)] = 0.0
            return out

    out = (hm - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out


def _red_rgb01(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0).astype(np.float32)
    z = np.zeros_like(u, dtype=np.float32)
    return np.stack([u, z, z], axis=-1)


def _jet_rgb01(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0).astype(np.float32)
    r = np.clip(1.5 - np.abs(4.0 * u - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * u - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * u - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _hot_rgb01(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0).astype(np.float32)
    r = np.clip(3.0 * u, 0.0, 1.0)
    g = np.clip(3.0 * u - 1.0, 0.0, 1.0)
    b = np.clip(3.0 * u - 2.0, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _gray_rgb01(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0).astype(np.float32)
    return np.stack([u, u, u], axis=-1)


def _apply_cmap_rgb01(u01: np.ndarray, cmap: str) -> np.ndarray:
    cmap = str(cmap).lower().strip()
    if cmap == "jet":
        return _jet_rgb01(u01)
    if cmap == "hot":
        return _hot_rgb01(u01)
    if cmap in ("gray", "grey"):
        return _gray_rgb01(u01)
    if cmap in ("red", "pure_red", "full_red"):
        return _red_rgb01(u01)
    return _jet_rgb01(u01)


def overlay_heatmap_cmap(
    pil_rgb: Image.Image,
    heat: np.ndarray,
    *,
    alpha: float = 0.45,
    cmap: str = "jet",
    p_low: float = 5.0,
    p_high: float = 99.0,
    threshold: float = 0.0,
) -> Image.Image:
    """
    Overlay heatmap using a colormap. NaN-safe.

    - heat normalized to [0,1] with percentile contrast stretch.
    - threshold (in [0,1]) zeros out low activations after normalization (optional).
    - alpha controls overall opacity; per-pixel alpha is scaled by heat intensity.
    """
    im = pil_rgb.convert("RGB")
    w, h = im.size

    hm = normalize_heatmap(heat, p_low=p_low, p_high=p_high)
    if threshold > 0:
        thr = float(threshold)
        hm = np.where(hm >= thr, hm, 0.0).astype(np.float32)

    if hm.shape != (h, w):
        hm_img = Image.fromarray((hm * 255).astype(np.uint8), mode="L").resize((w, h), resample=Image.BILINEAR)
        hm = (np.array(hm_img).astype(np.float32) / 255.0)

    rgb01 = _apply_cmap_rgb01(hm, cmap=cmap)
    rgb_u8 = (rgb01 * 255.0).astype(np.uint8)

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., 0:3] = rgb_u8
    overlay[..., 3] = (hm * 255.0 * float(alpha)).astype(np.uint8)

    overlay_img = Image.fromarray(overlay, mode="RGBA")
    out = Image.alpha_composite(im.convert("RGBA"), overlay_img)
    return out.convert("RGB")


# Backward-compatible wrapper
def overlay_heatmap_red(
    pil_rgb: Image.Image,
    heat: np.ndarray,
    *,
    alpha: float = 0.45,
    p_low: float = 5.0,
    p_high: float = 99.0,
) -> Image.Image:
    return overlay_heatmap_cmap(pil_rgb, heat, alpha=alpha, cmap="hot", p_low=p_low, p_high=p_high, threshold=0.0)


# -----------------------------
# Resize + square padding
# -----------------------------

@dataclass(frozen=True)
class ResizeMeta:
    scale: float


def resize_keep_aspect_with_scale(im: Image.Image, max_side: Optional[int]) -> Tuple[Image.Image, ResizeMeta]:
    if not max_side or max_side <= 0:
        return im, ResizeMeta(scale=1.0)
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im, ResizeMeta(scale=1.0)
    scale = max_side / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return im.resize((nw, nh), resample=Image.BILINEAR), ResizeMeta(scale=scale)


def make_square(im: Image.Image, *, fill=(0, 0, 0), out_size: Optional[int] = None) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    side = max(w, h)

    if w == h:
        sq = im
    else:
        pad_left = (side - w) // 2
        pad_top = (side - h) // 2
        sq = Image.new("RGB", (side, side), color=fill)
        sq.paste(im, (pad_left, pad_top))

    if out_size is not None and out_size > 0 and out_size != side:
        sq = sq.resize((out_size, out_size), resample=Image.BILINEAR)

    return sq


# -----------------------------
# H5 image decoding
# -----------------------------

def _decode_h5_image(obj) -> Image.Image:
    """
    Decode a single H5 item into a PIL image.

    Supports:
      - bytes / bytearray
      - np.ndarray uint8 vector (encoded bytes)
      - np.void (encoded bytes)
      - np.ndarray image arrays: (H,W), (H,W,1), (H,W,3), (H,W,4)
      - float arrays in [0,1] or [0,255]
    """
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj)).convert("RGB")

    if isinstance(obj, np.void):
        return Image.open(io.BytesIO(bytes(obj))).convert("RGB")

    if isinstance(obj, np.ndarray):
        if obj.dtype == np.uint8 and obj.ndim == 1:
            return Image.open(io.BytesIO(obj.tobytes())).convert("RGB")

        arr = obj
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]

        if arr.ndim not in (2, 3):
            raise ValueError(f"Unsupported image shape for export: {arr.shape}")

        if np.issubdtype(arr.dtype, np.floating):
            a = arr.astype(np.float32)
            if a.max() <= 1.5:
                a = a * 255.0
            a = np.clip(a, 0, 255).astype(np.uint8)
            arr = a
        elif arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            if a.max() <= 1.5:
                a = a * 255.0
            a = np.clip(a, 0, 255).astype(np.uint8)
            arr = a

        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return Image.fromarray(arr, mode="RGB")
        if arr.ndim == 3 and arr.shape[-1] == 4:
            return Image.fromarray(arr, mode="RGBA").convert("RGB")

    raise ValueError(f"Unsupported H5 item type: {type(obj)}")


def _read_h5_image(h5: h5py.File, idx: int, dataset_name: str = "images") -> Image.Image:
    ds = h5[dataset_name]
    obj = ds[idx]
    return _decode_h5_image(obj)


# -----------------------------
# BioViL-T localizer (phrase grounding similarity map)
# -----------------------------

class BioViLTLocalizer:
    """
    Uses hi-ml-multimodal ImageTextInferenceEngine to compute a similarity map (H x W) for (image, text).
    The engine expects image_path to a DICOM or JPEG. We write a temporary JPEG per image.
    """

    def __init__(self, *, device: str = "cuda", interpolation: str = "nearest"):
        if not _HIML_OK:
            raise RuntimeError(
                "hi-ml-multimodal is not available. Install/upgrade: pip install --upgrade hi-ml-multimodal"
            )

        self.device = device
        self.interpolation = interpolation

        # Paired encoders
        img_inf = get_image_inference(ImageModelType.BIOVIL_T)
        txt_inf = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
        self.engine = ImageTextInferenceEngine(img_inf, txt_inf)
        self.engine.to(device)

    def similarity_map_from_pil(self, pil_img: Image.Image, *, query_text: str, tmp_dir: Path) -> np.ndarray:
        tmp_path = tmp_dir / f"tmp_{os.getpid()}_{np.random.randint(0, 1_000_000)}.jpg"
        pil_img.convert("RGB").save(tmp_path, format="JPEG", quality=95)

        try:
            hm = self.engine.get_similarity_map_from_raw_data(
                image_path=tmp_path,
                query_text=str(query_text),
                interpolation=self.interpolation,
            )
            hm = np.asarray(hm)
            if hm.ndim != 2:
                hm = hm.squeeze()
            return hm
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def build_query_text(
    *,
    label: str,
    caption: Optional[str],
    prompt_template: str,
    use_caption: bool,
    caption_max_chars: int,
) -> str:
    base = prompt_template.format(label=label)
    if use_caption and caption:
        cap = str(caption).strip().replace("\n", " ")
        if caption_max_chars > 0 and len(cap) > caption_max_chars:
            cap = cap[:caption_max_chars]
        return f"{base}. {cap}"
    return base


# -----------------------------
# Selection logic
# -----------------------------

def _stable_u32(s: str) -> int:
    # deterministic across processes (unlike Python's built-in hash)
    h = hashlib.md5(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little", signed=False)

def select_indices_per_class(
    df: pd.DataFrame,
    classes: List[str],
    n_per_class: int,
    rng: np.random.Generator,
    *,
    label_col: str,
    split_col: Optional[str] = None,
    split_value: Optional[str] = None,
) -> Dict[str, List[int]]:
    # Keep original logic
    if split_col and split_value is not None:
        mask = df[split_col].astype(str).str.upper() == str(split_value).upper()
        df_eff = df.loc[mask].copy()
    else:
        df_eff = df

    # Extract the global seed from the generator (best effort).
    # If you prefer, pass args.seed explicitly instead of this.
    global_seed = 0
    try:
        # Works for default BitGenerators; otherwise just falls back to 0
        state = rng.bit_generator.state
        # Use something stable-ish from state to anchor the per-class seeds
        global_seed = int(state.get("state", 0)) & 0xFFFFFFFF
    except Exception:
        global_seed = 0

    selected: Dict[str, List[int]] = {}

    for cls in classes:
        cls_mask = df_eff[label_col].astype(str) == str(cls)
        idxs = df_eff.index[cls_mask].to_list()

        if len(idxs) == 0:
            selected[cls] = []
            continue

        if n_per_class <= 0 or n_per_class >= len(idxs):
            selected[cls] = idxs
            continue

        # Per-class RNG: independent of loop order / other classes
        seed_seq = np.random.SeedSequence([global_seed, _stable_u32(str(cls))])
        rng_cls = np.random.default_rng(seed_seq)

        pick = rng_cls.choice(idxs, size=n_per_class, replace=False).tolist()
        selected[cls] = pick

    return selected

# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True, help="Base CSV (must contain label_col and id_col).")
    ap.add_argument("--real_h5", required=True, help="Real H5 file.")
    ap.add_argument("--synth_h5", required=True, action="append",
                    help="Synthetic H5 file. Provide multiple times to add multiple synth sources.")
    ap.add_argument("--synth_tag", action="append", default=None,
                    help="Optional tag for each --synth_h5 (same order). If omitted, uses file stem.")

    ap.add_argument("--out_dir", required=True, help="Output folder for PNGs.")
    ap.add_argument("--classes", required=True, nargs="+", help="Comma-separated list of classes to export.")
    ap.add_argument("--n_per_class", type=int, default=8, help="Number of REAL examples per class.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--label_col", default="label", help="CSV column containing the class label.")
    ap.add_argument("--id_col", default="image_id", help="CSV column used for output naming (filename/id/etc.).")
    ap.add_argument("--img_dataset", default="images", help="Dataset name in both H5 files (default: images).")

    ap.add_argument("--skip_existing", action="store_true", help="Skip PNGs that already exist (append-friendly).")
    ap.add_argument("--max_side", type=int, default=1024, help="Downscale decoded images so max(H,W)<=max_side (0 disables).")
    ap.add_argument("--square_size", type=int, default=1024, help="Final square side length (0 keeps padded square size).")

    # Optional split restriction
    ap.add_argument("--split_col", default=None)
    ap.add_argument("--split_value", default=None)

    # BioViL overlay (optional)
    ap.add_argument("--loc_overlay", action="store_true", help="(Alias) Overlay BioViL-T similarity map.")
    ap.add_argument("--vlp_overlay", action="store_true", help="Overlay BioViL-T similarity map.")
    ap.add_argument("--vlp_alpha", type=float, default=0.35, help="Heatmap overlay alpha.")
    ap.add_argument("--vlp_p_low", type=float, default=5.0, help="Low percentile for heatmap contrast stretch.")
    ap.add_argument("--vlp_p_high", type=float, default=99.0, help="High percentile for heatmap contrast stretch.")
    ap.add_argument("--vlp_threshold", type=float, default=0.0, help="After normalization, zero-out heatmap values < threshold (0..1).")
    ap.add_argument("--vlp_cmap", choices=["jet", "hot", "gray", "red"], default="red",
                    help="Colormap for overlay (default: red).")

    ap.add_argument("--vlp_prompt_template", default="{label}", help="Prompt template. Use {label} placeholder.")
    ap.add_argument("--vlp_use_caption", action="store_true",
                    help="Append CSV caption/text to the query (if you have it).")
    ap.add_argument("--vlp_caption_col", default="caption", help="CSV column for optional caption/text.")
    ap.add_argument("--vlp_caption_max_chars", type=int, default=300)
    ap.add_argument("--vlp_interpolation", default="nearest",
                    help="Interpolation used by ImageTextInferenceEngine to upsample the heatmap.")
    ap.add_argument("--cpu_loc", action="store_true", help="Force localization to run on CPU.")

    ap.add_argument("--synth_pick", choices=["index", "latest", "first"], default="latest",
                    help="How to pick the synth counterpart if the synth H5 is looped/longer than base_len.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    _ensure_dir(out_root)

    classes = args.classes
    if not classes:
        raise ValueError("No classes provided via --classes")

    df = pd.read_csv(args.csv)
    for col in (args.label_col, args.id_col):
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'")

    rng = np.random.default_rng(args.seed)

    # Localizer (optional)
    do_loc = bool(args.loc_overlay or args.vlp_overlay)
    localizer: Optional[BioViLTLocalizer] = None
    if do_loc:
        device = "cpu" if args.cpu_loc or not torch.cuda.is_available() else "cuda"
        localizer = BioViLTLocalizer(device=device, interpolation=args.vlp_interpolation)
        print(f"[BioViL-T] Using device={device}, interpolation={args.vlp_interpolation}")

    # Open H5 files
    real_h5 = h5py.File(args.real_h5, "r")

    # Build synth sources: list of (tag, h5_file)
    synth_paths = [Path(p) for p in args.synth_h5]
    if args.synth_tag is not None and len(args.synth_tag) not in (0, len(synth_paths)):
        raise ValueError("--synth_tag must be omitted or provided once per --synth_h5 in the same order.")

    synth_tags: List[str] = []
    for i, sp in enumerate(synth_paths):
        if args.synth_tag is None or len(args.synth_tag) == 0:
            synth_tags.append(_slug(sp.stem))
        else:
            synth_tags.append(_slug(args.synth_tag[i]))

    synth_files: List[h5py.File] = []
    try:
        for sp in synth_paths:
            synth_files.append(h5py.File(str(sp), "r"))
    except Exception:
        for f in synth_files:
            try:
                f.close()
            except Exception:
                pass
        real_h5.close()
        raise

    try:
        base_len = len(df)

        selected = select_indices_per_class(
            df,
            classes,
            args.n_per_class,
            rng,
            label_col=args.label_col,
            split_col=args.split_col,
            split_value=args.split_value,
        )

        with tempfile.TemporaryDirectory(prefix="biovil_tmp_") as td:
            tmp_dir = Path(td)

            for cls in classes:
                cls_out = out_root / _slug(cls)
                _ensure_dir(cls_out)

                idxs = selected.get(cls, [])
                if not idxs:
                    print(f"[WARN] No eligible samples for class '{cls}' (after filters).")
                    continue

                for base_idx in idxs:
                    img_id = str(df.loc[base_idx, args.id_col])
                    base_name = f"{_slug(cls)}__{_slug(img_id)}__row{base_idx:06d}"

                    caption = None
                    if bool(args.vlp_use_caption) and str(args.vlp_caption_col) in df.columns:
                        caption = df.loc[base_idx, str(args.vlp_caption_col)]

                    # ----- REAL -----
                    real_decoded = _read_h5_image(real_h5, int(base_idx), dataset_name=args.img_dataset)
                    real_resized, _ = resize_keep_aspect_with_scale(real_decoded, args.max_side)
                    real_sq = make_square(real_resized, out_size=(args.square_size if args.square_size > 0 else None))
                    out_size = real_sq.size

                    real_out = real_sq

                    if localizer is not None:
                        try:
                            query_text = build_query_text(
                                label=cls,
                                caption=str(caption) if caption is not None else None,
                                prompt_template=args.vlp_prompt_template,
                                use_caption=bool(args.vlp_use_caption),
                                caption_max_chars=int(args.vlp_caption_max_chars),
                            )
                            hm = localizer.similarity_map_from_pil(real_sq, query_text=query_text, tmp_dir=tmp_dir)
                            real_out = overlay_heatmap_cmap(
                                real_out,
                                hm,
                                alpha=float(args.vlp_alpha),
                                cmap=str(args.vlp_cmap),
                                p_low=float(args.vlp_p_low),
                                p_high=float(args.vlp_p_high),
                                threshold=float(args.vlp_threshold),
                            )
                        except Exception as e:
                            print(f"[WARN] BioViL overlay failed for REAL {base_name}: {e}")

                    real_path = cls_out / f"{base_name}__real.png"
                    if not (args.skip_existing and real_path.exists()):
                        real_out.save(real_path)
                    else:
                        print(f"[SKIP] {real_path.name}")

                    # ----- SYNTH sources -----
                    for tag, syn_h5 in zip(synth_tags, synth_files):
                        syn_ds = syn_h5[args.img_dataset]
                        syn_len = int(syn_ds.shape[0])

                        synth_idx = pick_synth_index_single(int(base_idx), base_len, syn_len, args.synth_pick)

                        synth_decoded = _read_h5_image(syn_h5, int(synth_idx), dataset_name=args.img_dataset)
                        synth_resized, _ = resize_keep_aspect_with_scale(synth_decoded, args.max_side)
                        synth_sq = make_square(synth_resized, out_size=(args.square_size if args.square_size > 0 else None))

                        if synth_sq.size != out_size:
                            synth_sq = synth_sq.resize(out_size, resample=Image.BILINEAR)

                        synth_out = synth_sq

                        if localizer is not None:
                            try:
                                query_text = build_query_text(
                                    label=cls,
                                    caption=str(caption) if caption is not None else None,
                                    prompt_template=args.vlp_prompt_template,
                                    use_caption=bool(args.vlp_use_caption),
                                    caption_max_chars=int(args.vlp_caption_max_chars),
                                )
                                hm_s = localizer.similarity_map_from_pil(
                                    synth_sq,
                                    query_text=query_text,
                                    tmp_dir=tmp_dir,
                                )
                                synth_out = overlay_heatmap_cmap(
                                    synth_out,
                                    hm_s,
                                    alpha=float(args.vlp_alpha),
                                    cmap=str(args.vlp_cmap),
                                    p_low=float(args.vlp_p_low),
                                    p_high=float(args.vlp_p_high),
                                    threshold=float(args.vlp_threshold),
                                )
                            except Exception as e:
                                print(f"[WARN] BioViL overlay failed for SYNTH {base_name} tag={tag}: {e}")

                        synth_path = cls_out / f"{base_name}__synth_{tag}__idx{synth_idx:07d}.png"
                        if args.skip_existing and synth_path.exists():
                            print(f"[SKIP] {synth_path.name}")
                            continue
                        synth_out.save(synth_path)

        print("Done.")

    finally:
        try:
            real_h5.close()
        except Exception:
            pass
        for f in synth_files:
            try:
                f.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
