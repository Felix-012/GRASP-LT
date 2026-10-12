#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuttal qualitative exporter (REAL + multiple SYNTH variants), with optional MS-CXR mode.

Inputs:
  - base CSV with columns: dicom_id, Finding Labels (single label string), impression (optional)
  - real H5:  images stored either as:
        (A) raw encoded file bytes per row (vlen bytes / uint8 vector), OR
        (B) arrays (H,W) or (H,W,C)
  - synth H5: images stored as arrays (H,W) or (H,W,C), length >= base_len (looped dataset)

Core:
  - For each requested class, sample N rows from the base CSV.
  - Export REAL PNG and up to K SYNTH variants for each row, where synth_idx = base_idx + loop*base_len.
  - Append-friendly: skip files that already exist.

MS-CXR mode:
  - Only sample rows whose dicom_id is present in MS-CXR annotations AND has GT boxes for that class.
  - Draw GT boxes on REAL image (correctly mapped even after square padding).
  - NEW: bbox thresholding via --mscxr_min_box_side_px / --mscxr_min_box_area_px / --mscxr_min_box_area_frac

Localization overlay (BioViL-T phrase grounding):
  - Optional overlay of a patchwise text-image similarity heatmap for a query phrase.
  - Uses hi-ml-multimodal: health_multimodal.vlp.inference_engine.ImageTextInferenceEngine
  - NEW: colormap overlay via --vlp_cmap (default: jet)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


import torch

# Optional dependency (only required when --loc_overlay/--vlp_overlay is used)
try:
    from health_multimodal.image.utils import get_image_inference, ImageModelType
    from health_multimodal.text.utils import get_bert_inference, BertEncoderType
    from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine
    _HIML_OK = True
except Exception:
    _HIML_OK = False

def _pil_resample(name: str):
    """
    Map a string to a Pillow resampling enum in a version-safe way.
    """
    name = str(name).lower().strip()
    try:
        R = Image.Resampling  # Pillow>=9
    except Exception:
        R = Image  # older Pillow

    if name == "nearest":
        return R.NEAREST
    if name == "bilinear":
        return R.BILINEAR
    if name == "bicubic":
        return R.BICUBIC
    if name == "lanczos":
        return R.LANCZOS
    return R.BICUBIC



# -----------------------------
# Utilities: naming / sanitation
# -----------------------------

def pick_synth_index_single(base_idx: int, base_len: int, synth_len: int, mode: str) -> int:
    """
    Returns a single synth index for a given base index.
    - mode='index': base_idx (requires synth_len > base_idx)
    - mode='first': base_idx (same as index; kept for symmetry)
    - mode='latest': base_idx + k*base_len where k is the largest such that < synth_len
    """
    if synth_len <= base_idx:
        raise ValueError(f"synth_len={synth_len} too small for base_idx={base_idx}")

    if mode in ("index", "first"):
        return int(base_idx)

    if mode == "latest":
        # latest available occurrence of that base_idx in a partially looped file
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
        # fallback: simple nanmin/nanmax
        lo = np.nanmin(hm)
        hi = np.nanmax(hm)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
            out = np.zeros_like(hm, dtype=np.float32)
            out[~np.isfinite(out)] = 0.0
            return out

    out = (hm - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0  # NaNs -> transparent
    return out



def _red_rgb01(u: np.ndarray) -> np.ndarray:
    """
    Pure red colormap. Input u in [0,1], returns RGB in [0,1] with G=B=0.
    """
    u = np.clip(u, 0.0, 1.0).astype(np.float32)
    z = np.zeros_like(u, dtype=np.float32)
    return np.stack([u, z, z], axis=-1)

def _jet_rgb01(u: np.ndarray) -> np.ndarray:
    """
    Simple jet-like colormap approximation. Input u in [0,1], returns RGB in [0,1].
    """
    u = np.clip(u, 0.0, 1.0).astype(np.float32)
    r = np.clip(1.5 - np.abs(4.0 * u - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * u - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * u - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _hot_rgb01(u: np.ndarray) -> np.ndarray:
    """
    Very lightweight 'hot'-like colormap. Input u in [0,1], returns RGB in [0,1].
    """
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
    if cmap in ("jet",):
        return _jet_rgb01(u01)
    if cmap in ("hot",):
        return _hot_rgb01(u01)
    if cmap in ("gray", "grey"):
        return _gray_rgb01(u01)
    if cmap in ("red", "pure_red", "full_red"):
        return _red_rgb01(u01)
    # default fallback
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
    resize_resample: str = "bicubic",
    blur_radius: float = 1.5,
) -> Image.Image:
    """
    Overlay heatmap using a colormap (default: jet). NaN-safe.

    Smoothing knobs:
      - resize_resample: nearest|bilinear|bicubic|lanczos
      - blur_radius: Gaussian blur radius (0 disables)

    Pipeline:
      normalize -> resize to image size (high-quality) -> optional blur -> optional threshold -> colormap -> composite
    """
    im = pil_rgb.convert("RGB")
    w, h = im.size

    hm = normalize_heatmap(heat, p_low=p_low, p_high=p_high)

    # Always resize via Pillow so we can control resampling quality
    # Use float mode for resizing, then clip back to [0,1]
    hm_img_f = Image.fromarray(hm.astype(np.float32), mode="F").resize(
        (w, h), resample=_pil_resample(resize_resample)
    )
    hm = np.array(hm_img_f, dtype=np.float32)
    hm = np.clip(hm, 0.0, 1.0)

    # Optional Gaussian blur (helps a lot against blocky upsampling)
    br = float(blur_radius)
    if br > 0.0:
        hm_img_l = Image.fromarray((hm * 255.0).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=br)
        )
        hm = np.array(hm_img_l, dtype=np.float32) / 255.0

    # Optional threshold AFTER smoothing
    if threshold > 0:
        thr = float(threshold)
        hm = np.where(hm >= thr, hm, 0.0).astype(np.float32)

    rgb01 = _apply_cmap_rgb01(hm, cmap=cmap)
    rgb_u8 = (rgb01 * 255.0).astype(np.uint8)

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., 0:3] = rgb_u8
    overlay[..., 3] = (hm * 255.0 * float(alpha)).astype(np.uint8)

    overlay_img = Image.fromarray(overlay, mode="RGBA")
    out = Image.alpha_composite(im.convert("RGBA"), overlay_img)
    return out.convert("RGB")



# Backward-compatible wrapper (kept to avoid breaking older flags/usages)
def overlay_heatmap_red(
    pil_rgb: Image.Image,
    heat: np.ndarray,
    *,
    alpha: float = 0.45,
    p_low: float = 5.0,
    p_high: float = 99.0,
) -> Image.Image:
    return overlay_heatmap_cmap(
        pil_rgb, heat, alpha=alpha, cmap="hot", p_low=p_low, p_high=p_high, threshold=0.0
    )


# -----------------------------
# Resize + square padding (with metadata for bbox mapping)
# -----------------------------

@dataclass(frozen=True)
class ResizeMeta:
    scale: float  # uniform scale applied during keep-aspect resize


@dataclass(frozen=True)
class SquareMeta:
    side: int
    pad_left: int
    pad_top: int
    scale_to_out: float  # final resize factor from (side,side) -> (out_size,out_size)
    out_size: int


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


def make_square_with_meta(im: Image.Image, *, fill=(0, 0, 0), out_size: Optional[int] = None) -> Tuple[Image.Image, SquareMeta]:
    """
    Ensures H == W by padding (letterboxing) to a square and optionally resizing to out_size.
    Returns both the image and metadata needed to map coordinates through the transform.
    """
    im = im.convert("RGB")
    w, h = im.size
    side = max(w, h)

    pad_left = (side - w) // 2
    pad_top = (side - h) // 2

    if w == h:
        sq = im
    else:
        sq = Image.new("RGB", (side, side), color=fill)
        sq.paste(im, (pad_left, pad_top))

    if out_size is not None and out_size > 0 and out_size != side:
        scale_to_out = out_size / float(side)
        sq = sq.resize((out_size, out_size), resample=Image.BILINEAR)
        return sq, SquareMeta(side=side, pad_left=pad_left, pad_top=pad_top, scale_to_out=scale_to_out, out_size=out_size)

    return sq, SquareMeta(side=side, pad_left=pad_left, pad_top=pad_top, scale_to_out=1.0, out_size=side)


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
      - np.ndarray image arrays: (H,W), (H,W,1), (H,W,3)
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
# MS-CXR parsing (COCO JSON)
# -----------------------------

@dataclass(frozen=True)
class MSCXRImageInfo:
    image_id: int
    dicom_id: str
    width: int
    height: int


@dataclass(frozen=True)
class MSCXRAnn:
    dicom_id: str
    category: str
    bbox_xywh: Tuple[float, float, float, float]
    orig_w: int
    orig_h: int


def _infer_dicom_id_from_filename(file_name: str) -> str:
    stem = Path(str(file_name)).name
    stem = stem.split(".")[0]
    return stem


def load_mscxr_coco_json(path: str) -> Tuple[Dict[str, MSCXRImageInfo], Dict[Tuple[str, str], List[MSCXRAnn]], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name: Dict[int, str] = {}
    for c in coco.get("categories", []):
        cat_id_to_name[int(c["id"])] = str(c["name"])

    image_id_to_info: Dict[int, MSCXRImageInfo] = {}
    dicom_to_info: Dict[str, MSCXRImageInfo] = {}
    for im in coco.get("images", []):
        image_id = int(im["id"])
        file_name = str(im.get("file_name", ""))
        dicom_id = str(im.get("dicom_id") or _infer_dicom_id_from_filename(file_name))
        w = int(im.get("width", 0))
        h = int(im.get("height", 0))
        info = MSCXRImageInfo(image_id=image_id, dicom_id=dicom_id, width=w, height=h)
        image_id_to_info[image_id] = info
        dicom_to_info[dicom_id] = info

    boxes: Dict[Tuple[str, str], List[MSCXRAnn]] = {}
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        bbox = ann.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            continue
        info = image_id_to_info.get(image_id)
        if info is None:
            continue
        cat_name = str(cat_id_to_name.get(cat_id, f"cat_{cat_id}"))
        key = (info.dicom_id, cat_name)
        boxes.setdefault(key, []).append(
            MSCXRAnn(
                dicom_id=info.dicom_id,
                category=cat_name,
                bbox_xywh=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                orig_w=int(info.width),
                orig_h=int(info.height),
            )
        )

    categories = sorted(set(cat_id_to_name.values()))
    return dicom_to_info, boxes, categories


def _transform_mscxr_bbox_to_final_xyxy(
    ann: MSCXRAnn,
    *,
    decoded_size: Tuple[int, int],   # (W0,H0) of decoded image before any resizing
    resize_meta: ResizeMeta,
    square_meta: SquareMeta,
) -> Tuple[float, float, float, float]:
    """
    Map MS-CXR bbox (in ann.orig_w/ann.orig_h coords) to final output coords after:
      decoded -> keep-aspect resize -> square pad -> optional resize-to-out_size
    """
    w0, h0 = decoded_size
    ms_w = max(1, int(ann.orig_w))
    ms_h = max(1, int(ann.orig_h))

    x, y, bw, bh = ann.bbox_xywh

    # 1) MS-CXR coords -> decoded image coords (robust if sizes differ)
    sx0 = w0 / float(ms_w)
    sy0 = h0 / float(ms_h)
    x0 = x * sx0
    y0 = y * sy0
    bw0 = bw * sx0
    bh0 = bh * sy0

    # 2) decoded -> keep-aspect resized
    s1 = float(resize_meta.scale)
    x1 = x0 * s1
    y1 = y0 * s1
    bw1 = bw0 * s1
    bh1 = bh0 * s1

    # 3) square padding
    x2 = x1 + float(square_meta.pad_left)
    y2 = y1 + float(square_meta.pad_top)

    # 4) square -> out_size
    s2 = float(square_meta.scale_to_out)
    x3 = x2 * s2
    y3 = y2 * s2
    bw3 = bw1 * s2
    bh3 = bh1 * s2

    return (x3, y3, x3 + bw3, y3 + bh3)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _jet_rgb_tuple(u01: float) -> Tuple[int, int, int]:
    u = _clip01(u01)
    r = max(0.0, min(1.0, 1.5 - abs(4.0 * u - 3.0)))
    g = max(0.0, min(1.0, 1.5 - abs(4.0 * u - 2.0)))
    b = max(0.0, min(1.0, 1.5 - abs(4.0 * u - 1.0)))
    return (int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0)))


def filter_boxes_xyxy(
    boxes_xyxy: List[Tuple[float, float, float, float]],
    *,
    img_size: Tuple[int, int],
    min_side_px: float = 0.0,
    min_area_px: float = 0.0,
    min_area_frac: float = 0.0,
    clip_to_image: bool = True,
) -> List[Tuple[float, float, float, float]]:
    """
    Threshold / clean bboxes after transformation into final output coordinates.
    """
    W, H = int(img_size[0]), int(img_size[1])
    img_area = float(max(1, W * H))

    out: List[Tuple[float, float, float, float]] = []
    for (x1, y1, x2, y2) in boxes_xyxy:
        x1, x2 = (float(x1), float(x2))
        y1, y2 = (float(y1), float(y2))

        # normalize ordering
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        if clip_to_image:
            x1 = max(0.0, min(float(W - 1), x1))
            x2 = max(0.0, min(float(W - 1), x2))
            y1 = max(0.0, min(float(H - 1), y1))
            y2 = max(0.0, min(float(H - 1), y2))

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = bw * bh

        if bw <= 1.0 or bh <= 1.0:
            continue
        if min_side_px > 0 and (bw < float(min_side_px) or bh < float(min_side_px)):
            continue
        if min_area_px > 0 and area < float(min_area_px):
            continue
        if min_area_frac > 0 and area < float(min_area_frac) * img_area:
            continue

        out.append((x1, y1, x2, y2))

    return out


def draw_boxes_xyxy(
    pil_rgb: Image.Image,
    boxes_xyxy: List[Tuple[float, float, float, float]],
    *,
    color_mode: str = "jet",            # "jet" or "fixed"
    fixed_color: Tuple[int, int, int] = (0, 255, 0),
    color_by: str = "area",             # "area" or "index"
    width: int = 3,
) -> Image.Image:
    im = pil_rgb.convert("RGB")
    draw = ImageDraw.Draw(im)

    if not boxes_xyxy:
        return im

    areas = []
    for (x1, y1, x2, y2) in boxes_xyxy:
        areas.append(max(0.0, (x2 - x1) * (y2 - y1)))
    max_area = max(areas) if areas else 1.0
    n = len(boxes_xyxy)

    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        if str(color_mode).lower() == "fixed":
            col = fixed_color
        else:
            if str(color_by).lower() == "index":
                u = 1.0 if n <= 1 else float(i) / float(n - 1)
            else:
                u = float(areas[i]) / float(max_area + 1e-8)
            col = _jet_rgb_tuple(u)
        draw.rectangle([x1, y1, x2, y2], outline=col, width=int(width))

    return im


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

        # Pair BIOVIL_T image encoder with BIOVIL_T_BERT text encoder (the supported aligned pair)
        img_inf = get_image_inference(ImageModelType.BIOVIL_T)
        txt_inf = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
        self.engine = ImageTextInferenceEngine(img_inf, txt_inf)
        self.engine.to(device)

    def similarity_map_from_pil(self, pil_img: Image.Image, *, query_text: str, tmp_dir: Path) -> np.ndarray:
        tmp_path = tmp_dir / f"tmp_{os.getpid()}_{np.random.randint(0, 1_000_000)}.jpg"
        # Save grayscale JPEG (CXR-appropriate; engine expects DICOM/JPEG)
        pil_img.convert("L").save(tmp_path, format="JPEG", quality=95)

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
    impression: Optional[str],
    prompt_template: str,
    use_impression: bool,
    impression_max_chars: int,
) -> str:
    base = prompt_template.format(label=label)
    if use_impression and impression:
        imp = str(impression).strip().replace("\n", " ")
        if impression_max_chars > 0 and len(imp) > impression_max_chars:
            imp = imp[:impression_max_chars]
        return f"{base}. {imp}"
    return base


# -----------------------------
# Selection logic
# -----------------------------

def select_indices_per_class(
    df: pd.DataFrame,
    classes: List[str],
    n_per_class: int,
    rng: np.random.Generator,
    *,
    label_col: str,
    dicom_col: str,
    split_col: Optional[str] = None,
    split_value: Optional[str] = None,
    mscxr_boxes: Optional[Dict[Tuple[str, str], List[MSCXRAnn]]] = None,
    mscxr_cat_map: Optional[Dict[str, str]] = None,
) -> Dict[str, List[int]]:
    if split_col and split_value is not None:
        mask = df[split_col].astype(str).str.upper() == str(split_value).upper()
        df_eff = df.loc[mask].copy()
    else:
        df_eff = df

    selected: Dict[str, List[int]] = {}

    for cls in classes:
        cls_mask = df_eff[label_col].astype(str) == str(cls)
        idxs = df_eff.index[cls_mask].to_list()

        if mscxr_boxes is not None:
            want_cat = (mscxr_cat_map or {}).get(cls, cls)
            want_low = str(want_cat).lower()

            filtered = []
            for i in idxs:
                dicom_id = str(df.loc[i, dicom_col])
                ok = False
                for (d, cat), anns in mscxr_boxes.items():
                    if d == dicom_id and str(cat).lower() == want_low and len(anns) > 0:
                        ok = True
                        break
                if ok:
                    filtered.append(i)
            idxs = filtered

        if len(idxs) == 0:
            selected[cls] = []
            continue

        if n_per_class <= 0 or n_per_class >= len(idxs):
            pick = idxs
        else:
            pick = rng.choice(idxs, size=n_per_class, replace=False).tolist()
        selected[cls] = pick

    return selected


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Base CSV with dicom_id, Finding Labels, impression.")
    ap.add_argument("--real_h5", required=True, help="Real H5 file.")
    ap.add_argument("--out_dir", required=True, help="Output folder for PNGs.")
    ap.add_argument("--classes", required=True, help="Comma-separated list of classes to export.")
    ap.add_argument("--n_per_class", type=int, default=8, help="Number of REAL examples per class.")
    ap.add_argument("--max_synth_per_real", type=int, default=3, help="(Unused in single-pick mode) kept for compatibility.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label_col", default="Finding Labels")
    ap.add_argument("--dicom_col", default="dicom_id")
    ap.add_argument("--img_dataset", default="images", help="Dataset name in both H5 files (default: images).")

    ap.add_argument("--skip_existing", action="store_true", help="Skip PNGs that already exist (append-friendly).")
    ap.add_argument("--max_side", type=int, default=1024, help="Downscale decoded images so max(H,W)<=max_side (0 disables).")
    ap.add_argument("--square_size", type=int, default=1024, help="Final square side length (0 keeps padded square size).")

    # Optional split restriction
    ap.add_argument("--split_col", default=None)
    ap.add_argument("--split_value", default=None)

    # MS-CXR mode
    ap.add_argument("--mscxr_json", default=None, help="Path to MS-CXR COCO JSON (enables MS-CXR mode).")
    ap.add_argument("--mscxr_label_map", default=None,
                    help="Optional JSON file mapping your labels -> MS-CXR category names.")

    # NEW: bbox thresholding + coloring
    ap.add_argument("--mscxr_min_box_side_px", type=float, default=0.0,
                    help="Filter GT boxes: minimum box side length (in final output pixels). 0 disables.")
    ap.add_argument("--mscxr_min_box_area_px", type=float, default=0.0,
                    help="Filter GT boxes: minimum box area (in final output pixels). 0 disables.")
    ap.add_argument("--mscxr_min_box_area_frac", type=float, default=0.0,
                    help="Filter GT boxes: minimum box area as fraction of final image area. 0 disables.")
    ap.add_argument("--mscxr_clip_boxes", action="store_true",
                    help="Clip transformed GT boxes to final image bounds before filtering/drawing.")
    ap.add_argument("--mscxr_box_width", type=int, default=3, help="GT box stroke width.")
    ap.add_argument("--mscxr_box_color_mode", choices=["fixed", "jet"], default="fixed",
                    help="How to color GT boxes. 'fixed' uses a single RGB (default: intense green).")

    ap.add_argument("--mscxr_box_color_by", choices=["area", "index"], default="area",
                    help="If --mscxr_box_color_mode=jet, choose mapping input: 'area' or 'index'.")

    # Localization overlay (BioViL-T similarity map)
    ap.add_argument("--loc_overlay", action="store_true", help="(Alias) Overlay BioViL-T similarity map.")
    ap.add_argument("--vlp_overlay", action="store_true", help="Overlay BioViL-T similarity map (phrase grounding).")
    ap.add_argument("--vlp_alpha", type=float, default=0.35, help="Heatmap overlay alpha.")
    ap.add_argument("--vlp_p_low", type=float, default=5.0, help="Low percentile for heatmap contrast stretch.")
    ap.add_argument("--vlp_p_high", type=float, default=99.0, help="High percentile for heatmap contrast stretch.")
    ap.add_argument("--vlp_threshold", type=float, default=0.0,
                    help="After normalization, zero-out heatmap values < threshold (0..1).")
    # in parse_args()
    ap.add_argument(
        "--vlp_cmap",
        choices=["jet", "hot", "gray", "red"],
        default="red",
        help="Colormap for overlay (default: jet).",
    )

    ap.add_argument("--vlp_prompt_template", default="{label}", help="Prompt template. Use {label} placeholder.")
    ap.add_argument("--vlp_use_impression", action="store_true",
                    help="Append the CSV impression text to the query (helps align with your text-conditioning).")
    ap.add_argument("--vlp_impression_max_chars", type=int, default=300)
    ap.add_argument("--vlp_interpolation", default="nearest",
                    help="Interpolation used by ImageTextInferenceEngine to upsample the heatmap.")
    ap.add_argument("--cpu_loc", action="store_true", help="Force localization to run on CPU.")

    ap.add_argument(
        "--synth_h5",
        required=True,
        action="append",
        help="Synthetic H5 file. Can be provided multiple times to append multiple synth sources.",
    )
    ap.add_argument(
        "--synth_tag",
        action="append",
        default=None,
        help="Optional tag for each --synth_h5 (same order). If omitted, uses file stem.",
    )

    ap.add_argument(
        "--synth_pick",
        choices=["index", "latest", "first"],
        default="index",
        help="How to pick the synth counterpart within a synth H5 if it has more than base_len samples. "
             "'index' uses synth_idx=base_idx. 'latest' uses the largest available loop for that base_idx.",
    )

    ap.add_argument(
        "--vlp_resize_resample",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="bicubic",
        help="Resampling used when resizing the similarity map to the output image size.",
    )
    ap.add_argument(
        "--vlp_blur_radius",
        type=float,
        default=1.5,
        help="Gaussian blur radius applied to the resized similarity map (0 disables).",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    _ensure_dir(out_root)

    classes = [c.strip() for c in str(args.classes).split(",") if c.strip()]
    if not classes:
        raise ValueError("No classes provided via --classes")

    df = pd.read_csv(args.csv)
    for col in (args.label_col, args.dicom_col):
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'")

    rng = np.random.default_rng(args.seed)

    # MS-CXR (optional)
    mscxr_boxes = None
    mscxr_categories = None
    mscxr_cat_map = None

    if args.mscxr_json:
        _, mscxr_boxes, mscxr_categories = load_mscxr_coco_json(args.mscxr_json)

        if args.mscxr_label_map:
            with open(args.mscxr_label_map, "r", encoding="utf-8") as f:
                mscxr_cat_map = json.load(f)
        else:
            mscxr_cat_map = {}

        print(f"[MS-CXR] Loaded categories: {mscxr_categories}")
        print(f"[MS-CXR] Box-keys: {len(mscxr_boxes)}")

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
            dicom_col=args.dicom_col,
            split_col=args.split_col,
            split_value=args.split_value,
            mscxr_boxes=mscxr_boxes,
            mscxr_cat_map=mscxr_cat_map,
        )

        # Temp dir for JPEGs needed by hi-ml-multimodal engine
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
                    dicom_id = str(df.loc[base_idx, args.dicom_col])
                    base_name = f"{_slug(cls)}__{dicom_id}__row{base_idx:06d}"

                    impression = None
                    if "impression" in df.columns:
                        impression = df.loc[base_idx, "impression"]

                    # ----- REAL -----
                    real_decoded = _read_h5_image(real_h5, int(base_idx), dataset_name=args.img_dataset)
                    decoded_size = real_decoded.size  # (W0,H0)

                    real_resized, rmeta = resize_keep_aspect_with_scale(real_decoded, args.max_side)
                    real_sq, smeta = make_square_with_meta(
                        real_resized,
                        out_size=(args.square_size if args.square_size > 0 else None),
                    )
                    out_size = real_sq.size  # (W,H) final

                    # Start with the final processed image (before any drawings)
                    real_out = real_sq

                    # BioViL overlay on REAL (before drawing GT boxes)
                    if localizer is not None:
                        try:
                            query_text = build_query_text(
                                label=cls,
                                impression=str(impression) if impression is not None else None,
                                prompt_template=args.vlp_prompt_template,
                                use_impression=bool(args.vlp_use_impression),
                                impression_max_chars=int(args.vlp_impression_max_chars),
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
                                resize_resample=str(args.vlp_resize_resample),
                                blur_radius=float(args.vlp_blur_radius),
                            )

                        except Exception as e:
                            print(f"[WARN] BioViL overlay failed for REAL {base_name}: {e}")

                    # GT boxes (MS-CXR) on REAL (after overlay so boxes are clearly visible)
                    if mscxr_boxes is not None:
                        anns_for_real: List[MSCXRAnn] = []
                        want_cat = (mscxr_cat_map or {}).get(cls, cls)
                        want_low = str(want_cat).lower()
                        for (d, cat), anns in mscxr_boxes.items():
                            if d == dicom_id and str(cat).lower() == want_low:
                                anns_for_real.extend(anns)

                        if anns_for_real:
                            boxes_xyxy = [
                                _transform_mscxr_bbox_to_final_xyxy(
                                    a,
                                    decoded_size=decoded_size,
                                    resize_meta=rmeta,
                                    square_meta=smeta,
                                )
                                for a in anns_for_real
                            ]

                            # NEW: threshold/clip boxes in final pixel coords
                            boxes_xyxy = filter_boxes_xyxy(
                                boxes_xyxy,
                                img_size=real_out.size,
                                min_side_px=float(args.mscxr_min_box_side_px),
                                min_area_px=float(args.mscxr_min_box_area_px),
                                min_area_frac=float(args.mscxr_min_box_area_frac),
                                clip_to_image=bool(args.mscxr_clip_boxes),
                            )

                            if boxes_xyxy:
                                real_out = draw_boxes_xyxy(
                                    real_out,
                                    boxes_xyxy,
                                    color_mode=str(args.mscxr_box_color_mode),
                                    fixed_color=(0, 255, 0),
                                    color_by=str(args.mscxr_box_color_by),
                                    width=int(args.mscxr_box_width),
                                )

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
                        synth_sq, _ = make_square_with_meta(
                            synth_resized,
                            out_size=(args.square_size if args.square_size > 0 else None),
                        )

                        if synth_sq.size != out_size:
                            synth_sq = synth_sq.resize(out_size, resample=Image.BILINEAR)

                        synth_out = synth_sq

                        # BioViL overlay (if enabled)
                        if localizer is not None:
                            try:
                                query_text = build_query_text(
                                    label=cls,
                                    impression=str(impression) if impression is not None else None,
                                    prompt_template=args.vlp_prompt_template,
                                    use_impression=bool(args.vlp_use_impression),
                                    impression_max_chars=int(args.vlp_impression_max_chars),
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
                                    resize_resample=str(args.vlp_resize_resample),
                                    blur_radius=float(args.vlp_blur_radius),
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
