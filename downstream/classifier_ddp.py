#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-node / Multi-GPU classifier training + evaluation using Hugging Face Accelerate (DDP).

- Works on a single GPU, single node, or many nodes — same script.
- Uses `accelerate.prepare(...)` to wrap model, loaders, optimizer, scheduler.
- Distributed-safe saving (main process only) and barriers.
- Per-process seeding for reproducibility across nodes.
- Keeps your H5/CSV/Directory datasets + metrics/exports intact.

New:
- Support HDF5-based evaluation sets via --eval_h5 to avoid inode limits.
  H5 must contain:
    • images  : vlen bytes (PNG/JPEG) or uint8 arrays [N]
    • targets : int64 [N]
    • classes : vlen utf-8 strings (HDF5 dataset or attribute)

Resume/Skip:
- By default, the script SKIPS (run, seed) that appear completed, based on summary.csv
  and/or presence of outputs inside seed directories. Use --overwrite_existing to force recompute.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import time
from pathlib import Path
import random
from collections import Counter


def configure_runtime_cache() -> None:
    """Keep model caches off $HOME on compute nodes where home is not mounted."""
    workspace = os.environ.get("WORKSPACE")
    cache_root = os.environ.get("GRASP_CACHE_DIR")
    if not cache_root:
        cache_root = os.path.join(workspace, ".cache") if workspace else os.path.join(os.getcwd(), ".cache")

    os.environ.setdefault("XDG_CACHE_HOME", cache_root)
    os.environ.setdefault("TORCH_HOME", os.path.join(cache_root, "torch"))
    os.environ.setdefault("HF_HOME", os.path.join(cache_root, "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))

    for key in (
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
    ):
        os.makedirs(os.environ[key], exist_ok=True)


configure_runtime_cache()

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

try:
    import h5py
except Exception:
    h5py = None


def _open_h5_readonly(path: str):
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def _norm_label(x) -> str:
    # bytes -> utf-8; strings -> strip; numbers -> str then strip
    if isinstance(x, (bytes, np.bytes_)):
        x = x.decode("utf-8", errors="replace")
    return str(x).strip()

# ----------------------- Optional torch.compile ----------------------- #

def maybe_compile(model: nn.Module, mode: str | None):
    if not mode:
        return model
    try:
        return torch.compile(model, mode=mode)
    except Exception:
        # Fallback silently if compile not available on system
        return model

# ---------- Model factory ----------

def infer_model(name: str, num_classes: int):
    name = name.lower()
    try_imagenet_weights = True
    if name in {"densenet121", "densenet"}:
        if try_imagenet_weights:
            try:
                m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            except Exception:
                m = models.densenet121(pretrained=True)
        else:
            m = models.densenet121()
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m

    if name in {"resnet50", "resnet"}:
        if try_imagenet_weights:
            try:
                m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            except Exception:
                m = models.resnet50(pretrained=True)
        else:
            m = models.resnet50()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name in {"vit_b_16", "vit"}:
        if try_imagenet_weights:
            try:
                m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            except Exception:
                m = models.vit_b_16(pretrained=True)
        else:
            m = models.vit_b_16()
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m

    raise ValueError(f"Unknown model: {name}")

# ---------- Datasets ----------

class H5ImageDataset(Dataset):
    """
    TRAIN HDF5 dataset that returns (tensor_image, target_idx).
    Images come from H5 (dataset=`images_key`) and labels from an external CSV (index-aligned).
    """
    def __init__(self, path, targets: np.ndarray, classes: list, transform=None, images_key="images"):
        if h5py is None:
            raise RuntimeError("h5py is required for HDF5 datasets. Install with `pip install h5py`.")
        self.path = str(path)
        self.images_key = images_key
        self.transform = transform

        self._targets = np.asarray(targets, dtype=np.int64)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        with _open_h5_readonly(self.path) as f:
            if self.images_key not in f:
                raise ValueError(f"Missing dataset '{self.images_key}' in {self.path}")
            imgs = f[self.images_key]
            self._len_images = int(imgs.shape[0]) if getattr(imgs, "shape", None) else len(imgs)

        if self._len_images < len(self._targets):
            # Reduced budgets generate fewer images than there are prompts: image p was generated from CSV row p,
            # so the first n rows are the matching labels. (Paper runs: 50,000 images >= 23,550 rows, unaffected.)
            print(f"[warn] {self.path}: {self._len_images} images < {len(self._targets)} CSV rows; "
                  f"using the first {self._len_images} rows")
            self._targets = self._targets[: self._len_images]

        self._effective_len = int(len(self._targets))
        self._h5 = None  # lazily opened per worker / per process

    def __len__(self):
        return self._effective_len

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = _open_h5_readonly(self.path)

    @staticmethod
    def _decode_image(arr):
        # Encoded bytes (vlen) or raw arrays -> PIL.Image
        if isinstance(arr, (bytes, bytearray, np.bytes_)):
            bio = io.BytesIO(arr)
            return Image.open(bio)
        if isinstance(arr, np.ndarray) and arr.dtype == np.uint8:
            if arr.ndim == 1:
                bio = io.BytesIO(arr.tobytes())
                return Image.open(bio)
            if arr.ndim == 2:
                return Image.fromarray(arr, mode="L")
            if arr.ndim == 3:
                if arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[2]:
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[2] == 1:
                    return Image.fromarray(arr[..., 0], mode="L")
                return Image.fromarray(arr, mode="RGB")
        return Image.fromarray(np.asarray(arr))

    def __getitem__(self, idx):
        self._ensure_open()
        imgs = self._h5[self.images_key]
        target = int(self._targets[idx])

        arr = imgs[idx]
        img = self._decode_image(arr)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class H5EvalDataset(Dataset):
    """
    EVAL HDF5 dataset that returns (tensor_image, target_idx).
    Requires datasets:
      - images: vlen bytes (PNG/JPEG) or uint8 arrays, shape [N]
      - targets: int64 [N]
    Also expects 'classes' (dataset or attribute): list[str] length C.
    Exposes `.classes` and `._targets` (for RemapTargets compatibility).
    """
    def __init__(self, h5_path: str, transform=None,
                 images_key: str = "images", targets_key: str = "targets"):
        if h5py is None:
            raise RuntimeError("h5py is required for HDF5 datasets. Install with `pip install h5py`.")
        self.path = str(h5_path)
        self.transform = transform
        self.images_key = images_key
        self.targets_key = targets_key

        # Keep one shared handle per Dataset instance; worker procs get their copy via fork/spawn
        self._h5 = _open_h5_readonly(self.path)
        for k in (self.images_key, self.targets_key):
            if k not in self._h5:
                raise ValueError(f"Missing dataset '{k}' in {self.path}")

        # classes (prefer dataset, fallback to attribute)
        def _read_classes_from_h5(h5):
            if "classes" in h5:
                raw = h5["classes"][()]
            elif "classes" in h5.attrs:
                raw = h5.attrs["classes"]
            else:
                return None
            arr = np.atleast_1d(raw)
            out = []
            for x in arr:
                if isinstance(x, (bytes, np.bytes_)):
                    out.append(_norm_label(x))  # decode + strip
                else:
                    out.append(_norm_label(x))
            return out

        classes = _read_classes_from_h5(self._h5)
        if classes is None:
            t = np.asarray(self._h5[self.targets_key][()], dtype=np.int64)
            C = int(t.max()) + 1 if t.size else 0
            classes = [str(i) for i in range(C)]
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # expose targets as np array & as _targets for RemapTargets wrapper
        self._targets = np.asarray(self._h5[self.targets_key], dtype=np.int64)

    def __len__(self):
        return int(self._h5[self.targets_key].shape[0])

    @staticmethod
    def _decode_image(arr):
        if isinstance(arr, (bytes, bytearray, np.bytes_)):
            bio = io.BytesIO(arr)
            return Image.open(bio)
        if isinstance(arr, np.ndarray) and arr.dtype == np.uint8:
            if arr.ndim == 1:
                bio = io.BytesIO(arr.tobytes())
                return Image.open(bio)
            if arr.ndim == 2:
                return Image.fromarray(arr, mode="L")
            if arr.ndim == 3:
                if arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[2]:
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[2] == 1:
                    return Image.fromarray(arr[..., 0], mode="L")
                return Image.fromarray(arr, mode="RGB")
        return Image.fromarray(np.asarray(arr))

    def __getitem__(self, idx):
        arr = self._h5[self.images_key][idx]
        img = self._decode_image(arr)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self._targets[idx])


def _resolve_path(p: str, root_prefix: str | None, csv_dir: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    if root_prefix and Path(root_prefix).is_dir():
        return Path(root_prefix) / pp
    return csv_dir / pp


class CSVImageDataset(Dataset):
    """
    Dataset reading image FILE PATHS from a CSV (used for EVAL legacy mode).
    """
    def __init__(self, csv_path: str, label_col: str = "Finding Labels", path_col: str = "path",
                 transform=None, root_prefix: str | None = None):
        self.csv_path = str(csv_path)
        self.transform = transform
        self.label_col = label_col
        self.path_col = path_col
        self.root_prefix = root_prefix

        df = pd.read_csv(self.csv_path)
        if self.label_col not in df.columns:
            raise ValueError(f"CSV {self.csv_path} missing '{self.label_col}' column.")
        if self.path_col not in df.columns:
            raise ValueError(f"CSV {self.csv_path} missing '{self.path_col}' column.")

        labels_raw = [_norm_label(s) for s in df[self.label_col].astype(str).tolist()]
        paths_raw = df[self.path_col].astype(str).tolist()

        classes = []
        class_to_idx = {}
        targets = np.empty(len(labels_raw), dtype=np.int64)
        for i, name in enumerate(labels_raw):
            if name not in class_to_idx:
                class_to_idx[name] = len(classes)
                classes.append(name)
            targets[i] = class_to_idx[name]

        csv_dir = Path(self.csv_path).parent
        self.paths = [str(_resolve_path(p, self.root_prefix, csv_dir)) for p in paths_raw]
        self._targets = targets
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, idx):
        path = self.paths[idx]
        target = int(self._targets[idx])
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class DirectoryImagesDataset(Dataset):
    """
    TRAIN dataset for directories where labels come from a CSV (index-aligned).
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(self, root: str, csv_path: str, label_col: str = "Finding Labels", transform=None):
        self.root = str(root)
        self.transform = transform

        all_files = [p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in self.IMG_EXTS]
        if not all_files:
            raise ValueError(f"No image files found under directory: {root}")
        rel_sorted = sorted((str(p) for p in all_files), key=lambda s: os.path.relpath(s, self.root))

        df = pd.read_csv(csv_path)
        if label_col not in df.columns:
            raise ValueError(f"CSV {csv_path} missing '{label_col}' column.")
        labels_raw = [_norm_label(s) for s in df[label_col].astype(str).tolist()]
        n_csv = len(labels_raw)

        if len(rel_sorted) < n_csv:
            raise ValueError(
                f"Directory images ({len(rel_sorted)}) < CSV rows ({n_csv}) for root {root}. "
                f"Please fix or reduce the CSV."
            )

        self.paths = rel_sorted[:n_csv]

        classes, class_to_idx, targets = [], {}, np.empty(n_csv, dtype=np.int64)
        for i, name in enumerate(labels_raw):
            if name not in class_to_idx:
                class_to_idx[name] = len(classes)
                classes.append(name)
            targets[i] = class_to_idx[name]

        self._targets = targets
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, idx):
        path = self.paths[idx]
        target = int(self._targets[idx])
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class RemapTargets(Dataset):
    """
    Wrap a dataset (H5/CSV/DirectoryImages) to map integer targets to the shared eval mapping via class name.
    If the base dataset contains classes absent in eval, those samples are DROPPED (not fatal).
    """
    def __init__(self, base: Dataset, shared_class_to_idx: dict, is_main_process: bool = True, announce: bool = True):
        self.base = base
        self.shared_class_to_idx = dict(shared_class_to_idx)
        self.classes = list(shared_class_to_idx.keys())
        self.class_to_idx = dict(shared_class_to_idx)

        base_classes = getattr(self.base, "classes", None)
        base_targets = getattr(self.base, "_targets", None)
        if base_classes is None or base_targets is None:
            raise ValueError("Base dataset must expose `.classes` and internal `_targets`.")

        base_targets = np.asarray(base_targets, dtype=np.int64)

        # Keep only indices whose class exists in eval mapping
        keep_indices = []
        dropped_labels = []
        for i, t in enumerate(base_targets):
            cname = base_classes[int(t)]
            if cname in self.shared_class_to_idx:
                keep_indices.append(i)
            else:
                dropped_labels.append(cname)

        if len(keep_indices) == 0:
            raise ValueError("After dropping labels absent from eval, the train dataset is empty.")

        self._keep_indices = np.asarray(keep_indices, dtype=np.int64)
        self._dropped_count = len(dropped_labels)

        if announce and is_main_process and self._dropped_count > 0:
            cnt = Counter(dropped_labels)
            breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(cnt.items()))
            print(f"[INFO] Dropped {self._dropped_count} train samples whose labels are not in eval. "
                  f"Per-label: {breakdown}")

    def __len__(self):
        return int(self._keep_indices.size)

    def __getitem__(self, idx):
        base_idx = int(self._keep_indices[idx])
        img, base_t = self.base[base_idx]
        class_name = self.base.classes[int(base_t)]
        return img, self.shared_class_to_idx[class_name]

# ---------- CSV helpers ----------

def labels_from_csv(csv_path: str, label_col: str):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"CSV {csv_path} is missing column '{label_col}'")
    labels_raw = [ _norm_label(s) for s in df[label_col].astype(str).tolist() ]
    classes, class_to_idx, targets = [], {}, np.empty(len(labels_raw), dtype=np.int64)
    for i, name in enumerate(labels_raw):
        if name not in class_to_idx:
            class_to_idx[name] = len(classes)
            classes.append(name)
        targets[i] = class_to_idx[name]
    return targets, classes

# ---------- Transforms ----------

def build_transforms(image_size: int, use_imagenet_norm: bool, hflip: bool):
    normalize_mean = [0.485, 0.456, 0.406] if use_imagenet_norm else [0.5, 0.5, 0.5]
    normalize_std  = [0.229, 0.224, 0.225] if use_imagenet_norm else [0.5, 0.5, 0.5]

    to_rgb = transforms.Lambda(lambda im: im.convert("RGB"))
    train_tfms = transforms.Compose([
        to_rgb,
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip() if hflip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])
    eval_tfms = transforms.Compose([
        to_rgb,
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])
    return train_tfms, eval_tfms

# ---------- Metrics ----------

@torch.no_grad()
def evaluate(accelerator: Accelerator, model, dataloader, num_classes: int, topk: int = 5):
    model.eval()

    device = accelerator.device
    C = num_classes
    local_conf = torch.zeros((C, C), dtype=torch.long, device=device)  # rows: true, cols: pred

    total = torch.tensor(0, device=device, dtype=torch.long)
    correct_top1 = torch.tensor(0, device=device, dtype=torch.long)
    correct_topk = torch.tensor(0, device=device, dtype=torch.long)

    k = min(topk, C)

    for images, targets in dataloader:
        logits = model(images)  # (B, C)
        preds = logits.argmax(dim=1)  # top-1
        total += targets.size(0)
        correct_top1 += (preds == targets).sum()

        if k > 1:
            topk_idx = logits.topk(k=k, dim=1).indices  # (B, k)
            correct_topk += (topk_idx == targets.unsqueeze(1)).any(dim=1).sum()
        else:
            correct_topk += (preds == targets).sum()

        # Confusion matrix update
        idx = targets * C + preds
        cm_batch = torch.bincount(idx, minlength=C * C).reshape(C, C)
        local_conf += cm_batch

    # gather/reduce
    total = accelerator.gather_for_metrics(total.unsqueeze(0)).sum()
    correct_top1 = accelerator.gather_for_metrics(correct_top1.unsqueeze(0)).sum()
    correct_topk = accelerator.gather_for_metrics(correct_topk.unsqueeze(0)).sum()
    conf_all = accelerator.gather_for_metrics(local_conf.unsqueeze(0))  # (world, C, C)
    conf = conf_all.sum(dim=0)  # -> (C, C)

    acc = (correct_top1.float() / total.clamp_min(1).float()).item()
    acc_topk = (correct_topk.float() / total.clamp_min(1).float()).item()

    # per-class metrics from confusion
    tp = conf.diag().float()
    support = conf.sum(dim=1).float()          # true instances per class
    pred_pos = conf.sum(dim=0).float()         # predicted instances per class

    per_class_recall = (tp / support.clamp_min(1.0)).cpu().tolist()
    per_class_precision = (tp / pred_pos.clamp_min(1.0)).cpu().tolist()
    per_class_f1 = []
    for p, r in zip(per_class_precision, per_class_recall):
        if p + r > 0:
            per_class_f1.append(2 * p * r / (p + r))
        else:
            per_class_f1.append(0.0)

    macro_precision = float((tp / pred_pos.clamp_min(1.0)).mean().item())
    macro_recall    = float((tp / support.clamp_min(1.0)).mean().item())
    macro_f1        = float(np.mean(per_class_f1))
    balanced_acc    = macro_recall

    return {
        "acc": acc,
        "topk_acc": acc_topk,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "support": support.long().cpu().numpy(),
        "pred_pos": pred_pos.long().cpu().numpy(),
        "tp": tp.long().cpu().numpy(),
        "confusion": conf.long().cpu().numpy(),
    }

# ---------- IO helpers & resume/skip ----------

def save_metrics_row(csv_path: Path, row: dict, field_order=None):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order or list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def read_completed_seeds_from_summary(summary_csv: Path, run_name: str) -> set[int]:
    """
    Reads summary.csv and returns a set of seeds that have a row for this run
    AND whose best_ckpt exists on disk (to avoid stale rows).
    """
    done = set()
    if not summary_csv.exists():
        return done
    try:
        df = pd.read_csv(summary_csv)
    except Exception:
        return done
    if df.empty:
        return done
    # Filter by run name
    df = df[df.get("run", "") == run_name]
    if df.empty:
        return done
    for _, r in df.iterrows():
        seed = int(r.get("seed", -1))
        ckpt = r.get("best_ckpt", None)
        try:
            if seed >= 0 and isinstance(ckpt, str) and ckpt and Path(ckpt).is_file():
                done.add(seed)
        except Exception:
            # ignore malformed rows
            pass
    return done

def seed_dir_looks_complete(seed_dir: Path) -> bool:
    """
    Heuristic: consider the seed completed if we have best.pt and at least one metrics JSON
    (best_metrics.json or last_metrics.json). This works even if summary.csv hasn't been written yet.
    """
    best_pt = seed_dir / "best.pt"
    last_json = seed_dir / "last_metrics.json"
    best_json = seed_dir / "best_metrics.json"
    if best_pt.is_file() and (last_json.is_file() or best_json.is_file()):
        return True
    # Also accept: both best.pt and last.pt present
    last_pt = seed_dir / "last.pt"
    if best_pt.is_file() and last_pt.is_file():
        return True
    return False

def dump_labelwise_and_confusion(out_dir: Path, metrics: dict, classes: list[str], prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-class CSV
    per_class_csv = out_dir / f"per_class_{prefix}.csv"
    df = pd.DataFrame({
        "class": classes,
        "precision": metrics["per_class_precision"],
        "recall": metrics["per_class_recall"],
        "f1": metrics["per_class_f1"],
        "support": metrics.get("support", [None]*len(classes)),
        "pred": metrics.get("pred_pos", [None]*len(classes)),
        "tp": metrics.get("tp", [None]*len(classes)),
    })
    df.to_csv(per_class_csv, index=False)

    # Confusion matrix CSV with class headers
    conf_csv = out_dir / f"confusion_{prefix}.csv"
    conf = np.asarray(metrics["confusion"], dtype=np.int64)
    conf_df = pd.DataFrame(conf, columns=classes, index=classes)
    conf_df.index.name = "true\\pred"
    conf_df.to_csv(conf_csv)

    # JSON summary
    json_path = out_dir / f"{prefix}_metrics.json"
    per_class = {}
    for i, c in enumerate(classes):
        per_class[c] = {
            "precision": float(metrics["per_class_precision"][i]),
            "recall": float(metrics["per_class_recall"][i]),
            "f1": float(metrics["per_class_f1"][i]),
            "support": int(metrics.get("support", [0]*len(classes))[i]),
            "pred": int(metrics.get("pred_pos", [0]*len(classes))[i]),
            "tp": int(metrics.get("tp", [0]*len(classes))[i]),
        }

    import json
    json_payload = {
        "aggregates": {
            "acc": float(metrics["acc"]),
            "topk_acc": float(metrics["topk_acc"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "balanced_acc": float(metrics["balanced_acc"]),
        },
        "per_class": per_class,
        "confusion_csv": str(conf_csv),
        "per_class_csv": str(per_class_csv),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    return {
        "per_class_csv": per_class_csv,
        "confusion_csv": conf_csv,
        "json": json_path,
    }

def open_dataset_any(path_str: str, transform, images_key: str,
                     csv_path: str | None, csv_label_col: str):
    p = Path(path_str)

    if p.is_dir():
        if csv_path is None:
            raise ValueError(f"{p} is a directory but no CSV was provided; labels must come from CSV.")
        return DirectoryImagesDataset(root=str(p), csv_path=csv_path, label_col=csv_label_col, transform=transform)

    if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}:
        if csv_path is None:
            raise ValueError(f"{p} is an HDF5 file; please provide a CSV for labels.")
        targets, classes = labels_from_csv(csv_path, csv_label_col)
        return H5ImageDataset(p, targets=targets, classes=classes, transform=transform, images_key=images_key)

    raise ValueError(f"Unsupported train root: {p}. Expected a directory or an .h5/.hdf5 file.")

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _count_images_quick(dir_path: Path, min_needed: int, img_exts: set[str]) -> bool:
    count = 0
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in img_exts:
            count += 1
            if count >= min_needed:
                return True
    return False

def discover_subdatasets(parent: Path, min_images: int = 50000, h5_name: str = "all_images.h5") -> list[Path]:
    results = []
    for child in sorted([d for d in parent.iterdir() if d.is_dir()], key=lambda p: p.name):
        h5_path = child / h5_name
        if h5_path.is_file():
            results.append(h5_path)
            continue
        if _count_images_quick(child, min_images, DirectoryImagesDataset.IMG_EXTS):
            results.append(child)
    return results

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_roots", nargs="+", required=True)
    parser.add_argument("--train_csvs", nargs="+", required=True)

    # EVAL input (choose exactly one)
    parser.add_argument("--eval_csv", default=None, help="Legacy: CSV with label/path for eval (inode-heavy).")
    parser.add_argument("--eval_root", default=None, type=str, help="Base dir to resolve eval CSV relative paths.")
    parser.add_argument("--eval_h5", default=None, help="HDF5 eval with datasets 'images' & 'targets', plus 'classes'.")
    parser.add_argument("--eval_h5_images_key", default="images", type=str)
    parser.add_argument("--eval_h5_targets_key", default="targets", type=str)

    parser.add_argument("--csv_label_col", default="Finding Labels", type=str)
    parser.add_argument("--csv_path_col", default="path", type=str)
    parser.add_argument("--h5_images_key", default="images", type=str)

    # Scanning options
    parser.add_argument("--scan_subdirs", action="store_true")
    parser.add_argument("--scan_min_images", default=50000, type=int)
    parser.add_argument("--scan_h5_name", default="all_images.h5", type=str)

    parser.add_argument("--output_dir", default="runs_cls", type=str)
    parser.add_argument("--model", default="densenet121", choices=["densenet121", "resnet50", "vit_b_16"])
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--imagenet_norm", action="store_true")
    parser.add_argument("--label_smoothing", default=0.0, type=float)
    parser.add_argument("--accum_steps", default=1, type=int)
    parser.add_argument("--topk", default=5, type=int)

    # Distributed / performance toggles (Accelerate-driven)
    parser.add_argument("--mixed_precision", default=None, choices=[None, "fp16", "bf16", "fp8"],
                        help="Overrides accelerate config if set.")
    parser.add_argument("--compile", default=None, choices=[None, "default", "reduce-overhead"],
                        help="Optionally torch.compile the model (PyTorch 2.x).")

    # Resume / skip behavior
    parser.add_argument("--overwrite_existing", action="store_true",
                        help="If set, recompute even if outputs/summary exist for (run, seed).")

    args = parser.parse_args()

    # Validate eval input choice
    if (args.eval_csv is None) == (args.eval_h5 is None):
        raise ValueError("Please specify exactly one of --eval_csv or --eval_h5.")

    # Limit CPU intra-op threads per process to reduce contention on large nodes
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
        kwargs_handlers=None,
    )

    # Make each process seed unique and reproducible across nodes
    base_seed = args.seed if args.seed is not None else 0
    set_seed(base_seed + accelerator.process_index)

    # ---- Expand train roots if requested
    effective_train_roots: list[str] = []
    for root in args.train_roots:
        p = Path(root)
        if args.scan_subdirs and p.is_dir():
            subs = discover_subdatasets(p, min_images=args.scan_min_images, h5_name=args.scan_h5_name)
            if accelerator.is_main_process:
                if subs:
                    print(f"[SCAN] {p} -> {len(subs)} datasets:")
                    for s in subs:
                        print(f"  - {s.parent.name} :: {'H5' if s.suffix.lower() in ('.h5', '.hdf5') else 'DIR'} -> {s}")
                else:
                    print(f"[SCAN] {p} -> 0 qualifying subdirectories (need {args.scan_h5_name} or >= {args.scan_min_images} images).")
            if not subs:
                raise ValueError(f"No qualifying subdirectories found under {p}.")
            effective_train_roots.extend([str(s) for s in subs])
        else:
            effective_train_roots.append(str(p))

    # ---- Effective CSVs
    if len(args.train_csvs) == 1:
        effective_train_csvs = args.train_csvs * len(effective_train_roots)
    elif len(args.train_csvs) == len(effective_train_roots):
        effective_train_csvs = args.train_csvs
    else:
        raise ValueError(
            f"Number of --train_csvs ({len(args.train_csvs)}) must be 1 or exactly equal to the number of train roots ({len(effective_train_roots)})."
        )

    train_tfms, eval_tfms = build_transforms(args.image_size, args.imagenet_norm, args.hflip)

    # ---- Build eval dataset (CSV mode or H5 mode)
    if args.eval_h5:
        eval_ds_raw = H5EvalDataset(
            h5_path=args.eval_h5,
            transform=eval_tfms,
            images_key=args.eval_h5_images_key,
            targets_key=args.eval_h5_targets_key,
        )
    else:
        eval_root_dir = args.eval_root if (args.eval_root and Path(args.eval_root).is_dir()) else None
        eval_ds_raw = CSVImageDataset(
            csv_path=args.eval_csv,
            label_col=args.csv_label_col,
            path_col=args.csv_path_col,
            transform=eval_tfms,
            root_prefix=eval_root_dir
        )

    shared_classes = getattr(eval_ds_raw, "classes", None)
    if not shared_classes:
        raise RuntimeError("Evaluation set has no classes. Check eval input.")
    shared_class_to_idx = {c: i for i, c in enumerate(shared_classes)}
    num_classes = len(shared_classes)

    metrics_csv = Path(args.output_dir) / "summary.csv"

    if len(eval_ds_raw) == 0:
        raise RuntimeError("Evaluation set is empty.")

    # ---- Iterate train roots + CSVs
    for train_root, csv_for_this in zip(effective_train_roots, effective_train_csvs):
        # Determine a consistent run name
        run_name = Path(train_root).parent.name if str(train_root).endswith((".h5", ".hdf5")) else Path(train_root).name
        run_dir = Path(args.output_dir) / run_name
        if accelerator.is_main_process:
            run_dir.mkdir(parents=True, exist_ok=True)

        # Pre-read completed seeds from summary (only once per run, on main)
        completed_from_summary = set()
        if accelerator.is_main_process:
            completed_from_summary = read_completed_seeds_from_summary(metrics_csv, run_name)
            if completed_from_summary:
                print(
                    f"[RESUME] summary.csv indicates completed seeds for run '{run_name}': {sorted(completed_from_summary)}")

        # --- Broadcast rank 0's set to all ranks (no gathers; handles empty cleanly)
        obj = sorted(completed_from_summary) if accelerator.is_main_process else []
        if accelerator.num_processes > 1:
            # Prefer torch.distributed broadcast
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    buf = [obj]  # broadcast_object_list requires a list container
                    dist.broadcast_object_list(buf, src=0)
                    obj = buf[0]
                else:
                    # Fallback to Accelerate's helper if available
                    if hasattr(accelerator, "broadcast_object_list"):
                        buf = [obj]
                        accelerator.broadcast_object_list(buf, from_process=0)
                        obj = buf[0]
            except Exception:
                # Last-resort fallback to Accelerate's helper (if torch.distributed path errored)
                if hasattr(accelerator, "broadcast_object_list"):
                    buf = [obj]
                    accelerator.broadcast_object_list(buf, from_process=0)
                    obj = buf[0]

        # Reconstruct the set on every rank
        completed_from_summary = set(obj)

        # Probe once to announce drops (only if we will train at least one seed or user forces overwrite)
        will_train_any_seed = False

        # Decide seeds
        seeds = args.seeds if args.seeds else ([args.seed] if args.seed is not None else [0, 1, 2])

        # Skip planning: check each seed upfront
        planned_seeds = []
        for seed in seeds:
            seed_dir = run_dir / f"seed_{seed:03d}"
            if args.overwrite_existing:
                planned_seeds.append((seed, False))  # not skipped
                continue

            skip = False
            # Criterion 1: summary.csv confirms completion
            if seed in completed_from_summary:
                skip = True
            else:
                # Criterion 2: seed dir has artifacts
                if seed_dir.exists() and seed_dir_looks_complete(seed_dir):
                    skip = True

            if skip:
                if accelerator.is_main_process:
                    print(f"[SKIP] {run_name} | seed {seed} already completed (summary/outputs detected).")
            else:
                planned_seeds.append((seed, False))
                will_train_any_seed = True

        # If nothing to do for this run and not overwriting, move on
        if not planned_seeds and not args.overwrite_existing:
            if accelerator.is_main_process:
                print(f"[RESUME] No pending seeds for run '{run_name}'.")
            continue

        # Probe drops only if we actually need to train anything (or overwrite)
        if args.overwrite_existing or will_train_any_seed:
            train_ds_base_probe = open_dataset_any(
                train_root, train_tfms, images_key=args.h5_images_key,
                csv_path=csv_for_this, csv_label_col=args.csv_label_col
            )
            _ = RemapTargets(train_ds_base_probe, shared_class_to_idx,
                             is_main_process=accelerator.is_main_process, announce=True)

        # ---- Train each seed (only the pending ones unless overwrite)
        seeds_to_run = seeds if args.overwrite_existing else [s for s, _ in planned_seeds]
        for seed in seeds_to_run:
            seed_dir = run_dir / f"seed_{seed:03d}"
            if accelerator.is_main_process:
                seed_dir.mkdir(parents=True, exist_ok=True)

            # Per-seed deterministic shuffling
            set_seed(seed + accelerator.process_index)

            # Build datasets fresh per-seed
            train_ds_base_seed = open_dataset_any(
                train_root, train_tfms, images_key=args.h5_images_key,
                csv_path=csv_for_this, csv_label_col=args.csv_label_col
            )
            train_ds = RemapTargets(train_ds_base_seed, shared_class_to_idx,
                                    is_main_process=accelerator.is_main_process, announce=False)

            # Wrap eval as well
            eval_ds = RemapTargets(eval_ds_raw, shared_class_to_idx,
                                   is_main_process=accelerator.is_main_process, announce=False)

            g = torch.Generator(); g.manual_seed(seed)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True,
                worker_init_fn=_seed_worker, generator=g, persistent_workers=(args.num_workers > 0)
            )
            eval_loader = DataLoader(
                eval_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
                persistent_workers=(args.num_workers > 0)
            )

            model = infer_model(args.model, num_classes)
            model = maybe_compile(model, args.compile)

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            accelerator.wait_for_everyone()
            model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
                model, optimizer, train_loader, eval_loader, scheduler
            )
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

            best_acc = -1.0
            best_snapshot = None
            best_path = seed_dir / "best.pt"
            last_path = seed_dir / "last.pt"
            last_metrics = None

            if accelerator.is_main_process:
                print(f"\n===== Run: {run_name} | seed {seed} | world_size={accelerator.num_processes} =====")
                print(f"Classes ({num_classes}): {shared_classes}")

            for epoch in range(1, args.epochs + 1):
                model.train()
                epoch_loss = 0.0
                step = 0
                t0 = time.time()

                pbar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process,
                            desc=f"[{run_name} s{seed}] Epoch {epoch}/{args.epochs}")
                optimizer.zero_grad(set_to_none=True)

                for images, targets in train_loader:
                    logits = model(images)
                    loss = criterion(logits, targets)
                    # accelerator.backward divides by gradient_accumulation_steps itself; dividing here as
                    # well scaled gradients twice for accum_steps > 1 (identical for accum_steps == 1).
                    accelerator.backward(loss)

                    if (step + 1) % args.accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    epoch_loss += loss.item()
                    step += 1
                    pbar.update(1)
                if step % args.accum_steps != 0:      # flush the last partial accumulation group of the epoch
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                pbar.close()
                scheduler.step()

                # Evaluate
                last_metrics = evaluate(accelerator, model, eval_loader, num_classes, topk=args.topk)
                acc = last_metrics["acc"]
                if accelerator.is_main_process:
                    dt = time.time() - t0
                    print(f"Epoch {epoch}: train_loss={epoch_loss/max(step, 1):.4f} "
                          f"eval_acc={acc*100:.2f}% top{args.topk}={last_metrics['topk_acc']*100:.2f}% "
                          f"macroF1={last_metrics['macro_f1']:.4f} ({dt:.1f}s)")
                    save_epoch_history(seed_dir, run_name, seed, epoch, last_metrics, topk=args.topk)

                if acc > best_acc:
                    best_acc = acc
                    # Save only on main process to avoid write races
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)
                        torch.save(
                            {"model": unwrapped.state_dict(),
                             "classes": shared_classes,
                             "args": vars(args),
                             "seed": seed,
                             "acc": best_acc,
                             "epoch": epoch},
                            best_path
                        )
                        best_snapshot = {"epoch": epoch, "metrics": last_metrics}
                accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                torch.save(
                    {"model": unwrapped.state_dict(),
                     "classes": shared_classes,
                     "args": vars(args),
                     "seed": seed,
                     "acc_last": (last_metrics["acc"] if last_metrics else None),
                     "epoch": args.epochs},
                    last_path
                )

                # --- Save label-wise + confusion for BEST and LAST ---
                paths_last = dump_labelwise_and_confusion(seed_dir, last_metrics, shared_classes, prefix="last")
                if best_snapshot is not None:
                    paths_best = dump_labelwise_and_confusion(seed_dir, best_snapshot["metrics"], shared_classes, prefix="best")
                else:
                    paths_best = {k: None for k in ("per_class_csv", "confusion_csv", "json")}

                # Summary CSV row (global summary)
                row = {
                    "run": run_name,
                    "seed": seed,
                    "model": args.model,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "image_size": args.image_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "label_smoothing": args.label_smoothing,
                    "acc_best": round(best_acc, 6),
                    "acc_last": round(last_metrics["acc"], 6) if last_metrics else None,
                    "best_ckpt": str(best_path),
                    "last_ckpt": str(last_path),
                    "num_train": len(train_ds),
                    "num_eval": len(eval_ds),
                    "classes": "|".join(shared_classes),
                    "per_class_best_csv": str(paths_best.get("per_class_csv")),
                    "per_class_last_csv": str(paths_last.get("per_class_csv")),
                    "confusion_best_csv": str(paths_best.get("confusion_csv")),
                    "confusion_last_csv": str(paths_last.get("confusion_csv")),
                    "best_metrics_json": str(paths_best.get("json")),
                    "last_metrics_json": str(paths_last.get("json")),
                }
                save_metrics_row(
                    Path(args.output_dir) / "summary.csv",
                    row,
                    field_order=[
                        "run","seed","model","epochs","batch_size","image_size","lr","weight_decay",
                        "label_smoothing","acc_best","acc_last","best_ckpt","last_ckpt",
                        "num_train","num_eval","classes",
                        "per_class_best_csv","per_class_last_csv","confusion_best_csv","confusion_last_csv",
                        "best_metrics_json","last_metrics_json"
                    ]
                )

                print(f"[{run_name} | seed {seed}] Saved label-wise metrics -> \n"
                      f"  BEST: {paths_best.get('per_class_csv')} | Confusion: {paths_best.get('confusion_csv')}\n"
                      f"  LAST: {paths_last.get('per_class_csv')} | Confusion: {paths_last.get('confusion_csv')}")

                print(f"[{run_name} | seed {seed}] Best acc: {best_acc*100:.2f}% -> {best_path.name} | "
                      f"Last acc: {row['acc_last']*100 if row['acc_last'] is not None else float('nan'):.2f}% "
                      f"-> {last_path.name}")

    if accelerator.is_main_process:
        print(f"\nAll done. Summary CSV: {Path(args.output_dir) / 'summary.csv'}")

# ---------- Existing helpers kept at end ----------

def save_epoch_history(seed_dir: Path, run: str, seed: int, epoch: int, metrics: dict, topk: int):
    hist_csv = seed_dir / "metrics_history.csv"
    row = {
        "run": run,
        "seed": seed,
        "epoch": epoch,
        "acc": round(metrics.get("acc", float("nan")), 6),
        f"top{topk}_acc": round(metrics.get("topk_acc", float("nan")), 6),
        "macro_precision": round(metrics.get("macro_precision", float("nan")), 6),
        "macro_recall": round(metrics.get("macro_recall", float("nan")), 6),
        "macro_f1": round(metrics.get("macro_f1", float("nan")), 6),
        "balanced_acc": round(metrics.get("balanced_acc", float("nan")), 6),
    }
    save_metrics_row(hist_csv, row, field_order=list(row.keys()))

if __name__ == "__main__":
    main()
