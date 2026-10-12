#!/usr/bin/env python3
"""Check that the environment matches the one used for the paper runs.

    python environment/check_env.py            # versions, GPU, Hugging Face access
    python environment/check_env.py --data     # additionally verify $GRASP_DATA (docs/DATA.md)
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

PINNED = {  # package -> version used for every reported number
    "torch": "2.9.0", "torchvision": "0.24.0", "diffusers": "0.38.0.dev0", "transformers": "4.57.1",
    "accelerate": "1.11.0", "peft": "0.18.1", "bitsandbytes": "0.49.2", "h5py": "3.16.0", "numpy": "2.2.6",
    "scipy": "1.16.3", "sklearn": "1.7.2", "pandas": "2.3.3", "timm": "1.0.21", "pyroaring": "1.0.4",
    "beyondfid": "0.1.0",
}
OPTIONAL = {"metis": "only for partitions/metis_label_partition.py", "wandb": "only with GRASP_REPORT_TO=wandb"}

DATA_FILES = [
    "mimic-lt/mimic-cxr-lt_train.csv", "mimic-lt/mimic-cxr-lt_test.csv",
    "mimic-lt/image_latents_flux-lt.h5", "mimic-lt/embeddings_flux-lt.h5", "mimic-lt/embeddings_flux-lt_test.h5",
    "mimic-lt/embeddings_mimic-lt_labels_train.h5", "mimic-lt/null_embedding.h5",
    "mimic-lt/labels_gt.txt", "mimic-lt/test/labels_gt.txt",
    "mimic-lt/text_cluster/train/labels.txt", "mimic-lt/text_cluster/train/centroids.npz",
    "mimic-lt/mimic-lt_train.h5", "mimic-lt/mimic-lt_test.h5", "nih-cxr-lt/classifier_eval.h5",
]


def version(mod):
    import importlib.metadata as md
    name = {"sklearn": "scikit-learn"}.get(mod, mod)
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return getattr(importlib.import_module(mod), "__version__", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true", help="also check the files under $GRASP_DATA")
    args = ap.parse_args()
    ok = True
    print(f"python {sys.version.split()[0]} (paper: 3.11.14)")
    for mod, want in PINNED.items():
        try:
            importlib.import_module(mod)
            have = version(mod)
            flag = "ok " if have.split("+")[0] == want else "DIFF"
            print(f"  [{flag}] {mod:13s} {have:16s} paper: {want}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [MISS] {mod:13s} {type(e).__name__}: {e}")
    for mod, why in OPTIONAL.items():
        try:
            importlib.import_module(mod)
            print(f"  [ok ] {mod:13s} (optional)")
        except Exception:  # noqa: BLE001
            print(f"  [ -- ] {mod:13s} not installed ({why})")

    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name}, {p.total_memory / 2**30:.0f} GB")
        if torch.cuda.get_device_properties(0).total_memory < 70 * 2**30:
            print("  note: < 80 GB per GPU; only the smoke profile is expected to fit (docs/COMPUTE.md)")
    else:
        print("  no CUDA GPU visible (fine for planning, tests and the CPU-only tools)")

    print(f"  MODEL_NAME={os.environ.get('MODEL_NAME', 'black-forest-labs/FLUX.1-dev (default)')}")
    if not (os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")):
        print("  note: HUGGINGFACE_HUB_TOKEN not set; FLUX.1-dev is a gated model (accept its license on the Hub)")

    if args.data:
        root = os.environ.get("GRASP_DATA")
        if not root:
            sys.exit("GRASP_DATA is not set")
        for rel in DATA_FILES:
            p = Path(root) / rel
            exists = p.exists()
            ok &= exists
            print(f"  [{'ok ' if exists else 'MISS'}] {rel}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
