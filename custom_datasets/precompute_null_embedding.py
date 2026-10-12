#!/usr/bin/env python
# coding: utf-8
"""Precompute the empty-prompt ("null") text embedding once for CCUA training.

Writes a single .h5 file with two datasets:
  - prompt_embeds        [seq_len, d]   (T5 hidden states for "")
  - pooled_prompt_embeds [d_pool]       (CLIP pooled embedding for "")

These are loaded by ccua.py and broadcast across the batch at training time.

Usage:
    python custom_datasets/precompute_null_embedding.py \
        --output-h5-path "$WORKSPACE/data/mimic-lt/null_embedding.h5" \
        --max-sequence-length 256
"""

import argparse
import os


def configure_hf_cache():
    """Keep Hugging Face cache off $HOME on compute nodes where home is not mounted."""
    workspace = os.environ.get("WORKSPACE")
    if workspace:
        base = os.path.join(workspace, ".cache", "huggingface")
    else:
        base = os.path.join(os.getcwd(), ".cache", "huggingface")

    def set_cache_dir(key, default):
        current = os.environ.get(key)
        if current and not current.startswith("/home/hpc/"):
            try:
                os.makedirs(current, exist_ok=True)
                return
            except OSError:
                pass
        os.environ[key] = default
        os.makedirs(default, exist_ok=True)

    set_cache_dir("HF_HOME", base)
    set_cache_dir("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
    set_cache_dir("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
    set_cache_dir("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))


configure_hf_cache()

import h5py
import torch
from transformers import CLIPTokenizer, T5TokenizerFast

from utils.train_utils import (
    compute_text_embeddings,
    import_model_class_from_model_name_or_path,
    load_text_encoders,
)

DEVICE = "cuda"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute the null/empty-prompt embedding for CCUA."
    )
    parser.add_argument("--output-h5-path", type=str, required=True)
    parser.add_argument(
        "--pretrained-model-name-or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    class_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder"
    )
    class_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_one, text_encoder_two = load_text_encoders(args, class_one, class_two)
    text_encoder_one.eval().to(DEVICE)
    text_encoder_two.eval().to(DEVICE)

    tok1 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    tok2 = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )

    pe, ppe, _ = compute_text_embeddings(
        [""],
        (text_encoder_one, text_encoder_two),
        (tok1, tok2),
        max_sequence_length=args.max_sequence_length,
        device=DEVICE,
    )
    pe = pe.detach().cpu().to(torch.float32).numpy()[0]   # [seq_len, d]
    ppe = ppe.detach().cpu().to(torch.float32).numpy()[0]  # [d_pool]

    os.makedirs(os.path.dirname(args.output_h5_path) or ".", exist_ok=True)
    np_dtype = "float16" if args.dtype == "float16" else "float32"
    with h5py.File(args.output_h5_path, "w") as f:
        f.create_dataset("prompt_embeds", data=pe.astype(np_dtype))
        f.create_dataset("pooled_prompt_embeds", data=ppe.astype(np_dtype))
    print(f"Saved null embedding to {args.output_h5_path}: "
          f"prompt_embeds={pe.shape}, pooled_prompt_embeds={ppe.shape}")


if __name__ == "__main__":
    main()
