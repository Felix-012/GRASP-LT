#!/usr/bin/env python
# coding: utf-8

import os
import argparse

import h5py
import pandas as pd
import torch
from transformers import CLIPTokenizer, T5TokenizerFast
from tqdm import tqdm

from utils.train_utils import compute_text_embeddings, load_text_encoders, import_model_class_from_model_name_or_path

DEVICE="cuda"

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute text embeddings to HDF5 (one group per item).")
    parser.add_argument("--csv-path", type=str, required=True, help="CSV file containing a text column.")
    parser.add_argument("--text-column", type=str, default="Finding Labels", help="Column name in CSV with the text/prompt.")
    parser.add_argument("--output-h5-path", type=str, required=True, help="Output HDF5 path.")
    parser.add_argument(
        "--pretrained-model-name-or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Model repo or local path containing text_encoder and text_encoder_2 subfolders.",
    )
    parser.add_argument("--revision", type=str, default=None, help="HF revision/tag.")
    parser.add_argument("--variant", type=str, default=None, help="HF variant (if any).")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Storage dtype for saved float arrays.",
    )
    return parser.parse_args()


def main():
    # Load CSV
    df = pd.read_csv(args.csv_path)
    #df = df.loc[df["Split"] == "TEST"]
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in {args.csv_path}.")
    texts = df[args.text_column].fillna("").astype(str).tolist()

    # Load model classes and encoders
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
        token=os.environ["HUGGINGFACE_HUB_TOKEN"]
    )
    tok2 = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        token=os.environ["HUGGINGFACE_HUB_TOKEN"]
    )

    np_dtype = {
        "float16": "float16",
        "bfloat16": "float32",
        "float32": "float32",
    }[args.dtype]

    os.makedirs(os.path.dirname(args.output_h5_path), exist_ok=True)
    with h5py.File(args.output_h5_path, "w") as hf:
        for start in tqdm(range(0, len(texts), args.batch_size), desc="Encoding"):
            batch_prompts = texts[start : start + args.batch_size]
            pe, ppe, ids = compute_text_embeddings(batch_prompts, (text_encoder_one, text_encoder_two), (tok1, tok2),
                                                    max_sequence_length=args.max_sequence_length,
                                                    device=DEVICE)
            # Move to CPU and possibly cast
            pe = pe.detach().cpu()
            ppe = ppe.detach().cpu()

            if args.dtype == "float16":
                pe = pe.half()
                ppe = ppe.half()
            elif args.dtype == "float32":
                pe = pe.float()
                ppe = ppe.float()
            elif args.dtype == "bfloat16":
                # keep in torch bf16, but will convert to float32 numpy for HDF5
                pass

            # Save each item in the batch as its own group
            bsz = pe.shape[0]
            for i in range(bsz):
                idx = start + i
                grp = hf.create_group(f"item_{idx}")

                # Convert to numpy for h5py; handle bf16 -> float32
                pe_np = pe[i].to(dtype=torch.float32).numpy() if args.dtype == "bfloat16" else pe[i].numpy()
                ppe_np = ppe[i].to(dtype=torch.float32).numpy() if args.dtype == "bfloat16" else ppe[i].numpy()

                grp.create_dataset("prompt_embeds", data=pe_np.astype(np_dtype))
                grp.create_dataset("pooled_prompt_embeds", data=ppe_np.astype(np_dtype))


if __name__ == "__main__":
    args = parse_args()
    main()