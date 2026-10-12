#!/usr/bin/env python3
"""Encode training images with the FLUX VAE and store the latent distribution per item.

Output HDF5: one group `item_<i>` per CSV row (same order) with datasets `mean` and
`logvar` of shape (16, 128, 128) for 1024x1024 inputs, as read by train_flux.py.

    python custom_datasets/precompute_latents.py \
        --csv $GRASP_DATA/mimic-lt/mimic-cxr-lt_train.csv --image_root $MIMIC_JPG_ROOT \
        --out $GRASP_DATA/mimic-lt/image_latents_flux-lt.h5
"""
import argparse
import os

import h5py
import pandas as pd
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--path_column", default="path", help="column with the image path relative to --image_root")
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--suffix", default="", help="optional suffix appended to every path (e.g. .png)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "black-forest-labs/FLUX.1-dev"))
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--split", default=None, help="optional value of the CSV 'Split' column to keep")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").eval().to(device)
    tf = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    df = pd.read_csv(args.csv)
    if args.split is not None:
        df = df.loc[df["Split"] == args.split]
    paths = df[args.path_column].tolist()

    with h5py.File(args.out, "w") as hf:
        for start in tqdm(range(0, len(paths), args.batch_size), desc="encoding"):
            batch = paths[start:start + args.batch_size]
            x = torch.stack([tf(Image.open(os.path.join(args.image_root, p) + args.suffix).convert("RGB"))
                             for p in batch]).to(device)
            with torch.no_grad():
                dist = vae.encode(x).latent_dist
            mean, logvar = dist.mean.float().cpu(), dist.logvar.float().cpu()
            for i in range(len(batch)):
                g = hf.create_group(f"item_{start + i}")
                g.create_dataset("mean", data=mean[i])
                g.create_dataset("logvar", data=logvar[i])


if __name__ == "__main__":
    main()
