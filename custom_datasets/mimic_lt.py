from itertools import zip_longest

import h5py
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms
from tqdm import tqdm

from custom_datasets.auto_encoder import AutoencoderKLDataset
from custom_datasets.embeddings import EmbeddingDataset


def mimic_image_transforms():
    return transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])
class MimicLTDataset(Dataset):
    def __init__(self, split, latent_path, meta_data_csv, caption_key="impression", embeddings_path=None):
        self.split = split
        self.meta_data = pd.read_csv(meta_data_csv)
        idx = self.meta_data.index.tolist()
        self.ids = list(self.meta_data["dicom_id"])
        self.labels = list(self.meta_data["Finding Labels"])
        self.latent_path = latent_path
        self.embeddings_path = embeddings_path
        if embeddings_path is not None:
            self.embeddings = EmbeddingDataset(embeddings_path)
        self.latents = AutoencoderKLDataset(latent_path)
        self.latents = Subset(self.latents, idx)

        if caption_key in self.meta_data:
            self.captions = list(self.meta_data[caption_key])
        else:
            raise KeyError(f"{caption_key} not found in meta_data.csv, columns found are {self.meta_data.columns}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (self.latents[idx],
                self.captions[idx],
                self.ids[idx],
                self.labels[idx],
                self.embeddings[idx] if self.embeddings_path is not None else None)

    def get_indices(self):
        return self.ids

    def collate_fn(self, batch):
        lens = {len(x) for x in batch}
        if not lens.issubset({4, 5}):
            raise ValueError(f"Samples must have length 4 or 5, got lengths: {sorted(lens)}")

        if 5 in lens:
            images, captions, ids, labels, embeddings = zip_longest(*batch, fillvalue=None)
        else:
            images, captions, ids, labels = zip(*batch)
            embeddings = None
        return images, captions, ids, labels, embeddings

def get_dataloader(split, latent_path, meta_data_csv, batch_size=16, num_workers=0):
    dataset = MimicLTDataset(split=split, latent_path=latent_path, meta_data_csv=meta_data_csv)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    return dataloader, dataset



