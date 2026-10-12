from itertools import zip_longest


import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset


from custom_datasets.auto_encoder import AutoencoderKLDataset
from custom_datasets.embeddings import EmbeddingDataset


class ImageNetLT(Dataset):
    def __init__(self, split, latent_path, meta_data_csv, caption_key="impression", embeddings_path=None):
        self.split = split
        self.meta_data = pd.read_csv(meta_data_csv)
        if self.split == "train":
            self.meta_data = self.meta_data.loc[self.meta_data["Split"] == "TRAIN"]
        elif self.split == "test":
            self.meta_data = self.meta_data.loc[self.meta_data["Split"] == "TEST"]
        else:
            print("WARNING: No valid split specified. Defaulting to 'train'")
            self.meta_data = self.meta_data.loc[self.meta_data["Split"] == "TRAIN"]
        idx = self.meta_data.index.tolist()
        self.ids = list(self.meta_data["id"])
        self.labels = list(self.meta_data["Finding Labels"])
        self.captions = self.labels
        self.latent_path = latent_path
        self.embeddings_path = embeddings_path
        if embeddings_path is not None:
            self.embeddings = EmbeddingDataset(embeddings_path)
        self.latents = AutoencoderKLDataset(latent_path)
        self.latents = Subset(self.latents, idx)
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
    dataset = ImageNetLT(split=split, latent_path=latent_path, meta_data_csv=meta_data_csv)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    return dataloader, dataset



