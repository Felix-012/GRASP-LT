import h5py
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms
from tqdm import tqdm

from custom_datasets.auto_encoder import AutoencoderKLDataset

SPLIT_MAPPING = {"test":0, "train":1, "val":2, "p19":3, "mscxr":4}

def mimic_image_transforms():
    return transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])
class MimicDataset(Dataset):
    def __init__(self, split, latent_path, meta_data_csv, caption_key="impression"):
        self.split = split
        self.meta_data = pd.read_csv(meta_data_csv)
        idx = self.meta_data.index[self.meta_data["split"] == SPLIT_MAPPING[split]].tolist()
        self.meta_data = self.meta_data[self.meta_data["split"] == SPLIT_MAPPING[split]]
        self.ids = list(self.meta_data["dicom_id"])
        self.labels = list(self.meta_data["Finding Labels"])
        self.latent_path = latent_path

        self.latents = AutoencoderKLDataset(latent_path)
        self.latents = Subset(self.latents, idx)


        if caption_key in self.meta_data:
            self.captions = list(self.meta_data[caption_key])
        else:
            raise KeyError(f"{caption_key} not found in meta_data.csv, columns found are {self.meta_data.columns}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.latents[idx], self.captions[idx], self.ids[idx], self.labels[idx]

    def get_indices(self):
        return self.ids

    def collate_fn(self, batch):
        images, captions, ids, labels = zip(*batch)
        return images, captions, ids, labels


def encode_and_save_text_embeddings(text_encoder, tokenizer, dataset, output_path, batch_size=32, device="cuda"):
    """
    Encodes captions from a Diffusers-style text encoder and saves embeddings to an HDF5 file.

    Args:
        text_encoder (CLIPTextModel): Pretrained text encoder from Diffusers.
        tokenizer (CLIPTokenizer): Matching tokenizer.
        dataset (MimicDataset): Dataset object containing captions and IDs.
        output_path (str): Output HDF5 file path.
        batch_size (int): Batch size for encoding.
        device (str): Device to run on.
    """
    text_encoder.to(device)
    text_encoder.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    all_embeddings = []
    all_ids = []

    with torch.no_grad():
        for _, captions, ids in tqdm(dataloader, desc="Encoding text captions"):
            inputs = tokenizer(captions, padding="max_length",
                                         max_length=77,
                                         truncation=True,
                                         return_tensors="pt",).to(device).input_ids
            embeddings = text_encoder(inputs.to(text_encoder.device),  output_hidden_states=True)
            embeddings = embeddings.hidden_states[-2]

            all_embeddings.append(embeddings.cpu())
            all_ids.extend(ids)

    # Combine and save
    all_embeddings = torch.cat(all_embeddings, dim=0)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("embeddings", data=all_embeddings.numpy())
        f.create_dataset("ids", data=[str(i) for i in all_ids], dtype=h5py.string_dtype(encoding='utf-8'))

    print(f"Saved {len(all_ids)} embeddings to {output_path}")
    return all_embeddings

def get_dataloader(split, latent_path, meta_data_csv, batch_size=16, num_workers=0):
    dataset = MimicDataset(split=split, latent_path=latent_path, meta_data_csv=meta_data_csv)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    return dataloader, dataset



