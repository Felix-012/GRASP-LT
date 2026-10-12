import os

import torch
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from custom_datasets.imagenet_lt import ImageNetLT
from custom_datasets.mimic import MimicDataset
from custom_datasets.mimic_lt import MimicLTDataset


def default_image_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class OpenMMloDataset(Dataset):
    def __init__(self, split, base_path, image_transform=None):
        self.dataset = load_dataset("MiaoMiaoYang/OpenMMlo", data_dir="ImageNet-LT", split=split)
        #unmask_captions(self.dataset)
        self.base_path = base_path
        self.image_transform = image_transform or default_image_transform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rel_image_path = self.dataset[idx]['image']
        full_image_path = os.path.join(self.base_path, rel_image_path)
        caption = self.dataset[idx]['caption']
        image_id = idx

        image = Image.open(full_image_path).convert("RGB")
        image = self.image_transform(image)

        return image, caption, image_id

    def get_indices(self):
        return [i for i in range(len(self.dataset))]


    def collate_fn(self, batch):
        images, captions, ids = zip(*batch)
        images = torch.stack(images)
        return images, captions, ids


def add_unmasked_caption(
    ds: Dataset,
    masked_key: str = "caption",
    synonyms_key: str = "category",
    new_key: str = "caption_unmasked",
    mask_token: str = "[mask]"
) -> Dataset:
    """
    Returns a new Dataset with an extra column `new_key` where all occurrences
    of `mask_token` in `masked_key` are replaced with the first item in `synonyms_key`.
    """
    def _unmask_first(example):
        caption = example.get(masked_key, "")
        syns    = example.get(synonyms_key, [])
        if not caption or not syns:
            return {new_key: caption}
        first = syns[0]
        return {new_key: caption.replace(mask_token, first)}

    return ds.map(_unmask_first, batched=False)





def get_dataloader(args, split, base_path, batch_size=16, num_workers=0, name="mimic-lt", shuffle=None):
    if name == "mimic":
        latent_path = args.precomputed_latents_path
        meta_data_csv = os.path.join(base_path, "mimic_metadata_preprocessed.csv")
        dataset = MimicDataset(split=split, latent_path=latent_path, meta_data_csv=meta_data_csv)
    elif name == "mimic-lt":
        latent_path = args.precomputed_latents_path
        embedding_path = args.precomputed_embeddings_path
        meta_data_csv = os.path.join(base_path, "mimic-cxr-lt_train.csv")
        dataset = MimicLTDataset(split=split,
                               latent_path=latent_path,
                               meta_data_csv=meta_data_csv,
                               embeddings_path=embedding_path)
    elif name == "imagenet-lt":
        latent_path = args.precomputed_latents_path
        embedding_path = args.precomputed_embeddings_path
        # the paper runs read imagenet_labeled-lt.csv (train + val + test with a "Split" column); its TRAIN rows are
        # identical to imagenet_lt_train.csv (custom_datasets/make_imagenet_lt_csv.py), which the release prepares
        meta_data_csv = os.path.join(base_path, "imagenet_lt_train.csv")
        if not os.path.exists(meta_data_csv):
            meta_data_csv = os.path.join(base_path, "imagenet_labeled-lt.csv")
        dataset = ImageNetLT(split=split,
                                 latent_path=latent_path,
                                 meta_data_csv=meta_data_csv,
                                 embeddings_path=embedding_path)

    else:
        dataset = OpenMMloDataset(split=split, base_path=base_path)
    if shuffle is None:
        shuffle = split == "train"
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    return dataloader, dataset

