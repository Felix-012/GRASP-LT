import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Iterable, Tuple, Optional, Union
from dataclasses import dataclass, astuple


@dataclass
class EmbeddingOutput:
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    text_ids: torch.Tensor

    def __iter__(self):
        return iter(astuple(self))

class EmbeddingDataset(Dataset):
    def __init__(self, h5_file_path: str):
        self.h5_file_path = h5_file_path
        with h5py.File(h5_file_path, "r") as hf:
            self.keys = sorted(hf.keys(), key=lambda x: int(x.split("_")[1]))

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> EmbeddingOutput:
        with h5py.File(self.h5_file_path, "r") as hf:
            grp = hf[self.keys[idx]]
            prompt_embeds = torch.from_numpy(grp["prompt_embeds"][...])
            pooled_prompt_embeds = torch.from_numpy(grp["pooled_prompt_embeds"][...])
            try:
                text_ids = torch.from_numpy(grp["text_ids"][...])
            except KeyError:
                text_ids = None


        return EmbeddingOutput(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids.long() if text_ids is not None else None,
        )

    @staticmethod
    def save_embeddings(
        items: Iterable[
            Union[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                EmbeddingOutput
            ]
        ],
        h5_file_path: str,
        *,
        float_dtype: Optional[str] = None,
        compression: Optional[str] = None,
        compression_opts: Optional[int] = None
    ) -> None:
        """
        Persist an iterable of (prompt_embeds, pooled_prompt_embeds, text_ids) to HDF5.
        Shapes are written exactly as provided. For best interoperability, save without a batch dim.
        """
        with h5py.File(h5_file_path, "w") as hf:
            for idx, item in enumerate(items):
                if isinstance(item, EmbeddingOutput):
                    pe = item.prompt_embeds
                    ppe = item.pooled_prompt_embeds
                    tids = item.text_ids
                else:
                    pe, ppe, tids = item

                # Ensure on CPU numpy
                pe_np = pe.detach().cpu().numpy()
                ppe_np = ppe.detach().cpu().numpy()
                tids_np = tids.detach().cpu().numpy()

                if float_dtype is not None:
                    pe_np = pe_np.astype(float_dtype)
                    ppe_np = ppe_np.astype(float_dtype)

                grp = hf.create_group(f"item_{idx}")
                grp.create_dataset(
                    "prompt_embeds",
                    data=pe_np,
                    compression=compression,
                    compression_opts=compression_opts,
                )
                grp.create_dataset(
                    "pooled_prompt_embeds",
                    data=ppe_np,
                    compression=compression,
                    compression_opts=compression_opts,
                )

                if tids_np.dtype != "int64":
                    tids_np = tids_np.astype("int64")
                grp.create_dataset(
                    "text_ids",
                    data=tids_np,
                    compression=compression,
                    compression_opts=compression_opts,
                )

    def load_ppe(
        self,
        *,
        device: Optional[torch.device] = None,
        cast_float_to: Optional[torch.dtype] = None,
    ) -> torch.Tensor:

        pe_list, ppe_list, tids_list = [], [], []

        with h5py.File(self.h5_file_path, "r") as hf:
            for k in self.keys:
                g = hf[k]
                ppe = g["pooled_prompt_embeds"][...]


                if ppe.ndim == 1:      # [D] -> [1, D]
                    ppe = ppe[None, ...]
                elif ppe.ndim != 2:
                    raise ValueError(f"Unexpected pooled_prompt_embeds shape {ppe.shape} in group {k}")

                ppe_list.append(ppe)


        ppe_all = torch.from_numpy(np.concatenate(ppe_list, axis=0))

        if cast_float_to is not None:
            ppe_all = ppe_all.to(cast_float_to)

        if device is not None:
            ppe_all = ppe_all.to(device)

        return ppe_all
