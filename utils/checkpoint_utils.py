import glob
import json
import os

import torch
from safetensors import safe_open


def _list_dir_head(path: str, limit: int = 20) -> str:
    try:
        names = sorted(os.listdir(path))
    except Exception:
        return "<unreadable>"
    if len(names) <= limit:
        return ", ".join(names) if names else "<empty>"
    return ", ".join(names[:limit]) + ", ..."


def load_sharded_state_dict(folder: str, subfolder: str) -> dict:
    dir_path = os.path.join(folder, subfolder)
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Checkpoint subfolder not found: {dir_path}")

    # Preferred path: indexed sharded checkpoint (diffusers-style).
    idx_candidates = sorted(glob.glob(os.path.join(dir_path, "*.index.json")))
    if idx_candidates:
        idx_path = idx_candidates[0]
        with open(idx_path, "r") as f:
            weight_map = json.load(f)["weight_map"]  # param -> shard filename

        # group keys by shard
        shards = {}
        for k, shard in weight_map.items():
            shards.setdefault(shard, []).append(k)

        # load tensors
        state = {}
        for shard, keys in shards.items():
            shard_path = os.path.join(dir_path, shard)
            if shard.endswith(".safetensors"):
                with safe_open(shard_path, framework="pt", device="cpu") as sf:
                    for k in keys:
                        state[k] = sf.get_tensor(k)
            else:  # assume PyTorch .bin
                sd = torch.load(shard_path, map_location="cpu")
                for k in keys:
                    state[k] = sd[k]
        return state

    # Fallback: single-file checkpoint in transformer subfolder.
    for name in (
        "diffusion_pytorch_model.safetensors",
        "model.safetensors",
        "pytorch_model.bin",
        "diffusion_pytorch_model.bin",
    ):
        single = os.path.join(dir_path, name)
        if not os.path.isfile(single):
            continue

        if single.endswith(".safetensors"):
            state = {}
            with safe_open(single, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    state[k] = sf.get_tensor(k)
            return state

        sd = torch.load(single, map_location="cpu")
        if isinstance(sd, dict):
            return sd
        raise ValueError(f"Unexpected checkpoint object type in {single}: {type(sd)}")

    raise FileNotFoundError(
        "No checkpoint index/single-file weights found in transformer subfolder.\n"
        f"Directory: {dir_path}\n"
        f"Entries: {_list_dir_head(dir_path)}"
    )
