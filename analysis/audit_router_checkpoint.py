#!/usr/bin/env python3
"""Inspect learned GRASP router weights without materializing the full model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().float().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also print the full embedding and gate tensors.",
    )
    args = parser.parse_args()

    transformer = args.checkpoint / "transformer"
    bin_index_path = transformer / "diffusion_pytorch_model.bin.index.json"
    safe_index_path = transformer / "diffusion_pytorch_model.safetensors.index.json"
    if bin_index_path.is_file():
        index = json.loads(bin_index_path.read_text())["weight_map"]
        shard_format = "bin"
    elif safe_index_path.is_file():
        index = json.loads(safe_index_path.read_text())["weight_map"]
        shard_format = "safetensors"
    else:
        raise FileNotFoundError(
            f"No sharded PyTorch or safetensors index found in {transformer}"
        )

    routing_keys = [
        key
        for key in index
        if key.endswith("router.gate.weight")
        or key.endswith("router.cluster_embed.weight")
    ]
    shards_to_keys: dict[str, list[str]] = defaultdict(list)
    for key in routing_keys:
        shards_to_keys[index[key]].append(key)

    tensors: dict[str, torch.Tensor] = {}
    for shard_name, keys in shards_to_keys.items():
        if shard_format == "bin":
            state = torch.load(
                transformer / shard_name,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        else:
            from safetensors.torch import load_file

            state = load_file(transformer / shard_name, device="cpu")
        for key in keys:
            tensors[key] = state[key].detach().float()
        del state

    gate_keys = sorted(k for k in tensors if k.endswith("router.gate.weight"))
    cluster_keys = sorted(
        k for k in tensors if k.endswith("router.cluster_embed.weight")
    )
    gate_digests = Counter(tensor_digest(tensors[k]) for k in gate_keys)
    cluster_digests = Counter(tensor_digest(tensors[k]) for k in cluster_keys)

    print(f"checkpoint={args.checkpoint}")
    print(f"gate_keys={len(gate_keys)} unique_gate_tensors={len(gate_digests)}")
    print(
        f"cluster_keys={len(cluster_keys)} "
        f"unique_cluster_tensors={len(cluster_digests)}"
    )
    print(f"gate_digest_counts={dict(gate_digests)}")
    print(f"cluster_digest_counts={dict(cluster_digests)}")

    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, tensor in tensors.items():
        prefix, leaf = key.rsplit(".router.", 1)
        pairs[prefix][leaf] = tensor

    routing_patterns = Counter()
    probability_digests = Counter()
    for prefix, pair in pairs.items():
        if set(pair) != {"gate.weight", "cluster_embed.weight"}:
            continue
        logits = pair["cluster_embed.weight"] @ pair["gate.weight"].T
        probs = torch.softmax(logits / args.temperature, dim=-1)
        top1 = tuple(probs.argmax(dim=-1).tolist())
        routing_patterns[top1] += 1
        probability_digests[tensor_digest(probs)] += 1

    print(f"module_pairs={sum(routing_patterns.values())}")
    print(f"top1_pattern_counts={dict(routing_patterns)}")
    print(f"unique_probability_tables={len(probability_digests)}")

    complete_pairs = sorted(
        prefix
        for prefix, pair in pairs.items()
        if set(pair) == {"gate.weight", "cluster_embed.weight"}
    )
    if not complete_pairs:
        raise RuntimeError(f"No complete learned-router pair found in {args.checkpoint}")
    representative_prefix = complete_pairs[0]
    representative = pairs[representative_prefix]
    logits = (
        representative["cluster_embed.weight"]
        @ representative["gate.weight"].T
    )
    probs = torch.softmax(logits / args.temperature, dim=-1)
    print(f"representative={representative_prefix}")
    if args.full:
        print("cluster_embeddings=")
        print(representative["cluster_embed.weight"].tolist())
        print("gate_weight=")
        print(representative["gate.weight"].tolist())
    print("routing_probabilities_rows_are_input_clusters=")
    for cluster_id, row in enumerate(probs.tolist()):
        entropy = -sum(value * torch.log(torch.tensor(value)).item() for value in row)
        print(
            f"cluster={cluster_id} "
            f"top1={max(range(len(row)), key=row.__getitem__)} "
            f"entropy_nats={entropy:.6f} "
            f"probs={','.join(f'{value:.8f}' for value in row)}"
        )


if __name__ == "__main__":
    main()
