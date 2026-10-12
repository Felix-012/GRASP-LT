# -*- coding: utf-8 -*-
"""
Gradient Consistency Analysis (GCA) for Flux / SD3-style training with:
  - expert-level gradient coherence (per expert, on its active set)
  - within-expert subgroup coherence (dominant label vs other labels)
  - label-level subgroup coherence (optional: per-label subsets within each expert's active set)
  - per-expert purity statistics (dominant label share, entropy, etc.)
  - stable partition_id from a fixed label->expert mapping

Key changes vs your previous version:
  1) DROPPED MoCLE gate rows (hard routing typically yields unused gate grads; those rows were misleading).
  2) FIXED CSV schema ambiguity by adding:
       - analyzed_expert : expert whose params we differentiate w.r.t. (or -1 for no-router)
       - group_id        : subset identifier (expert id, label id, or sentinel)
       - group_label     : human-readable subset label (e.g., label name, "__dominant__", "__other__")
     The legacy "expert" column is kept but now ALWAYS equals analyzed_expert (consistent meaning).
  3) REMOVED trivial routed-mode "subset==all" values:
       - For the expert "active" row, subset-vs-all metrics (gc_paper, mean_grad_cos, evr_sub, conflict_sub)
         are set to NaN because subset==all there by definition.
  4) MADE inter-group cosine meaningful in routed-mode by adding dominant-label vs other-label subgroup rows.
  5) ENSURED gc_num_batches batches are processed by iterating across epochs until the target count.

Assumptions:
  - labels are single-label strings per sample
  - routing is hard; router exposes current_expert_indices and respects router.current_expert_indices override
  - adapter/expert parameters can be selected via name predicate (see _is_expert_e)
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import FluxPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from custom_datasets.mapping import ToIndex
from utils.train_utils import compute_text_embeddings, tokenize_prompt, encode_prompt


# -------------------------
# Fixed label -> expert assignment (yours)
# -------------------------
LABEL_TO_EXPERT: Dict[str, int] = {
    # Expert 0
    "Lung Lesion": 0,
    "Pneumonia": 0,
    "Pneumothorax": 0,
    "Edema": 0,
    "Enlarged Cardiomediastinum": 0,
    "Support Devices": 0,
    "Fracture": 0,
    # Expert 1
    "Lung Opacity": 1,
    "Calcification of the Aorta": 1,
    "Consolidation": 1,
    "Tortuous Aorta": 1,
    "Pleural Other": 1,
    "Pneumoperitoneum": 1,
    "Subcutaneous Emphysema": 1,
    "Pneumomediastinum": 1,
    # Expert 2
    "Atelectasis": 2,
    "Cardiomegaly": 2,
    "Pleural Effusion": 2,
    # Expert 3
    "No Finding": 3,
}


def _norm_label(s: Any) -> str:
    return " ".join(str(s).strip().split())


def _partition_id_from_mapping(label_to_expert: Dict[str, int]) -> str:
    items = sorted(((str(k), int(v)) for k, v in label_to_expert.items()), key=lambda kv: kv[0])
    s = ";".join([f"{k}:{v}" for k, v in items])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


PARTITION_ID: str = _partition_id_from_mapping({_norm_label(k): v for k, v in LABEL_TO_EXPERT.items()})

# Stable label-id mapping for label grouping (deterministic across runs)
_LABEL_ORDER: List[str] = sorted([_norm_label(k) for k in LABEL_TO_EXPERT.keys()])
_LABEL_TO_GROUP_ID: Dict[str, int] = {lab: i for i, lab in enumerate(_LABEL_ORDER)}
_GROUP_ID_TO_LABEL: Dict[int, str] = {i: lab for lab, i in _LABEL_TO_GROUP_ID.items()}


# -------------------------
# Purity helpers
# -------------------------
def _safe_entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return float("nan")
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return float(ent)


def _expert_purity_stats_from_strings(
    *,
    routed: torch.Tensor,          # (B,) long
    labels_str: List[str],         # len B, normalized strings
    num_experts: int,
) -> Dict[int, Dict[str, Any]]:
    routed_list = routed.detach().to("cpu").long().tolist()
    if len(routed_list) != len(labels_str):
        raise ValueError(f"purity: routed len {len(routed_list)} != labels len {len(labels_str)}")

    out: Dict[int, Dict[str, Any]] = {}
    for e in range(num_experts):
        idx = [i for i, ee in enumerate(routed_list) if ee == e]
        n = len(idx)
        if n == 0:
            out[e] = dict(
                expert_count=0,
                purity=float("nan"),
                dominant_label="",
                dominant_frac=float("nan"),
                second_frac=float("nan"),
                num_labels_in_expert=0,
                label_entropy=float("nan"),
            )
            continue

        labs = [labels_str[i] for i in idx]
        ctr = Counter(labs)
        counts_sorted = sorted(ctr.items(), key=lambda kv: kv[1], reverse=True)

        dominant_label, dominant_cnt = counts_sorted[0]
        second_cnt = counts_sorted[1][1] if len(counts_sorted) > 1 else 0

        purity = dominant_cnt / n
        second_frac = second_cnt / n

        out[e] = dict(
            expert_count=n,
            purity=float(purity),
            dominant_label=str(dominant_label),
            dominant_frac=float(purity),
            second_frac=float(second_frac),
            num_labels_in_expert=int(len(ctr)),
            label_entropy=_safe_entropy([c for _, c in counts_sorted]),
        )
    return out


# -------------------------
# Grouping indices for GCA
# -------------------------
def get_group_indices_for_gca(
    *,
    grouping: str,
    device: torch.device,
    routed: Optional[torch.Tensor],
    labels_str: List[str],
) -> torch.Tensor:
    """
    Returns a (B,) long tensor of group ids for subset selection.

    grouping="routed": group by routed expert id (requires routed != None)
    grouping="labels": group by label id (stable mapping from LABEL_TO_EXPERT keys)
    """
    grouping = str(grouping).lower().strip()

    if grouping == "routed":
        if routed is None:
            raise ValueError("grouping='routed' requires routed != None")
        return routed.detach().to(device=device).long()

    if grouping == "labels":
        gids = []
        for s in labels_str:
            s = _norm_label(s)
            if s not in _LABEL_TO_GROUP_ID:
                raise KeyError(
                    f"Unknown label '{s}' not found in LABEL_TO_EXPERT mapping. "
                    f"Known labels: {sorted(_LABEL_TO_GROUP_ID.keys())}"
                )
            gids.append(_LABEL_TO_GROUP_ID[s])
        return torch.tensor(gids, device=device, dtype=torch.long)

    raise ValueError(f"Unknown gc_grouping='{grouping}'. Expected 'routed' or 'labels'.")


# -------------------------
# Gradient consistency helpers
# -------------------------
def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _mean_grad_cos(grads_projected: List[torch.Tensor], grads_projected_sub: List[torch.Tensor]) -> float:
    mu_all = torch.stack(grads_projected, dim=0).mean(dim=0).float()
    mu_sub = torch.stack(grads_projected_sub, dim=0).mean(dim=0).float()
    return _cosine(mu_all, mu_sub)


def _select_params_by_predicate(model, predicate):
    params = []
    for n, p in model.named_parameters():
        if predicate(n, p):
            params.append(p)
    if not params:
        raise ValueError("No parameters matched predicate.")
    return params


def _select_trainable_params(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found (requires_grad=True).")
    return params


def _is_expert_e(name: str, p: torch.nn.Parameter, e: int) -> bool:
    # Adjust this if your adapter parameter naming differs.
    return f"experts.{e}." in name


def _is_any_expert_param(name: str, p: torch.nn.Parameter) -> bool:
    return "experts." in name


@torch.no_grad()
def _project_gaussian(g_flat: torch.Tensor, k: int, seed: int, chunk: int = 16384) -> torch.Tensor:
    """
    y = R^T g, where R^T ~ N(0, 1/k) with shape [k, D], generated on-the-fly in chunks.
    Deterministic for fixed (seed, k, chunk, D).
    """
    device = g_flat.device
    dtype = g_flat.dtype
    y = torch.zeros(k, device=device, dtype=dtype)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    D = g_flat.numel()
    done = 0
    scale = 1.0 / (k ** 0.5)

    while done < D:
        n = min(chunk, D - done)
        R_block_T = torch.randn(k, n, device=device, dtype=dtype, generator=gen) * scale
        g_block = g_flat.narrow(0, done, n)
        y.add_(R_block_T.matmul(g_block))
        done += n

    return y


def _svd_stats(grads_projected: List[torch.Tensor]) -> Tuple[float, float, torch.Tensor]:
    """
    grads_projected: list of (k,) tensors
    returns: (EVR, conflict_frac, v)
      v is the principal direction (k,) normalized.
      EVR = S1^2 / sum(S^2)
      conflict_frac = fraction of samples with negative cosine to v (subset-internal conflict)
    """
    if len(grads_projected) == 0:
        return float("nan"), float("nan"), torch.zeros(1)

    G = torch.stack(grads_projected, dim=1).float()  # (k, N)
    U, S, _Vh = torch.linalg.svd(G, full_matrices=False)

    denom = (S ** 2).sum().clamp_min(1e-12)
    evr = (S[0] ** 2) / denom

    v = U[:, 0]
    v = v / (v.norm() + 1e-12)  # (k,)

    gT = G.T  # (N, k)
    cos_to_v = (gT @ v) / (gT.norm(dim=1) * v.norm() + 1e-12)
    conflict_frac = (cos_to_v < 0).float().mean().item()

    return float(evr.item()), float(conflict_frac), v


def _slice_batch_kwargs(batch_kwargs: Dict, i: int, B: int, device: torch.device) -> Dict:
    """
    Slice all tensors that are batch-shaped with leading dim B.
    Keep scalars and non-batched tensors as-is.
    """
    out = {}
    for k, v in batch_kwargs.items():
        if k == "return_dict":
            out[k] = v
        elif torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == B:
            out[k] = v[i:i + 1].to(device)
        else:
            out[k] = v
    return out


def _loss_per_sample_mse(pred: torch.Tensor, target: torch.Tensor, weighting: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    pred, target: (1, C, H, W)
    weighting: broadcastable to pred-target, or None
    returns scalar
    """
    diff2 = (pred.float() - target.float()) ** 2
    if weighting is not None:
        diff2 = weighting.float() * diff2
    return diff2.reshape(target.shape[0], -1).mean(dim=1).mean()


def _per_sample_grads_projected_routed(
    model_unwrapped,
    params_S: List[torch.nn.Parameter],
    *,
    loss_fn,
    router_obj,
    fixed_expert: Optional[int],
    per_sample_experts: Optional[torch.Tensor],  # (B,) long if fixed_expert is None
    vae_scale_factor: int,
    model_input_: torch.Tensor,
    target_: torch.Tensor,
    weighting_: Optional[torch.Tensor],
    batch_kwargs_: Dict,
    indices_: torch.Tensor,  # 1D long indices into batch
    proj_k_: int,
    proj_seed_: int,
) -> List[torch.Tensor]:
    """
    Compute per-sample projected grads for samples at `indices_`.

    If router_obj is not None:
      - if fixed_expert is not None: force router_obj.current_expert_indices=[fixed_expert] for each sample forward
      - else: force router_obj.current_expert_indices=[per_sample_experts[i]] per sample
    """
    device_ = next(model_unwrapped.parameters()).device
    model_unwrapped.eval()

    B_ = int(batch_kwargs_["hidden_states"].shape[0])
    grads_out: List[torch.Tensor] = []

    prev_idx = getattr(router_obj, "current_expert_indices", None) if router_obj is not None else None
    prev_topk = getattr(router_obj, "current_topk", None) if router_obj is not None else None

    for i in indices_.tolist():
        model_unwrapped.zero_grad(set_to_none=True)

        if router_obj is not None:
            if fixed_expert is not None:
                router_obj.current_expert_indices = torch.tensor([fixed_expert], device=device_)
            else:
                if per_sample_experts is None:
                    raise ValueError(
                        "per_sample_experts must be provided when fixed_expert is None and router_obj is not None."
                    )
                router_obj.current_expert_indices = per_sample_experts[i:i + 1].to(device_)
            # Force explicit routing override for this sliced forward.
            # Otherwise stale full-batch top-k state can mismatch the sliced batch size.
            if hasattr(router_obj, "current_topk"):
                router_obj.current_topk = None

        sample_kwargs = _slice_batch_kwargs(batch_kwargs_, i, B_, device_)

        pred_packed = model_unwrapped(**sample_kwargs)[0]
        pred = FluxPipeline._unpack_latents(
            pred_packed,
            height=model_input_.shape[2] * vae_scale_factor,
            width=model_input_.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

        t_i = target_[i:i + 1]
        if weighting_ is None:
            loss_i = loss_fn(pred, t_i)
        else:
            w_i = weighting_[i:i + 1]
            loss_i = loss_fn(pred, t_i, w_i)

        grads_list = torch.autograd.grad(
            loss_i, params_S, retain_graph=False, create_graph=False, allow_unused=True
        )

        flat = torch.cat(
            [(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads_list, params_S)]
        ).detach()

        proj = _project_gaussian(flat.float(), k=proj_k_, seed=proj_seed_)
        grads_out.append(proj)

    if router_obj is not None:
        router_obj.current_expert_indices = prev_idx
        if hasattr(router_obj, "current_topk"):
            router_obj.current_topk = prev_topk

    return grads_out


def _sample_pair_cosines(
    grads: List[torch.Tensor],
    groups: torch.Tensor,
    *,
    pairs_per_group: int,
    seed: int
) -> Dict[int, Dict[str, Any]]:
    """
    grads: list of projected grads (len=N, each [k])
    groups: (N,) long
    returns dict group_id -> stats (intra_cos, inter_cos, intra_minus_inter, n_intra, n_inter)

    Note:
      - inter_cos is defined for a group only if there exist samples in (groups != g)
    """
    if len(grads) == 0:
        return {}

    G = int(groups.max().item() + 1) if groups.numel() > 0 else 1
    X = torch.stack(grads, dim=0)  # [N,k]
    X = X / (X.norm(dim=1, keepdim=True) + 1e-12)

    gen = torch.Generator(device=X.device)
    gen.manual_seed(int(seed))

    out: Dict[int, Dict[str, Any]] = {}

    idx_by_g: List[torch.Tensor] = []
    for g in range(G):
        idx_by_g.append((groups == g).nonzero(as_tuple=True)[0])

    for g in range(G):
        idx_g = idx_by_g[g]
        n_g = int(idx_g.numel())
        if n_g < 2:
            continue

        # Intra
        n_intra_pairs = min(pairs_per_group, (n_g * (n_g - 1)) // 2)
        intra_vals: List[float] = []
        for _ in range(n_intra_pairs):
            a = idx_g[torch.randint(0, n_g, (1,), generator=gen, device=X.device)].item()
            b = idx_g[torch.randint(0, n_g, (1,), generator=gen, device=X.device)].item()
            if a == b:
                continue
            intra_vals.append(float((X[a] * X[b]).sum().item()))
        intra_mean = float(np.mean(intra_vals)) if intra_vals else float("nan")

        # Inter
        idx_not = (groups != g).nonzero(as_tuple=True)[0]
        n_not = int(idx_not.numel())
        inter_vals: List[float] = []
        if n_not > 0:
            n_inter_pairs = min(pairs_per_group, n_g * n_not)
            for _ in range(n_inter_pairs):
                a = idx_g[torch.randint(0, n_g, (1,), generator=gen, device=X.device)].item()
                b = idx_not[torch.randint(0, n_not, (1,), generator=gen, device=X.device)].item()
                inter_vals.append(float((X[a] * X[b]).sum().item()))
            inter_mean = float(np.mean(inter_vals)) if inter_vals else float("nan")
        else:
            inter_mean = float("nan")

        out[g] = dict(
            intra_cos=intra_mean,
            inter_cos=inter_mean,
            intra_minus_inter=(intra_mean - inter_mean)
            if (np.isfinite(intra_mean) and np.isfinite(inter_mean)) else float("nan"),
            n_intra=int(len(intra_vals)),
            n_inter=int(len(inter_vals)),
        )

    return out


def _dominant_vs_other_groups_from_labels(
    labels_local: List[str],
    dominant_label: str,
    device: torch.device,
) -> torch.Tensor:
    """
    For an expert's active-set labels, return a (N_active,) tensor:
      1 if label == dominant_label else 0
    """
    dom = _norm_label(dominant_label)
    return torch.tensor(
        [1 if _norm_label(lab) == dom else 0 for lab in labels_local],
        device=device,
        dtype=torch.long,
    )


def extract_labels_str_from_batch(batch_obj) -> List[str]:
    """
    Extract normalized label strings from a dataloader batch.
    """
    candidate_keys = []
    for attr in ("LABEL", "FINDING_LABEL", "FINDING_LABELS", "TARGET", "CLS"):
        if hasattr(ToIndex, attr):
            candidate_keys.append(getattr(ToIndex, attr).value)

    last_err = None
    for k in candidate_keys:
        try:
            raw = batch_obj[k]
            return [_norm_label(x) for x in raw]
        except Exception as e:
            last_err = e

    raise KeyError(
        "Could not extract label strings from batch. "
        "Please adjust extract_labels_str_from_batch() to your dataset mapping. "
        f"Tried ToIndex keys: {candidate_keys}. Last error: {repr(last_err)}"
    )


PROJECTED_ALIGNMENT_FIELDNAMES = [
    "step",
    "router_type",
    "analyzed_expert",
    "forward_definition",
    "active_size_full",
    "active_size_used",
    "dominant_size_used",
    "dominant_label",
    "alignment",
    "evr_all",
    "conflict_all",
    "evr_sub",
    "conflict_sub",
    "gc_paper",
    "mean_grad_cos",
    "intra_cos",
    "inter_cos",
    "intra_minus_inter",
    "pairs_intra",
    "pairs_inter",
    "proj_k",
    "proj_seed",
    "train_loss",
    "lr",
    "partition_id",
]

LABEL_PAIR_FIELDNAMES = [
    "step",
    "router_type",
    "param_group",
    "label_i",
    "label_j",
    "count_i_full",
    "count_j_full",
    "count_i_used",
    "count_j_used",
    "grad_cosine",
    "proj_k",
    "proj_seed",
    "train_loss",
    "lr",
    "partition_id",
]


def init_projected_alignment_csv(csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=PROJECTED_ALIGNMENT_FIELDNAMES).writeheader()


def append_projected_alignment_rows(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROJECTED_ALIGNMENT_FIELDNAMES)
        for row in rows:
            writer.writerow(row)


def init_label_pair_csv(csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=LABEL_PAIR_FIELDNAMES).writeheader()


def append_label_pair_rows(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_PAIR_FIELDNAMES)
        for row in rows:
            writer.writerow(row)


def _cap_indices(indices: torch.Tensor, max_count: int, seed: int) -> torch.Tensor:
    if max_count <= 0 or indices.numel() <= max_count:
        return indices
    gen = torch.Generator(device=indices.device)
    gen.manual_seed(int(seed))
    perm = torch.randperm(indices.numel(), generator=gen, device=indices.device)[:max_count]
    return indices.index_select(0, perm)


def _index_batch_kwargs(batch_kwargs: Dict, indices: torch.Tensor, device: torch.device) -> Dict:
    """
    Select a subset of batch-shaped tensors (leading dimension = B).
    Non-batched tensors are kept as-is.
    """
    out = {}
    if "hidden_states" not in batch_kwargs:
        raise KeyError("batch_kwargs must contain 'hidden_states' to infer batch size.")
    B = int(batch_kwargs["hidden_states"].shape[0])
    idx = indices.to(dtype=torch.long)
    for k, v in batch_kwargs.items():
        if k == "return_dict":
            out[k] = v
        elif torch.is_tensor(v) and v.ndim >= 1 and int(v.shape[0]) == B:
            out[k] = v.index_select(0, idx.to(v.device)).to(device)
        else:
            out[k] = v
    return out


def _projected_grad_for_subset(
    model_unwrapped,
    params_S: List[torch.nn.Parameter],
    *,
    router_obj,
    fixed_expert: Optional[int],
    per_sample_experts: Optional[torch.Tensor],
    vae_scale_factor: int,
    model_input_: torch.Tensor,
    target_: torch.Tensor,
    weighting_: Optional[torch.Tensor],
    batch_kwargs_: Dict,
    indices_: torch.Tensor,
    proj_k_: int,
    proj_seed_: int,
) -> torch.Tensor:
    """
    Compute one projected gradient vector for a subset (mean loss over subset).
    """
    device_ = next(model_unwrapped.parameters()).device
    prev_idx = getattr(router_obj, "current_expert_indices", None) if router_obj is not None else None
    prev_topk = getattr(router_obj, "current_topk", None) if router_obj is not None else None

    model_unwrapped.zero_grad(set_to_none=True)

    try:
        if router_obj is not None:
            if fixed_expert is not None:
                router_obj.current_expert_indices = torch.full(
                    (indices_.numel(),),
                    fill_value=int(fixed_expert),
                    device=device_,
                    dtype=torch.long,
                )
            else:
                if per_sample_experts is None:
                    raise ValueError("per_sample_experts is required when router_obj is used without fixed_expert.")
                router_obj.current_expert_indices = per_sample_experts.index_select(0, indices_.to(per_sample_experts.device)).to(device_)
            # Force explicit routing override for this subset forward.
            if hasattr(router_obj, "current_topk"):
                router_obj.current_topk = None

        subset_kwargs = _index_batch_kwargs(batch_kwargs_, indices_, device_)
        pred_packed = model_unwrapped(**subset_kwargs)[0]
        pred = FluxPipeline._unpack_latents(
            pred_packed,
            height=model_input_.shape[2] * vae_scale_factor,
            width=model_input_.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

        target_sub = target_.index_select(0, indices_.to(target_.device))
        weighting_sub = None if weighting_ is None else weighting_.index_select(0, indices_.to(weighting_.device))
        loss_sub = _loss_per_sample_mse(pred, target_sub, weighting_sub)

        grads_list = torch.autograd.grad(
            loss_sub, params_S, retain_graph=False, create_graph=False, allow_unused=True
        )
        flat = torch.cat(
            [(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads_list, params_S)]
        ).detach()
    finally:
        if router_obj is not None:
            router_obj.current_expert_indices = prev_idx
            if hasattr(router_obj, "current_topk"):
                router_obj.current_topk = prev_topk

    model_unwrapped.zero_grad(set_to_none=True)
    return _project_gaussian(flat.float(), k=proj_k_, seed=proj_seed_)


def compute_projected_alignment_for_step(
    *,
    model_unwrapped,
    router,
    args,
    step: int,
    labels_str: List[str],
    model_input: torch.Tensor,
    target: torch.Tensor,
    weighting: Optional[torch.Tensor],
    batch_kwargs: Dict,
    vae_scale_factor: int,
    train_loss: Optional[float] = None,
    lr: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Per-step projected alignment between each expert's active-set gradient and
    its dominant-label subset gradient.
    """
    if not labels_str:
        return []

    device = model_input.device
    gc_proj_k = int(getattr(args, "gc_proj_k", 1024))
    gc_proj_seed = int(getattr(args, "gc_proj_seed", 0))
    min_subset = int(getattr(args, "gc_train_min_subset_size", 2))
    max_samples = int(getattr(args, "gc_train_max_samples_per_expert", 32))
    gc_train_extended_metrics = bool(getattr(args, "gc_train_extended_metrics", False))
    gc_pairs_per_group = int(getattr(args, "gc_pairs_per_group", 256))
    gc_pairs_seed = int(getattr(args, "gc_pairs_seed", 0))
    router_type = str(getattr(args, "router_type", "no"))

    num_labeled = min(len(labels_str), int(model_input.shape[0]))
    if num_labeled < min_subset:
        return []

    idx_batch = torch.arange(num_labeled, device=device, dtype=torch.long)
    rows: List[Dict[str, Any]] = []

    def _build_row(
        *,
        expert_id: int,
        forward_definition: str,
        active_size_full: int,
        active_size_used: int,
        dominant_size_used: int,
        dominant_label: str,
        alignment: float,
        proj_seed: int,
        evr_all: float = float("nan"),
        conflict_all: float = float("nan"),
        evr_sub: float = float("nan"),
        conflict_sub: float = float("nan"),
        gc_paper: float = float("nan"),
        mean_grad_cos: float = float("nan"),
        intra_cos: float = float("nan"),
        inter_cos: float = float("nan"),
        intra_minus_inter: float = float("nan"),
        pairs_intra: int = 0,
        pairs_inter: int = 0,
    ) -> Dict[str, Any]:
        return dict(
            step=int(step),
            router_type=router_type,
            analyzed_expert=int(expert_id),
            forward_definition=str(forward_definition),
            active_size_full=int(active_size_full),
            active_size_used=int(active_size_used),
            dominant_size_used=int(dominant_size_used),
            dominant_label=str(dominant_label),
            alignment=float(alignment),
            evr_all=float(evr_all),
            conflict_all=float(conflict_all),
            evr_sub=float(evr_sub),
            conflict_sub=float(conflict_sub),
            gc_paper=float(gc_paper),
            mean_grad_cos=float(mean_grad_cos),
            intra_cos=float(intra_cos),
            inter_cos=float(inter_cos),
            intra_minus_inter=float(intra_minus_inter),
            pairs_intra=int(pairs_intra),
            pairs_inter=int(pairs_inter),
            proj_k=int(gc_proj_k),
            proj_seed=int(proj_seed),
            train_loss=float(train_loss) if train_loss is not None else float("nan"),
            lr=float(lr) if lr is not None else float("nan"),
            partition_id=PARTITION_ID,
        )

    # -------------------------
    # No-router path
    # -------------------------
    if router is None or router_type == "no":
        active_idx = _cap_indices(idx_batch, max_samples, gc_proj_seed + 9901 * int(step))
        active_labels = [labels_str[i] for i in active_idx.tolist()]
        dominant_label, dom_count = Counter(active_labels).most_common(1)[0]
        dom_local = torch.tensor(
            [i for i, lab in enumerate(active_labels) if _norm_label(lab) == _norm_label(dominant_label)],
            device=device,
            dtype=torch.long,
        )
        if dom_local.numel() >= min_subset and active_idx.numel() >= min_subset:
            dom_idx = active_idx.index_select(0, dom_local)
            proj_seed = gc_proj_seed + 79001 * int(step) + 17
            params_S = _select_trainable_params(model_unwrapped)
            g_active = _projected_grad_for_subset(
                model_unwrapped,
                params_S,
                router_obj=None,
                fixed_expert=None,
                per_sample_experts=None,
                vae_scale_factor=vae_scale_factor,
                model_input_=model_input,
                target_=target,
                weighting_=weighting,
                batch_kwargs_=batch_kwargs,
                indices_=active_idx,
                proj_k_=gc_proj_k,
                proj_seed_=proj_seed,
            )
            g_dom = _projected_grad_for_subset(
                model_unwrapped,
                params_S,
                router_obj=None,
                fixed_expert=None,
                per_sample_experts=None,
                vae_scale_factor=vae_scale_factor,
                model_input_=model_input,
                target_=target,
                weighting_=weighting,
                batch_kwargs_=batch_kwargs,
                indices_=dom_idx,
                proj_k_=gc_proj_k,
                proj_seed_=proj_seed,
            )
            ext = {}
            if gc_train_extended_metrics:
                grads_active = _per_sample_grads_projected_routed(
                    model_unwrapped,
                    params_S,
                    loss_fn=_loss_per_sample_mse,
                    router_obj=None,
                    fixed_expert=None,
                    per_sample_experts=None,
                    vae_scale_factor=vae_scale_factor,
                    model_input_=model_input,
                    target_=target,
                    weighting_=weighting,
                    batch_kwargs_=batch_kwargs,
                    indices_=active_idx,
                    proj_k_=gc_proj_k,
                    proj_seed_=proj_seed + 100003,
                )
                grads_dom = [grads_active[i] for i in dom_local.tolist() if i < len(grads_active)]
                evr_all, conflict_all, v_all = _svd_stats(grads_active)
                evr_sub, conflict_sub, v_sub = _svd_stats(grads_dom)
                dom_vs_other = _dominant_vs_other_groups_from_labels(
                    active_labels,
                    dominant_label,
                    device=device,
                )
                pair_stats = _sample_pair_cosines(
                    grads_active,
                    groups=dom_vs_other,
                    pairs_per_group=gc_pairs_per_group,
                    seed=gc_pairs_seed + 500000 * int(step) + 11,
                )
                ps_dom = pair_stats.get(1, {})
                ext = dict(
                    evr_all=float(evr_all),
                    conflict_all=float(conflict_all),
                    evr_sub=float(evr_sub),
                    conflict_sub=float(conflict_sub),
                    gc_paper=float(_cosine(v_all, v_sub)),
                    mean_grad_cos=float(_mean_grad_cos(grads_active, grads_dom)),
                    intra_cos=float(ps_dom.get("intra_cos", float("nan"))),
                    inter_cos=float(ps_dom.get("inter_cos", float("nan"))),
                    intra_minus_inter=float(ps_dom.get("intra_minus_inter", float("nan"))),
                    pairs_intra=int(ps_dom.get("n_intra", 0)),
                    pairs_inter=int(ps_dom.get("n_inter", 0)),
                )
            rows.append(
                _build_row(
                    expert_id=-1,
                    forward_definition="no-router",
                    active_size_full=int(idx_batch.numel()),
                    active_size_used=int(active_idx.numel()),
                    dominant_size_used=int(dom_idx.numel()),
                    dominant_label=dominant_label,
                    alignment=_cosine(g_active, g_dom),
                    proj_seed=proj_seed,
                    **ext,
                )
            )
        return rows

    # -------------------------
    # Router path
    # -------------------------
    prev_router_indices = getattr(router, "current_expert_indices", None)
    try:
        try:
            current_experts = router.get_experts()
        except TypeError:
            current_experts = router.get_experts(None)
        router.set_current_expert_indices(current_experts, device=device)
        routed_nominal_full = router.current_expert_indices.detach().clone()
        if routed_nominal_full.numel() < num_labeled:
            return rows
        routed_nominal = routed_nominal_full[:num_labeled]

        num_experts = int(
            getattr(
                router,
                "num_experts",
                int(routed_nominal.max().item() + 1) if routed_nominal.numel() > 0 else 1,
            )
        )
        fixed_expert = int(getattr(args, "test_mode_expert", -1))
        is_fixed_forward = fixed_expert >= 0
        forward_definition = f"fixed:{fixed_expert}" if is_fixed_forward else "routed"

        experts = [fixed_expert] if is_fixed_forward else list(range(num_experts))
        for e in experts:
            if is_fixed_forward:
                active_idx_full = idx_batch
                per_sample_experts = None
            else:
                active_idx_full = (routed_nominal == e).nonzero(as_tuple=True)[0]
                per_sample_experts = routed_nominal

            if active_idx_full.numel() < min_subset:
                continue

            active_idx = _cap_indices(
                active_idx_full,
                max_samples,
                gc_proj_seed + 19997 * int(step) + 131 * int(e),
            )
            active_labels = [labels_str[i] for i in active_idx.tolist()]
            dominant_label, _dom_count = Counter(active_labels).most_common(1)[0]
            dom_local = torch.tensor(
                [i for i, lab in enumerate(active_labels) if _norm_label(lab) == _norm_label(dominant_label)],
                device=device,
                dtype=torch.long,
            )
            if dom_local.numel() < min_subset:
                continue
            dom_idx = active_idx.index_select(0, dom_local)

            try:
                params_S = _select_params_by_predicate(model_unwrapped, lambda n, p: _is_expert_e(n, p, e))
            except ValueError:
                continue

            proj_seed = gc_proj_seed + 79001 * int(step) + 97 * int(e + 1)
            g_active = _projected_grad_for_subset(
                model_unwrapped,
                params_S,
                router_obj=router,
                fixed_expert=(fixed_expert if is_fixed_forward else None),
                per_sample_experts=per_sample_experts,
                vae_scale_factor=vae_scale_factor,
                model_input_=model_input,
                target_=target,
                weighting_=weighting,
                batch_kwargs_=batch_kwargs,
                indices_=active_idx,
                proj_k_=gc_proj_k,
                proj_seed_=proj_seed,
            )
            g_dom = _projected_grad_for_subset(
                model_unwrapped,
                params_S,
                router_obj=router,
                fixed_expert=(fixed_expert if is_fixed_forward else None),
                per_sample_experts=per_sample_experts,
                vae_scale_factor=vae_scale_factor,
                model_input_=model_input,
                target_=target,
                weighting_=weighting,
                batch_kwargs_=batch_kwargs,
                indices_=dom_idx,
                proj_k_=gc_proj_k,
                proj_seed_=proj_seed,
            )
            ext = {}
            if gc_train_extended_metrics:
                grads_active = _per_sample_grads_projected_routed(
                    model_unwrapped,
                    params_S,
                    loss_fn=_loss_per_sample_mse,
                    router_obj=router,
                    fixed_expert=(fixed_expert if is_fixed_forward else None),
                    per_sample_experts=per_sample_experts,
                    vae_scale_factor=vae_scale_factor,
                    model_input_=model_input,
                    target_=target,
                    weighting_=weighting,
                    batch_kwargs_=batch_kwargs,
                    indices_=active_idx,
                    proj_k_=gc_proj_k,
                    proj_seed_=proj_seed + 200003,
                )
                grads_dom = [grads_active[i] for i in dom_local.tolist() if i < len(grads_active)]
                evr_all, conflict_all, v_all = _svd_stats(grads_active)
                evr_sub, conflict_sub, v_sub = _svd_stats(grads_dom)
                dom_vs_other = _dominant_vs_other_groups_from_labels(
                    active_labels,
                    dominant_label,
                    device=device,
                )
                pair_stats = _sample_pair_cosines(
                    grads_active,
                    groups=dom_vs_other,
                    pairs_per_group=gc_pairs_per_group,
                    seed=gc_pairs_seed + 900000 * int(step) + int(e),
                )
                ps_dom = pair_stats.get(1, {})
                ext = dict(
                    evr_all=float(evr_all),
                    conflict_all=float(conflict_all),
                    evr_sub=float(evr_sub),
                    conflict_sub=float(conflict_sub),
                    gc_paper=float(_cosine(v_all, v_sub)),
                    mean_grad_cos=float(_mean_grad_cos(grads_active, grads_dom)),
                    intra_cos=float(ps_dom.get("intra_cos", float("nan"))),
                    inter_cos=float(ps_dom.get("inter_cos", float("nan"))),
                    intra_minus_inter=float(ps_dom.get("intra_minus_inter", float("nan"))),
                    pairs_intra=int(ps_dom.get("n_intra", 0)),
                    pairs_inter=int(ps_dom.get("n_inter", 0)),
                )

            rows.append(
                _build_row(
                    expert_id=int(e),
                    forward_definition=forward_definition,
                    active_size_full=int(active_idx_full.numel()),
                    active_size_used=int(active_idx.numel()),
                    dominant_size_used=int(dom_idx.numel()),
                    dominant_label=dominant_label,
                    alignment=_cosine(g_active, g_dom),
                    proj_seed=proj_seed,
                    **ext,
                )
            )
    finally:
        if router is not None:
            router.current_expert_indices = prev_router_indices

    return rows


def compute_label_pair_projected_cosine_for_step(
    *,
    model_unwrapped,
    router,
    args,
    step: int,
    labels_str: List[str],
    model_input: torch.Tensor,
    target: torch.Tensor,
    weighting: Optional[torch.Tensor],
    batch_kwargs: Dict,
    vae_scale_factor: int,
    train_loss: Optional[float] = None,
    lr: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Compute per-step projected gradient cosine for label pairs.

    For each label that appears with enough samples in the current batch, a projected
    gradient is computed on a capped subset. Pairwise cosine similarities between
    label gradients are then written as rows.
    """
    if not labels_str:
        return []

    device = model_input.device
    gc_proj_k = int(getattr(args, "gc_proj_k", 1024))
    gc_proj_seed = int(getattr(args, "gc_proj_seed", 0))
    min_subset = int(getattr(args, "gc_train_label_pair_min_subset_size", 2))
    max_samples = int(getattr(args, "gc_train_label_pair_max_samples_per_label", 32))
    max_labels = int(getattr(args, "gc_train_label_pair_max_labels", 0))
    router_type = str(getattr(args, "router_type", "no"))
    requested_param_group = str(getattr(args, "gc_train_label_pair_param_group", "expert")).lower().strip()

    num_labeled = min(len(labels_str), int(model_input.shape[0]))
    if num_labeled < 2:
        return []

    labels_norm = [_norm_label(x) for x in labels_str[:num_labeled]]
    counts_full = Counter(labels_norm)
    labels_kept = [
        lab for lab, cnt in sorted(counts_full.items(), key=lambda kv: (-kv[1], kv[0])) if int(cnt) >= min_subset
    ]
    if max_labels > 0:
        labels_kept = labels_kept[:max_labels]
    if len(labels_kept) < 2:
        return []

    # Parameter subset selection.
    param_group = requested_param_group
    if requested_param_group == "expert":
        try:
            params_S = _select_params_by_predicate(model_unwrapped, _is_any_expert_param)
        except ValueError:
            params_S = _select_trainable_params(model_unwrapped)
            param_group = "trainable_fallback"
    elif requested_param_group == "trainable":
        params_S = _select_trainable_params(model_unwrapped)
    else:
        raise ValueError(f"Unknown gc_train_label_pair_param_group={requested_param_group}")

    idx_by_label_full: Dict[str, torch.Tensor] = {}
    for lab in labels_kept:
        idx = [i for i, x in enumerate(labels_norm) if x == lab]
        if not idx:
            continue
        idx_by_label_full[lab] = torch.tensor(idx, device=device, dtype=torch.long)

    # Preserve nominal per-sample routing from the training forward pass when available.
    per_sample_experts: Optional[torch.Tensor] = None
    prev_router_indices = getattr(router, "current_expert_indices", None) if router is not None else None
    try:
        if router is not None and router_type != "no":
            if (
                prev_router_indices is not None
                and torch.is_tensor(prev_router_indices)
                and int(prev_router_indices.numel()) >= num_labeled
            ):
                per_sample_experts = prev_router_indices[:num_labeled].detach().to(device=device).long()
            else:
                try:
                    current_experts = router.get_experts()
                except TypeError:
                    current_experts = router.get_experts(None)
                router.set_current_expert_indices(current_experts, device=device)
                routed_now = router.current_expert_indices
                if torch.is_tensor(routed_now) and int(routed_now.numel()) >= num_labeled:
                    per_sample_experts = routed_now[:num_labeled].detach().to(device=device).long()

        router_obj = router if per_sample_experts is not None else None
        proj_seed = gc_proj_seed + 179_429 * int(step) + 53

        # One projected gradient per kept label.
        grads_by_label: Dict[str, torch.Tensor] = {}
        counts_used: Dict[str, int] = {}
        for lab_idx, lab in enumerate(labels_kept):
            idx_full = idx_by_label_full.get(lab)
            if idx_full is None or idx_full.numel() < min_subset:
                continue
            idx_used = _cap_indices(
                idx_full,
                max_samples,
                gc_proj_seed + 9_973 * int(step) + 193 * int(lab_idx + 1),
            )
            if idx_used.numel() < min_subset:
                continue
            g_lab = _projected_grad_for_subset(
                model_unwrapped,
                params_S,
                router_obj=router_obj,
                fixed_expert=None,
                per_sample_experts=per_sample_experts,
                vae_scale_factor=vae_scale_factor,
                model_input_=model_input,
                target_=target,
                weighting_=weighting,
                batch_kwargs_=batch_kwargs,
                indices_=idx_used,
                proj_k_=gc_proj_k,
                proj_seed_=proj_seed,
            )
            grads_by_label[lab] = g_lab
            counts_used[lab] = int(idx_used.numel())

        final_labels = [lab for lab in labels_kept if lab in grads_by_label]
        if len(final_labels) < 2:
            return []

        rows: List[Dict[str, Any]] = []
        for i in range(len(final_labels)):
            li = final_labels[i]
            gi = grads_by_label[li]
            for j in range(i + 1, len(final_labels)):
                lj = final_labels[j]
                gj = grads_by_label[lj]
                rows.append(
                    dict(
                        step=int(step),
                        router_type=router_type,
                        param_group=param_group,
                        label_i=str(li),
                        label_j=str(lj),
                        count_i_full=int(counts_full[li]),
                        count_j_full=int(counts_full[lj]),
                        count_i_used=int(counts_used[li]),
                        count_j_used=int(counts_used[lj]),
                        grad_cosine=float(_cosine(gi, gj)),
                        proj_k=int(gc_proj_k),
                        proj_seed=int(proj_seed),
                        train_loss=float(train_loss) if train_loss is not None else float("nan"),
                        lr=float(lr) if lr is not None else float("nan"),
                        partition_id=PARTITION_ID,
                    )
                )
        return rows
    finally:
        if router is not None:
            router.current_expert_indices = prev_router_indices


def plot_projected_alignment_csv(csv_path: str, out_path: str) -> None:
    """
    Plot mean projected alignment over training steps from a single CSV.
    """
    import matplotlib.pyplot as plt

    if not os.path.exists(csv_path):
        return

    per_step: Dict[int, List[float]] = {}
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = int(row["step"])
                val = float(row["alignment"])
            except Exception:
                continue
            if np.isfinite(val):
                per_step.setdefault(step, []).append(val)

    if not per_step:
        return

    steps = sorted(per_step.keys())
    mean_vals = [float(np.mean(per_step[s])) for s in steps]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, mean_vals, linewidth=2.0)
    plt.xlabel("Training step")
    plt.ylabel("Projected gradient alignment")
    plt.ylim(-1.0, 1.0)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    if out_path.lower().endswith(".png"):
        plt.savefig(out_path[:-4] + ".pdf")
    plt.close()


# -------------------------
# Main entry
# -------------------------
def run_gradient_consistency_analysis(
    *,
    accelerator: Accelerator,
    transformer,
    text_encoder_one,
    text_encoder_two,
    vae,
    noise_scheduler_copy,
    train_dataloader,
    router,
    args,
    weight_dtype,
    tokenizers,
    text_encoders,
):
    """
    Gradient Consistency Analysis (GCA).

    Outputs CSV rows per (step, analyzed_expert, subset_definition[, group_id]).

    Paths:
      - no-router:
          analyzed_expert = -1, param_group="trainable_params"
          writes one "all" row + optional per-label rows
      - router:
          analyzed_expert = expert id (or fixed expert in fixed-forward baseline)
          param_group="expert_params"
          always writes one "active" row per analyzed expert
          optionally writes:
            - "dominant_label" and "other_labels" subgroup rows (dominant-vs-rest)
            - "labels" subgroup rows (per-label within active set) if gc_grouping == "labels"

    Important:
      - For subset_definition="active", subset==all by definition; subset-vs-all fields are NaN.
      - Use evr_all/conflict_all/intra_cos from the "active" row for claim (a).
      - Use dominant_label subgroup rows + purity stats for claim (b).
    """
    device = accelerator.device
    tokenizer_one, tokenizer_two = tokenizers

    # -------------------------
    # Args / defaults
    # -------------------------
    gc_num_batches = int(getattr(args, "gc_num_batches", 0))
    if gc_num_batches <= 0:
        # Convenience mode: <=0 means "one full (prepared/sharded) epoch".
        gc_num_batches = int(len(train_dataloader))
        if gc_num_batches <= 0:
            raise ValueError(
                "Could not infer a positive gc_num_batches from train_dataloader length."
            )
        if accelerator.is_main_process:
            accelerator.print(
                f"[GC] gc_num_batches<=0 -> using one full epoch: {gc_num_batches} batches."
            )

    gc_min_subset_size = int(getattr(args, "gc_min_subset_size", 2))
    gc_grouping = str(getattr(args, "gc_grouping", "routed")).lower().strip()  # "routed" or "labels"

    gc_pairs_per_group = int(getattr(args, "gc_pairs_per_group", 256))
    gc_pairs_seed = int(getattr(args, "gc_pairs_seed", 0))

    gc_proj_k = int(getattr(args, "gc_proj_k", 64))
    gc_proj_seed = int(getattr(args, "gc_proj_seed", 0))
    gc_max_samples = int(getattr(args, "gc_max_samples_per_expert", 256))

    # Whether to write dominant-vs-other subgroup rows (recommended for your purity/correlation claim)
    gc_write_dominant_subsets = bool(getattr(args, "gc_write_dominant_subsets", True))

    if gc_proj_k <= 1 and accelerator.is_main_process:
        accelerator.print(
            f"[GC WARNING] gc_proj_k={gc_proj_k} will make EVR often ~1 by construction. "
            "Use gc_proj_k >= 16 (commonly 64-256)."
        )

    # Determinism-ish: disable router noise if present
    if router is not None:
        try:
            router.eval()
        except Exception:
            pass
        if hasattr(router, "add_noise"):
            router.add_noise = False

    transformer.eval()
    if text_encoder_one is not None:
        text_encoder_one.eval()
    if text_encoder_two is not None:
        text_encoder_two.eval()
    vae.eval()

    # Disable buffering in GC mode
    batch_buffer_size_saved = int(getattr(args, "batch_buffer_size", 0))
    if batch_buffer_size_saved != 0:
        accelerator.print("GC mode: overriding batch_buffer_size to 0 for consistent tensor shapes.")
    args.batch_buffer_size = 0

    out_path = getattr(args, "gc_out", None) or os.path.join(args.output_dir, "gradient_consistency.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(device=device)
        timesteps = timesteps.to(device=device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # CSV schema (expert column kept but consistent: equals analyzed_expert)
    fieldnames = [
        "step",
        "router_type",
        "expert",           # legacy; ALWAYS equals analyzed_expert now
        "analyzed_expert",  # -1 for no-router; else expert id
        "group_id",         # subset id (label id, expert id, or sentinel)
        "group_label",      # label string or sentinel label
        "batch_size",
        "active_size",
        "subset_size",
        "evr_all",
        "conflict_all",
        "evr_sub",
        "conflict_sub",
        "gc_paper",
        "mean_grad_cos",
        "intra_cos",
        "inter_cos",
        "intra_minus_inter",
        "pairs_intra",
        "pairs_inter",
        "analyzed_param_group",
        "subset_definition",   # "all" | "labels" | "active" | "dominant_label" | "other_labels"
        "forward_definition",  # "routed" | "fixed:{id}" | "no-router"
        # purity fields (filled for router expert-level and dominant/other subsets)
        "purity",
        "dominant_label",
        "dominant_frac",
        "second_frac",
        "num_labels_in_expert",
        "label_entropy",
        "partition_id",
    ]

    if accelerator.is_main_process:
        with open(out_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    accelerator.wait_for_everyone()

    # Helper to extract label strings from batch robustly
    def _extract_labels_str(batch_obj) -> List[str]:
        return extract_labels_str_from_batch(batch_obj)

    # Expert param selection: no silent fallback
    def _select_expert_params_or_die(model_unwrapped, expert_id: int):
        params = _select_params_by_predicate(model_unwrapped, lambda n, p: _is_expert_e(n, p, expert_id))
        if not params:
            raise ValueError(
                f"GCA could not find any parameters for expert {expert_id}. "
                f"Check _is_expert_e() naming predicate against model.named_parameters()."
            )
        return params

    def _write_row(row: Dict[str, Any]):
        if accelerator.is_main_process:
            with open(out_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    # -------------------------
    # Iterate until gc_num_batches reached (may span multiple epochs)
    # -------------------------
    model_unwrapped = accelerator.unwrap_model(transformer)

    global_step = 0
    epoch = 0

    while global_step < gc_num_batches:
        # DDP sampler reshuffle, if available
        try:
            if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)
        except Exception:
            pass

        for batch in train_dataloader:
            step = global_step
            global_step += 1
            if step >= gc_num_batches:
                break

            # Routing needs ids for some router types
            if router is not None and str(args.router_type) in [
                "gradient", "clip-cluster", "set", "label", "fake", "textlabel", "mocle", "cluster"
            ]:
                router.set_current_ids(batch[ToIndex.ID.value])

            # Extract label strings (single label per sample)
            labels_str = _extract_labels_str(batch)

            # Images / prompts
            if not args.precomputed_latents_path:
                images = torch.stack(batch[ToIndex.IMG.value]).to(device=device, dtype=vae.dtype)
            else:
                images = batch[ToIndex.IMG.value]

            prompts = batch[ToIndex.CAPTION.value]

            # Embeddings + latents
            with torch.no_grad():
                if not args.train_text_encoder:
                    if args.precomputed_embeddings_path is None:
                        prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                            prompts,
                            text_encoders,
                            tokenizers,
                            max_sequence_length=args.max_sequence_length,
                            device=device,
                        )
                    else:
                        eos = batch[ToIndex.EMBEDDING.value]
                        prompt_embeds = torch.stack([eo.prompt_embeds for eo in eos]).to(device=device, dtype=weight_dtype)
                        pooled_prompt_embeds = torch.stack([eo.pooled_prompt_embeds for eo in eos]).to(device=device, dtype=weight_dtype)
                        try:
                            text_ids = eos[0].text_ids.unsqueeze(0).expand(args.max_sequence_length, -1).to(device=device)
                        except AttributeError:
                            text_ids = torch.zeros(size=(args.max_sequence_length, 3), dtype=weight_dtype, device=device)
                else:
                    tokens_one = tokenize_prompt(tokenizer_one, prompts, max_sequence_length=77)
                    tokens_two = tokenize_prompt(tokenizer_two, prompts, max_sequence_length=args.max_sequence_length)
                    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=[None, None],
                        text_input_ids_list=[tokens_one, tokens_two],
                        max_sequence_length=args.max_sequence_length,
                        device=device,
                        prompt=prompts,
                    )

                if not args.precomputed_latents_path:
                    model_input = vae.encode(images).latent_dist.sample()
                else:
                    model_input = torch.stack([image.latent_dist.sample() for image in images]).squeeze().to(device)

                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype, device=device)

            # Noise + timestep
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            t_idx = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[t_idx].to(device=device)

            sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2] // 2,
                model_input.shape[3] // 2,
                device,
                weight_dtype,
            )

            packed_noisy_model_input = FluxPipeline._pack_latents(
                noisy_model_input,
                batch_size=model_input.shape[0],
                num_channels_latents=model_input.shape[1],
                height=model_input.shape[2],
                width=model_input.shape[3],
            )

            # Guidance
            if model_unwrapped.config.guidance_embeds:
                guidance = torch.tensor([args.guidance_scale], device=device).expand(model_input.shape[0])
            else:
                guidance = None

            target = noise - model_input
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

            batch_kwargs = {
                "hidden_states": packed_noisy_model_input,
                "timestep": timesteps / 1000,
                "guidance": guidance,
                "pooled_projections": pooled_prompt_embeds,
                "encoder_hidden_states": prompt_embeds,
                "txt_ids": text_ids,
                "img_ids": latent_image_ids,
                "return_dict": False,
            }

            idx_batch = torch.arange(model_input.shape[0], device=device, dtype=torch.long)

            # ==========================================================
            # Path 1: no router
            # ==========================================================
            if router is None or str(args.router_type) == "no":
                params_S = _select_trainable_params(model_unwrapped)

                grads_all = _per_sample_grads_projected_routed(
                    model_unwrapped,
                    params_S,
                    loss_fn=_loss_per_sample_mse,
                    router_obj=None,
                    fixed_expert=None,
                    per_sample_experts=None,
                    vae_scale_factor=vae_scale_factor,
                    model_input_=model_input,
                    target_=target,
                    weighting_=weighting,
                    batch_kwargs_=batch_kwargs,
                    indices_=idx_batch,
                    proj_k_=gc_proj_k,
                    proj_seed_=gc_proj_seed + 31 * int(step),
                )

                evr_all, conflict_all, v_all = _svd_stats(grads_all)

                # Write overall "all" row
                row_all = dict(
                    step=int(step),
                    router_type=str(args.router_type),
                    expert=-1,
                    analyzed_expert=-1,
                    group_id=-1,
                    group_label="__all__",
                    batch_size=int(model_input.shape[0]),
                    active_size=int(model_input.shape[0]),
                    subset_size=int(model_input.shape[0]),
                    evr_all=float(evr_all),
                    conflict_all=float(conflict_all),
                    evr_sub=float("nan"),
                    conflict_sub=float("nan"),
                    gc_paper=float("nan"),
                    mean_grad_cos=float("nan"),
                    # intra within all (single group)
                    intra_cos=float(_sample_pair_cosines(
                        grads_all,
                        groups=torch.zeros(len(grads_all), device=device, dtype=torch.long),
                        pairs_per_group=gc_pairs_per_group,
                        seed=gc_pairs_seed + 1000 * int(step),
                    ).get(0, {}).get("intra_cos", float("nan"))),
                    inter_cos=float("nan"),
                    intra_minus_inter=float("nan"),
                    pairs_intra=int(_sample_pair_cosines(
                        grads_all,
                        groups=torch.zeros(len(grads_all), device=device, dtype=torch.long),
                        pairs_per_group=gc_pairs_per_group,
                        seed=gc_pairs_seed + 1000 * int(step),
                    ).get(0, {}).get("n_intra", 0)),
                    pairs_inter=0,
                    analyzed_param_group="trainable_params",
                    subset_definition="all",
                    forward_definition="no-router",
                    purity=float("nan"),
                    dominant_label="",
                    dominant_frac=float("nan"),
                    second_frac=float("nan"),
                    num_labels_in_expert=0,
                    label_entropy=float("nan"),
                    partition_id=PARTITION_ID,
                )
                _write_row(row_all)

                # Per-label rows (use stable label grouping)
                group_idx = get_group_indices_for_gca(
                    grouping="labels",
                    device=device,
                    routed=None,
                    labels_str=labels_str,
                )
                num_groups = int(group_idx.max().item() + 1) if group_idx.numel() > 0 else 1

                pair_stats = _sample_pair_cosines(
                    grads_all, group_idx, pairs_per_group=gc_pairs_per_group, seed=gc_pairs_seed + 1100 * int(step)
                )

                for g in range(num_groups):
                    idx_sub = (group_idx == g).nonzero(as_tuple=True)[0]
                    if idx_sub.numel() < gc_min_subset_size:
                        continue
                    if idx_sub.numel() > gc_max_samples:
                        idx_sub = idx_sub[:gc_max_samples]

                    grads_sub = [grads_all[i] for i in idx_sub.tolist()]
                    evr_sub, conflict_sub, v_sub = _svd_stats(grads_sub)

                    ps = pair_stats.get(g, {})
                    row = dict(
                        step=int(step),
                        router_type=str(args.router_type),
                        expert=-1,  # analyzed_expert is -1; legacy expert kept consistent
                        analyzed_expert=-1,
                        group_id=int(g),
                        group_label=_GROUP_ID_TO_LABEL.get(int(g), ""),
                        batch_size=int(model_input.shape[0]),
                        active_size=int(model_input.shape[0]),
                        subset_size=int(idx_sub.numel()),
                        evr_all=float(evr_all),
                        conflict_all=float(conflict_all),
                        evr_sub=float(evr_sub),
                        conflict_sub=float(conflict_sub),
                        gc_paper=_cosine(v_all, v_sub),
                        mean_grad_cos=_mean_grad_cos(grads_all, grads_sub),
                        intra_cos=float(ps.get("intra_cos", float("nan"))),
                        inter_cos=float(ps.get("inter_cos", float("nan"))),
                        intra_minus_inter=float(ps.get("intra_minus_inter", float("nan"))),
                        pairs_intra=int(ps.get("n_intra", 0)),
                        pairs_inter=int(ps.get("n_inter", 0)),
                        analyzed_param_group="trainable_params",
                        subset_definition="labels",
                        forward_definition="no-router",
                        purity=float("nan"),
                        dominant_label="",
                        dominant_frac=float("nan"),
                        second_frac=float("nan"),
                        num_labels_in_expert=0,
                        label_entropy=float("nan"),
                        partition_id=PARTITION_ID,
                    )
                    _write_row(row)

                if accelerator.is_main_process:
                    accelerator.print(f"[GC] step={step} wrote no-router results to {out_path}")
                continue

            # ==========================================================
            # Path 2/3: router exists
            # ==========================================================
            # Compute nominal routed indices once per batch (for "routed" forward mode).
            router.set_current_expert_indices(router.get_experts(), device=device)
            routed_nominal = router.current_expert_indices.detach().clone()  # (B,)

            num_experts = int(getattr(router, "num_experts", int(routed_nominal.max().item() + 1) if routed_nominal.numel() else 1))

            # Forward definition: routed vs fixed expert
            test_mode_expert = getattr(args, "test_mode_expert", None)
            is_fixed_forward = (test_mode_expert is not None) and (int(test_mode_expert) >= 0)
            fixed_expert = int(test_mode_expert) if is_fixed_forward else None
            forward_definition = (f"fixed:{fixed_expert}" if is_fixed_forward else "routed")

            # For purity stats: if fixed-forward, purity should reflect "all samples are on fixed_expert"
            if is_fixed_forward:
                routed_for_purity = torch.full_like(routed_nominal, fill_value=int(fixed_expert))
            else:
                routed_for_purity = routed_nominal

            purity_stats = _expert_purity_stats_from_strings(
                routed=routed_for_purity,
                labels_str=labels_str,
                num_experts=num_experts,
            )

            # If label grouping is requested, precompute label ids for the batch
            subset_assign_labels = get_group_indices_for_gca(
                grouping="labels",
                device=device,
                routed=routed_nominal,
                labels_str=labels_str,
            )

            # Analyze experts
            for e in range(num_experts):
                if is_fixed_forward:
                    # Only analyze the fixed expert; others are inactive by construction
                    if e != fixed_expert:
                        continue
                    active_idx = idx_batch
                    per_sample_experts = None  # because fixed_expert is used
                else:
                    active_idx = (routed_nominal == e).nonzero(as_tuple=True)[0]
                    per_sample_experts = routed_nominal

                if active_idx.numel() < gc_min_subset_size:
                    continue
                if active_idx.numel() > gc_max_samples:
                    active_idx = active_idx[:gc_max_samples]

                params_S = _select_expert_params_or_die(model_unwrapped, e)

                grads_active = _per_sample_grads_projected_routed(
                    model_unwrapped,
                    params_S,
                    loss_fn=_loss_per_sample_mse,
                    router_obj=router,
                    fixed_expert=fixed_expert,
                    per_sample_experts=per_sample_experts,
                    vae_scale_factor=vae_scale_factor,
                    model_input_=model_input,
                    target_=target,
                    weighting_=weighting,
                    batch_kwargs_=batch_kwargs,
                    indices_=active_idx,
                    proj_k_=gc_proj_k,
                    proj_seed_=gc_proj_seed + 17 * e + 31 * int(step),
                )
                evr_all, conflict_all, v_all = _svd_stats(grads_active)

                # Purity snapshot for this expert (or fixed expert)
                ps_purity = purity_stats.get(e, {})

                # Expert "active" row: subset==all -> subset-vs-all fields are NaN
                ps_intra_all = _sample_pair_cosines(
                    grads_active,
                    groups=torch.zeros(len(grads_active), device=device, dtype=torch.long),
                    pairs_per_group=gc_pairs_per_group,
                    seed=gc_pairs_seed + 2000 * int(step) + e,
                ).get(0, {})

                row_active = dict(
                    step=int(step),
                    router_type=str(args.router_type),
                    expert=int(e),
                    analyzed_expert=int(e),
                    group_id=int(e),
                    group_label="__active__",
                    batch_size=int(model_input.shape[0]),
                    active_size=int(active_idx.numel()),
                    subset_size=int(active_idx.numel()),
                    evr_all=float(evr_all),
                    conflict_all=float(conflict_all),
                    evr_sub=float("nan"),
                    conflict_sub=float("nan"),
                    gc_paper=float("nan"),
                    mean_grad_cos=float("nan"),
                    intra_cos=float(ps_intra_all.get("intra_cos", float("nan"))),
                    inter_cos=float("nan"),
                    intra_minus_inter=float("nan"),
                    pairs_intra=int(ps_intra_all.get("n_intra", 0)),
                    pairs_inter=0,
                    analyzed_param_group="expert_params",
                    subset_definition="active",
                    forward_definition=forward_definition,
                    purity=float(ps_purity.get("purity", float("nan"))),
                    dominant_label=str(ps_purity.get("dominant_label", "")),
                    dominant_frac=float(ps_purity.get("dominant_frac", float("nan"))),
                    second_frac=float(ps_purity.get("second_frac", float("nan"))),
                    num_labels_in_expert=int(ps_purity.get("num_labels_in_expert", 0)),
                    label_entropy=float(ps_purity.get("label_entropy", float("nan"))),
                    partition_id=PARTITION_ID,
                )
                _write_row(row_active)

                # ----------------------------------------------------------
                # Dominant-vs-other subgroup rows (recommended for claim (b))
                # ----------------------------------------------------------
                if gc_write_dominant_subsets and ps_purity.get("dominant_label", ""):
                    labels_local = [labels_str[i] for i in active_idx.tolist()]
                    dom_label = str(ps_purity["dominant_label"])
                    two_groups = _dominant_vs_other_groups_from_labels(labels_local, dom_label, device=device)

                    # Pair stats between dominant (1) and other (0)
                    pair_dom_other = _sample_pair_cosines(
                        grads_active,
                        groups=two_groups,
                        pairs_per_group=gc_pairs_per_group,
                        seed=gc_pairs_seed + 2100 * int(step) + e,
                    )

                    # Dominant subset row
                    dom_local = (two_groups == 1).nonzero(as_tuple=True)[0]
                    if dom_local.numel() >= gc_min_subset_size:
                        grads_dom = [grads_active[i] for i in dom_local.tolist()]
                        evr_sub, conflict_sub, v_sub = _svd_stats(grads_dom)
                        ps = pair_dom_other.get(1, {})

                        # label id if known
                        dom_gid = _LABEL_TO_GROUP_ID.get(_norm_label(dom_label), -1)

                        row_dom = dict(
                            step=int(step),
                            router_type=str(args.router_type),
                            expert=int(e),
                            analyzed_expert=int(e),
                            group_id=int(dom_gid),
                            group_label=str(dom_label),
                            batch_size=int(model_input.shape[0]),
                            active_size=int(active_idx.numel()),
                            subset_size=int(dom_local.numel()),
                            evr_all=float(evr_all),
                            conflict_all=float(conflict_all),
                            evr_sub=float(evr_sub),
                            conflict_sub=float(conflict_sub),
                            gc_paper=_cosine(v_all, v_sub),
                            mean_grad_cos=_mean_grad_cos(grads_active, grads_dom),
                            intra_cos=float(ps.get("intra_cos", float("nan"))),
                            inter_cos=float(ps.get("inter_cos", float("nan"))),  # dom vs other
                            intra_minus_inter=float(ps.get("intra_minus_inter", float("nan"))),
                            pairs_intra=int(ps.get("n_intra", 0)),
                            pairs_inter=int(ps.get("n_inter", 0)),
                            analyzed_param_group="expert_params",
                            subset_definition="dominant_label",
                            forward_definition=forward_definition,
                            purity=float(ps_purity.get("purity", float("nan"))),
                            dominant_label=str(ps_purity.get("dominant_label", "")),
                            dominant_frac=float(ps_purity.get("dominant_frac", float("nan"))),
                            second_frac=float(ps_purity.get("second_frac", float("nan"))),
                            num_labels_in_expert=int(ps_purity.get("num_labels_in_expert", 0)),
                            label_entropy=float(ps_purity.get("label_entropy", float("nan"))),
                            partition_id=PARTITION_ID,
                        )
                        _write_row(row_dom)

                    # Other subset row
                    oth_local = (two_groups == 0).nonzero(as_tuple=True)[0]
                    if oth_local.numel() >= gc_min_subset_size:
                        grads_oth = [grads_active[i] for i in oth_local.tolist()]
                        evr_sub, conflict_sub, v_sub = _svd_stats(grads_oth)
                        ps = pair_dom_other.get(0, {})

                        row_oth = dict(
                            step=int(step),
                            router_type=str(args.router_type),
                            expert=int(e),
                            analyzed_expert=int(e),
                            group_id=-2,
                            group_label="__other__",
                            batch_size=int(model_input.shape[0]),
                            active_size=int(active_idx.numel()),
                            subset_size=int(oth_local.numel()),
                            evr_all=float(evr_all),
                            conflict_all=float(conflict_all),
                            evr_sub=float(evr_sub),
                            conflict_sub=float(conflict_sub),
                            gc_paper=_cosine(v_all, v_sub),
                            mean_grad_cos=_mean_grad_cos(grads_active, grads_oth),
                            intra_cos=float(ps.get("intra_cos", float("nan"))),
                            inter_cos=float(ps.get("inter_cos", float("nan"))),  # other vs dom
                            intra_minus_inter=float(ps.get("intra_minus_inter", float("nan"))),
                            pairs_intra=int(ps.get("n_intra", 0)),
                            pairs_inter=int(ps.get("n_inter", 0)),
                            analyzed_param_group="expert_params",
                            subset_definition="other_labels",
                            forward_definition=forward_definition,
                            purity=float(ps_purity.get("purity", float("nan"))),
                            dominant_label=str(ps_purity.get("dominant_label", "")),
                            dominant_frac=float(ps_purity.get("dominant_frac", float("nan"))),
                            second_frac=float(ps_purity.get("second_frac", float("nan"))),
                            num_labels_in_expert=int(ps_purity.get("num_labels_in_expert", 0)),
                            label_entropy=float(ps_purity.get("label_entropy", float("nan"))),
                            partition_id=PARTITION_ID,
                        )
                        _write_row(row_oth)

                # ----------------------------------------------------------
                # Optional: label-level subgroup rows within active set
                # ----------------------------------------------------------
                if gc_grouping == "labels":
                    local_groups = subset_assign_labels[active_idx]  # (active_size,)
                    local_num_groups = int(local_groups.max().item() + 1) if local_groups.numel() > 0 else 1

                    pair_stats = _sample_pair_cosines(
                        grads_active,
                        local_groups,
                        pairs_per_group=gc_pairs_per_group,
                        seed=gc_pairs_seed + 3000 * int(step) + e,
                    )

                    for g in range(local_num_groups):
                        local_sub = (local_groups == g).nonzero(as_tuple=True)[0]
                        if local_sub.numel() < gc_min_subset_size:
                            continue
                        if local_sub.numel() > gc_max_samples:
                            local_sub = local_sub[:gc_max_samples]

                        grads_sub = [grads_active[i] for i in local_sub.tolist()]
                        evr_sub, conflict_sub, v_sub = _svd_stats(grads_sub)

                        ps = pair_stats.get(g, {})
                        row = dict(
                            step=int(step),
                            router_type=str(args.router_type),
                            expert=int(e),
                            analyzed_expert=int(e),
                            group_id=int(g),
                            group_label=_GROUP_ID_TO_LABEL.get(int(g), ""),
                            batch_size=int(model_input.shape[0]),
                            active_size=int(active_idx.numel()),
                            subset_size=int(local_sub.numel()),
                            evr_all=float(evr_all),
                            conflict_all=float(conflict_all),
                            evr_sub=float(evr_sub),
                            conflict_sub=float(conflict_sub),
                            gc_paper=_cosine(v_all, v_sub),
                            mean_grad_cos=_mean_grad_cos(grads_active, grads_sub),
                            intra_cos=float(ps.get("intra_cos", float("nan"))),
                            inter_cos=float(ps.get("inter_cos", float("nan"))),
                            intra_minus_inter=float(ps.get("intra_minus_inter", float("nan"))),
                            pairs_intra=int(ps.get("n_intra", 0)),
                            pairs_inter=int(ps.get("n_inter", 0)),
                            analyzed_param_group="expert_params",
                            subset_definition="labels",
                            forward_definition=forward_definition,
                            # purity repeated for convenience
                            purity=float(ps_purity.get("purity", float("nan"))),
                            dominant_label=str(ps_purity.get("dominant_label", "")),
                            dominant_frac=float(ps_purity.get("dominant_frac", float("nan"))),
                            second_frac=float(ps_purity.get("second_frac", float("nan"))),
                            num_labels_in_expert=int(ps_purity.get("num_labels_in_expert", 0)),
                            label_entropy=float(ps_purity.get("label_entropy", float("nan"))),
                            partition_id=PARTITION_ID,
                        )
                        _write_row(row)

            if accelerator.is_main_process:
                accelerator.print(f"[GC] step={step} wrote results to {out_path}")

        epoch += 1

    # restore
    args.batch_buffer_size = batch_buffer_size_saved
    accelerator.print(f"[GC] done. results at: {out_path}")
