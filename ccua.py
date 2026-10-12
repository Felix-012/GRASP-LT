"""
CCUA: Contrastive Conditional-Unconditional Alignment for Long-tailed Diffusion.

Reference: Liang et al., "Contrastive Conditional-Unconditional Alignment for
Long-tailed Diffusion Model", arXiv:2507.09052 (2025).

Implements the training-time loss

    L = L_fm + alpha * L_ucl + gamma * L_al

on top of an existing class-conditional Flux flow-matching trainer:

    L_al  = mean_b [ t_b * || v_theta(z_b, t_b, c_b) - v_theta(z_b, t_b, null) ||^2 ]
    L_ucl = - 1/B sum_b log( exp(<h_b,h_b>/tau) /
                             ( exp(<h_b,h_b>/tau) + sum_{j!=b} exp(<h_b,h_j>/tau) ) )

where v_theta is the model's velocity prediction (paper uses noise prediction;
for flow matching this is the equivalent regression target), and h is the output
of the last single transformer block under the null condition.

The paper does not specify alpha and gamma; defaults of 0.5 are used as a
reasonable starting point (see ccua_alpha / ccua_gamma in train_parser.py).
"""

from __future__ import annotations

import os
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def load_null_embeddings(path: str, device, dtype):
    """Load precomputed null/empty-prompt embeddings from an .h5 file.

    Expected keys:
      - prompt_embeds:        [seq_len, d]  (T5 hidden states for the empty string)
      - pooled_prompt_embeds: [d_pool]      (CLIP pooled embedding for the empty string)

    Returns
    -------
    null_prompt_embeds: torch.Tensor [seq_len, d]   (un-broadcast)
    null_pooled_embeds: torch.Tensor [d_pool]       (un-broadcast)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"CCUA: --ccua_null_embeddings_path={path} does not exist. "
            f"Run scripts/precompute_null_embedding.py first."
        )
    with h5py.File(path, "r") as f:
        if "prompt_embeds" not in f or "pooled_prompt_embeds" not in f:
            raise KeyError(
                f"CCUA: {path} must contain datasets 'prompt_embeds' and "
                f"'pooled_prompt_embeds'."
            )
        null_prompt = torch.from_numpy(np.asarray(f["prompt_embeds"]))
        null_pooled = torch.from_numpy(np.asarray(f["pooled_prompt_embeds"]))
    return (
        null_prompt.to(device=device, dtype=dtype),
        null_pooled.to(device=device, dtype=dtype),
    )


def broadcast_null_embeddings(null_prompt, null_pooled, batch_size):
    """Tile a single (seq_len, d) / (d_pool) null embedding across the batch."""
    if null_prompt.dim() == 2:
        prompt = null_prompt.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    else:
        # already batched (e.g., [1, seq_len, d]); just expand
        prompt = null_prompt.expand(batch_size, *null_prompt.shape[1:]).contiguous()
    if null_pooled.dim() == 1:
        pooled = null_pooled.unsqueeze(0).expand(batch_size, -1).contiguous()
    else:
        pooled = null_pooled.expand(batch_size, *null_pooled.shape[1:]).contiguous()
    return prompt, pooled


class FeatureGrabber:
    """Forward hook that grabs the output 'hidden_states' of a Flux block.

    Flux single-transformer blocks return (encoder_hidden_states, hidden_states);
    we capture only `hidden_states` (the image-token stream after the final
    block, before norm_out + proj_out).
    """

    def __init__(self):
        self.feat = None
        self.enabled = False
        self._handle = None

    def attach(self, module):
        # Register hook on the module's forward output.
        def _hook(_mod, _inp, output):
            if not self.enabled:
                return
            # output is (encoder_hidden_states, hidden_states) for single blocks.
            # Be defensive about diffusers version differences.
            if isinstance(output, (tuple, list)) and len(output) == 2:
                self.feat = output[1]
            else:
                self.feat = output

        self._handle = module.register_forward_hook(_hook)
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.feat = None


def compute_alignment_loss(
    model_pred_cond: torch.Tensor,
    model_pred_uncond: torch.Tensor,
    timesteps_norm: torch.Tensor,
) -> torch.Tensor:
    """L_al = mean_b [ t_b * || v_cond - v_uncond ||^2 ].

    Args
    ----
    model_pred_cond: velocity prediction with the real condition, after _unpack_latents.
                     Shape [B, C, H, W].
    model_pred_uncond: velocity prediction with the null condition, same shape.
    timesteps_norm: per-sample timestep in [0, 1]. Shape [B].
    """
    diff = (model_pred_cond.float() - model_pred_uncond.float()) ** 2
    per_sample = diff.reshape(diff.shape[0], -1).mean(dim=1)
    return (timesteps_norm.float() * per_sample).mean()


def compute_unsupervised_contrastive_loss(
    features: torch.Tensor,
    tau: float,
    pool: bool = False,
    timesteps_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_ucl: InfoNCE on encoder features under the null condition.

    Anchor and positive are both the same (no augmentation, as in CCUA paper);
    negatives are other batch elements' features.

    features: [B, T, D] token features from the chosen encoder cut.
    """
    if features.dim() == 3:
        if pool:
            h = features.mean(dim=1)  # [B, D]
        else:
            h = features.flatten(1)  # [B, T*D]
    else:
        h = features

    h = h.float()
    h = F.normalize(h, dim=-1)

    sim = h @ h.t() / tau  # [B, B]
    diag = sim.diagonal()  # [B], all == 1/tau after normalization

    # log-sum-exp over the row (anchor included as positive, and others as negs)
    log_denom = torch.logsumexp(sim, dim=-1)
    per_sample = log_denom - diag

    if timesteps_norm is not None:
        per_sample = timesteps_norm.float() * per_sample

    return per_sample.mean()


def compute_ccua_total_loss(
    fm_loss_per_sample: torch.Tensor,
    model_pred_cond: torch.Tensor,
    model_pred_uncond: torch.Tensor,
    null_features: torch.Tensor,
    timesteps_norm: torch.Tensor,
    args,
):
    """Combine into total CCUA loss.

    Returns (total_loss, dict_of_components_for_logging)
    """
    l_fm = fm_loss_per_sample.mean()
    l_al = compute_alignment_loss(model_pred_cond, model_pred_uncond, timesteps_norm)
    l_ucl_t_weight = timesteps_norm if args.ccua_t_weight_ucl else None
    l_ucl = compute_unsupervised_contrastive_loss(
        null_features,
        tau=args.ccua_tau,
        pool=args.ccua_pool_features,
        timesteps_norm=l_ucl_t_weight,
    )
    total = l_fm + args.ccua_gamma * l_al + args.ccua_alpha * l_ucl
    return total, {
        "ccua/l_fm": l_fm.detach(),
        "ccua/l_al": l_al.detach(),
        "ccua/l_ucl": l_ucl.detach(),
    }
