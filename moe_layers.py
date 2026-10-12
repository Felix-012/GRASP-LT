import heapq
import random

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward


class FeedForwardMOE(nn.Module):
    def __init__(self, dim, inner_dim, router, target_mixture, number_of_experts=1):
        super().__init__()
        self.router = router
        self.target_mixture = target_mixture
        self.experts = nn.ModuleList([
            FeedForward(dim=dim, inner_dim=inner_dim, activation_fn="gelu-approximate")
            for _ in range(number_of_experts)
        ])

    def forward(self, x):
        expert_indices = self.router.current_expert_indices
        if not torch.is_tensor(expert_indices):
            expert_indices = torch.as_tensor(expert_indices, device=x.device)
        expert_indices = expert_indices.to(device=x.device, dtype=torch.long)


        num_experts = len(self.experts)
        sample_indices = torch.arange(x.shape[0], device=x.device)
        indices_per_expert = [sample_indices[expert_indices == expert] for expert in range(num_experts)]

        outputs = torch.zeros_like(x)

        for expert_i, expert in enumerate(self.experts):
            idx_e = indices_per_expert[expert_i]
            if idx_e.numel() == 0:
                continue

            selected_x = x.index_select(0, idx_e)
            y = expert(selected_x)
            outputs.index_add_(0, idx_e, y.to(x.dtype))

        return outputs

    def copy_weights_from(self, source_ff):
        for expert in self.experts:
            expert.load_state_dict(source_ff.state_dict())

class DeltaFeedForwardMOE(nn.Module):
    def __init__(self, dim, inner_dim, router, target_mixture, base, number_of_experts=1, gamma=1, ortho_only=False):
        super().__init__()
        self.router = router
        self.target_mixture = target_mixture
        self.experts = nn.ModuleList([
            FeedForward(dim=dim, inner_dim=inner_dim, activation_fn="gelu-approximate")
            for _ in range(number_of_experts)
        ])
        for i in range(len(self.experts)):
            self.experts[i].net[-1].weight = nn.init.zeros_(self.experts[i].net[-1].weight)
            self.experts[i].net[-1].bias = nn.init.zeros_(self.experts[i].net[-1].bias)

        self.base = base
        self.gamma = nn.Parameter(torch.tensor(float(gamma)), requires_grad=False)
        self.ortho_only = ortho_only

    def forward(self, x):
        base_out = self.base(x)
        num_experts = len(self.experts)
        residual = torch.zeros_like(x)

        topk = getattr(self.router, "current_topk", None)
        if topk is not None:
            topk_idx, topk_w = topk  # (B, K), (B, K)
            topk_idx = topk_idx.to(x.device, torch.long)
            topk_w = topk_w.to(x.device, x.dtype)

            for expert_i, expert in enumerate(self.experts):
                mask_bk = (topk_idx == expert_i)  # (B, K)
                if not mask_bk.any():
                    continue

                w_per_sample = (topk_w * mask_bk.to(topk_w.dtype)).sum(dim=1)  # (B,)
                idx_e = torch.nonzero(w_per_sample > 0, as_tuple=False).squeeze(-1)
                if idx_e.numel() == 0:
                    continue

                x_e = x.index_select(0, idx_e)
                delta_e = expert(x_e)

                if self.ortho_only:
                    delta_e = _orthogonal_global(base_out.index_select(0, idx_e), delta_e, 1)

                w = w_per_sample.index_select(0, idx_e)
                delta_e = delta_e * w.view(-1, *([1] * (delta_e.ndim - 1)))
                residual.index_add_(0, idx_e, delta_e.to(residual.dtype))
        else:
            expert_indices = self.router.current_expert_indices
            if not torch.is_tensor(expert_indices):
                expert_indices = torch.as_tensor(expert_indices, device=x.device)
            expert_indices = expert_indices.to(device=x.device, dtype=torch.long)
            sample_indices = torch.arange(x.shape[0], device=x.device)
            indices_per_expert = [sample_indices[expert_indices == expert] for expert in range(num_experts)]

            for expert_i, expert in enumerate(self.experts):
                idx_e = indices_per_expert[expert_i]
                if idx_e.numel() == 0:
                    continue
                x_e = x.index_select(0, idx_e)
                delta_e = expert(x_e)

                if self.ortho_only:
                    delta_e = _orthogonal_global(base_out.index_select(0, idx_e), delta_e, 1)

                residual.index_copy_(0, idx_e, delta_e.to(residual.dtype))

        out = (base_out + self.gamma * residual).to(x.dtype)

        return out


class _FFExpert(nn.Module):
    """
    Expert that maps MLP slice H -> D via bottleneck (inner_dim).
    Works on tensors of shape [..., H] and returns [..., D].
    """
    def __init__(self, in_dim: int, inner_dim: int, out_dim: int, activation_fn: str = "gelu-approximate"):
        super().__init__()
        self.down = nn.Linear(in_dim, inner_dim)
        if activation_fn.startswith("gelu"):
            self.act = nn.GELU(approximate="tanh")
        else:
            self.act = nn.GELU()
        self.up = nn.Linear(inner_dim, out_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))

class DeltaProjOutFFMOE(nn.Module):
    """
    Wraps Flux 'proj_out' (D+H -> D) with MoE deltas that look only at the MLP slice (H).
    Out = base(fused) + gamma * Delta, where Delta is built per-expert per-sample.
    - router.current_expert_indices must be a length-B tensor/list of ints in [0, num_experts).
    - Keep base frozen for PEFT-style fine-tuning.
    """
    def __init__(
        self,
        d_model: int,                  # D
        mlp_hidden_dim: int,           # H
        router,
        target_mixture,                # kept for API compatibility; not used here
        base: nn.Module,               # the original proj_out: Linear(D+H -> D)
        number_of_experts: int = 1,
        inner_dim: int = 16,           # bottleneck
        gamma: float = 1.0,
        ortho_only: bool = False,
    ):
        super().__init__()
        self.router = router
        self.target_mixture = target_mixture
        self.base = base
        self.d_model = d_model
        self.h = mlp_hidden_dim
        self.ortho_only = ortho_only


        for p in self.base.parameters():
            p.requires_grad = False

        self.experts = nn.ModuleList([
            _FFExpert(in_dim=self.h, inner_dim=inner_dim, out_dim=self.d_model, activation_fn="gelu-approximate")
            for _ in range(number_of_experts)
        ])

        self.gamma = nn.Parameter(torch.tensor(float(gamma)), requires_grad=False)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        fused: [B, T, D+H] = cat([attn_out(D), mlp_hidden(H)], dim=-1)
        returns: [B, T, D]
        """
        B = fused.shape[0]

        # Base output from frozen proj_out
        base_out = self.base(fused)
        # Slice the MLP portion only
        mlp_part = fused[..., self.d_model:]

        num_experts = len(self.experts)
        residual = torch.zeros_like(base_out)

        topk = getattr(self.router, "current_topk", None)
        if topk is not None:
            topk_idx, topk_w = topk  # (B, K), (B, K)
            topk_idx = topk_idx.to(fused.device, torch.long)
            topk_w = topk_w.to(fused.device, fused.dtype)

            for e_i, expert in enumerate(self.experts):
                mask_bk = (topk_idx == e_i)
                if not mask_bk.any():
                    continue

                w_per_sample = (topk_w * mask_bk.to(topk_w.dtype)).sum(dim=1)  # (B,)
                idx = torch.nonzero(w_per_sample > 0, as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue

                mlp_e = mlp_part.index_select(0, idx)
                delta_e = expert(mlp_e)

                if self.ortho_only:
                    base_e = base_out.index_select(0, idx)
                    delta_e = _orthogonal_global(base_e, delta_e, dim=1)

                w = w_per_sample.index_select(0, idx)
                delta_e = delta_e * w.view(-1, *([1] * (delta_e.ndim - 1)))
                residual.index_add_(0, idx, delta_e.to(residual.dtype))
        else:
            expert_indices = self.router.current_expert_indices
            if not torch.is_tensor(expert_indices):
                expert_indices = torch.as_tensor(expert_indices, device=fused.device)
            expert_indices = expert_indices.to(device=fused.device, dtype=torch.long)

            if expert_indices.ndim == 0:
                expert_indices = expert_indices.expand(B)

            sample_indices = torch.arange(B, device=fused.device)
            indices_per_expert = [sample_indices[expert_indices == e] for e in range(num_experts)]

            for e_i, expert in enumerate(self.experts):
                idx = indices_per_expert[e_i]
                if idx.numel() == 0:
                    continue

                mlp_e = mlp_part.index_select(0, idx)
                delta_e = expert(mlp_e)

                if self.ortho_only:
                    base_e = base_out.index_select(0, idx)
                    delta_e = _orthogonal_global(base_e, delta_e, dim=1)

                residual.index_copy_(0, idx, delta_e.to(residual.dtype))

        out = (base_out + self.gamma * residual).to(fused.dtype)
        return out

def copy_to_moe(
    model,
    router,
    moe_type,
    target_mixture,
    *,
    dual_blocks=None,
    single_blocks=None,
    n=None,
    n_dual=None,
    n_single=None,
    num_experts=1,
    copy_weights=False,
    ortho_only=False,
    bottleneck = None
):
    dual_list = getattr(model, "transformer_blocks", [])
    single_list = getattr(model, "single_transformer_blocks", [])

    def _normalize_indices(indices, total_len):
        norm = set()
        for i in (indices or []):
            i = int(i)
            idx = i if i >= 0 else total_len + i
            if 0 <= idx < total_len:
                norm.add(idx)
        return sorted(norm)

    if n_dual is None and n is not None:
        n_dual = n

    if dual_blocks is not None:
        dual_sel = _normalize_indices(dual_blocks, len(dual_list))
    elif isinstance(n_dual, int) and n_dual > 0:
        dual_sel = list(range(max(0, len(dual_list) - n_dual), len(dual_list)))
    else:
        dual_sel = []

    # Determine which single blocks to modify
    if single_blocks is not None:
        single_sel = _normalize_indices(single_blocks, len(single_list))
    elif isinstance(n_single, int) and n_single > 0:
        single_sel = list(range(max(0, len(single_list) - n_single), len(single_list)))
    else:
        single_sel = []

    def _make_moe(module, type):
        if type == "dual":
            dim = module.net[0].proj.in_features
            if moe_type == "default":
                moe_ff = FeedForwardMOE(dim, bottleneck, router, target_mixture, num_experts)
            elif moe_type == "delta":
                moe_ff = DeltaFeedForwardMOE(
                    dim, bottleneck, router, target_mixture, module, num_experts, ortho_only=ortho_only
                )
            elif moe_type == "mocle":
                moe_ff = MoCLELoRAFF(base_ff=module,dim=dim,router=router,num_experts=num_experts,rank=bottleneck or 64,
                                     alpha=bottleneck or 64, gamma=1.0,ortho_only=ortho_only)
            else:
                raise ValueError(f"Unknown moe_type: {moe_type}")
            if copy_weights and hasattr(moe_ff, "copy_weights_from"):
                moe_ff.copy_weights_from(module)

        else:
            if moe_type == "default":
                raise NotImplementedError("Default MOE is currently only supported for dual blocks.")
            elif moe_type == "delta":
                dim = module.out_features
                moe_ff = DeltaProjOutFFMOE(d_model=dim,
                                           mlp_hidden_dim=mlp_hidden_dim,
                                           router=router,
                                           target_mixture=target_mixture,
                                           base=module,
                                           number_of_experts=num_experts,
                                           inner_dim=bottleneck)
            elif moe_type =="mocle":
                raise NotImplementedError("Default MOE is currently onot supported for mocle.")
        return moe_ff



    for i in dual_sel:
        block = dual_list[i]
        if hasattr(block, "ff") and isinstance(block.ff, FeedForward):
            block.ff = _make_moe(block.ff, "dual")
        if hasattr(block, "ff_context") and isinstance(block.ff_context, FeedForward):
            block.ff_context = _make_moe(block.ff_context, "dual")

    for i in single_sel:
        block = single_list[i]
        mlp_hidden_dim = block.mlp_hidden_dim
        if hasattr(block, "proj_out")  and isinstance(block.proj_out, nn.Linear):
            block.proj_out = _make_moe(block.proj_out, "single")

    return model


def _copy_transfer(
    x,
    expert_indices: torch.Tensor,
    target_mixture,
    num_experts: int = 1,
    total_additions: int = 0,
    random_sampling=False,
    router=None,
    dataset=None,
    vae_factors=None
):

    device = x.device

    # Current counts per expert
    expert_indices = expert_indices.to(device=device, dtype=torch.long)
    counts = torch.bincount(expert_indices, minlength=num_experts).tolist()

    # Original indices per expert (to pass as noisy_indices as you already do)
    per_expert_orig = {
        i: (expert_indices == i).nonzero(as_tuple=False).squeeze(-1).tolist()
        for i in range(num_experts)
    }

    # Decide how many additions each expert gets: allocate one-by-one to the least-loaded expert
    heap = [(cnt, i) for i, cnt in enumerate(counts)]  # (current_load, expert_id)
    heapq.heapify(heap)

    to_add = [0] * num_experts
    for _ in range(int(total_additions)):
        load, eid = heapq.heappop(heap)
        load += 1
        to_add[eid] += 1
        heapq.heappush(heap, (load, eid))

    # Sample the requested number for each expert
    new_latents = {}
    new_embeddings = {}
    new_embeddings_pooled = {}
    for eid, k in enumerate(to_add):
        if k <= 0:
            continue
        other_indices = []
        for index in range(num_experts):
            if index != eid:
                other_indices.extend(per_expert_orig[index])
        if random_sampling:
            indices = random.sample(router.label2member_index[eid], k)
            sampled_latents =  [(dataset.latents[index].latent_dist.sample() - vae_factors[1]) * vae_factors[0] for index in indices]
            sampled_latents = torch.stack(sampled_latents).to(x.device)

            emb_outs = [dataset.embeddings[index] for index in indices]
            sampled_embeddings = torch.stack([out.prompt_embeds for out in emb_outs]).to(x.device)
            sampled_embeddings_pooled = torch.stack([out.pooled_prompt_embeds for out in  emb_outs]).to(x.device)
        else:
            sampled_latents, sampled_embeddings, sampled_embeddings_pooled, probs = target_mixture.sample_furthest_latents(
                reference_batch=target_mixture.reference_batches[eid],
                noisy_indices=other_indices,
                k=k
            )
        if k == 1 and len(sampled_latents.shape) == 3:
            sampled_latents = sampled_latents.unsqueeze(0)
        if k == 1 and len(sampled_embeddings_pooled.shape) == 1:
            sampled_embeddings_pooled = sampled_embeddings_pooled.unsqueeze(0)
        if k == 1 and len(sampled_embeddings.shape) == 2:
            sampled_embeddings = sampled_embeddings.unsqueeze(0)
        new_latents[eid] = sampled_latents
        new_embeddings[eid] = sampled_embeddings
        new_embeddings_pooled[eid] = sampled_embeddings_pooled

    return new_latents, new_embeddings, new_embeddings_pooled



def _orthogonal_global(x: torch.Tensor, f_x: torch.Tensor, dim: int, eps:torch.Tensor=torch.as_tensor(10e-6)) -> torch.Tensor:
    """
    Orthogonal residual connection (global).
    x : residual stream tensor
    f_x : module output tensor (e.g., from a convolutional block)
    dim : starting dimension for flattening (all subsequent dims will be flattened)
    eps : small epsilon tensor for numerical stability
    """
    original_shape = x.shape
    # Convert negative dim to positive for consistent unsqueezing later
    positive_dim_idx = dim if dim >= 0 else len(original_shape) + dim
    eps = eps.to(x.device)
    x_flattened = x.flatten(start_dim=positive_dim_idx)
    f_x_flattened = f_x.flatten(start_dim=positive_dim_idx) #

    dot_product = (x_flattened * f_x_flattened).sum(dim=-1, keepdim=True)
    norm_x_squared = (x_flattened * x_flattened).sum(dim=-1, keepdim=True).float() + eps
    scale_factor = (dot_product / norm_x_squared).to(dtype=x.dtype)
    num_dims_to_unsqueeze = len(original_shape) - (positive_dim_idx + 1)
    for _ in range(num_dims_to_unsqueeze):
        scale_factor = scale_factor.unsqueeze(-1)
    projection_onto_x = scale_factor * x  # Broadcasting happens here
    f_orthogonal = f_x - projection_onto_x

    return f_orthogonal

class MoCLELoRAFF(nn.Module):
    def __init__(
        self,
        base_ff: nn.Module,
        dim: int,
        router,               # MoCLERouterBasic from earlier
        num_experts: int,
        rank: int = 16,
        alpha: int = 16,
        gamma: float = 1.0,
        ortho_only: bool = False,   # you can turn this off to be closer to the paper
    ):
        super().__init__()
        self.base_ff = base_ff
        self.router = router
        self.num_experts = num_experts
        self.gamma = nn.Parameter(torch.tensor(float(gamma)), requires_grad=False)
        self.ortho_only = ortho_only

        # LoRA experts (B,L,D) safe because nn.Linear works on leading dims
        self.experts = nn.ModuleList([])
        scaling = alpha / rank
        for _ in range(num_experts):
            A = nn.Linear(dim, rank, bias=False)
            B = nn.Linear(rank, dim, bias=False)
            nn.init.normal_(A.weight, std=0.01)
            nn.init.zeros_(B.weight)
            self.experts.append(nn.Sequential(A, B))
        self.lora_scaling = scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        base_out = self.base_ff(x)  # (B, L, D)

        # router should already have seen the batch's ids/clusters
        topk = getattr(self.router, "current_topk", None)
        if topk is not None:
            topk_idx_chk, topk_w_chk = topk
            # Guard against stale routing state (e.g., when router state was computed
            # for a different batch size and the current forward is sliced/subsetted).
            topk_invalid = (
                (not torch.is_tensor(topk_idx_chk))
                or (not torch.is_tensor(topk_w_chk))
                or topk_idx_chk.ndim != 2
                or topk_w_chk.ndim != 2
                or topk_idx_chk.shape != topk_w_chk.shape
                or int(topk_idx_chk.shape[0]) != int(B)
            )
            if topk_invalid:
                topk = None

        # legacy fallback: still support a single expert index per sample
        if topk is None:
            expert_indices = self.router.current_expert_indices
            if not torch.is_tensor(expert_indices):
                expert_indices = torch.as_tensor(expert_indices, device=x.device)
            expert_indices = expert_indices.to(x.device, torch.long)

            residual = torch.zeros_like(x)
            all_b = torch.arange(B, device=x.device)
            for e_id, expert in enumerate(self.experts):
                idx_b = all_b[expert_indices == e_id]
                if idx_b.numel() == 0:
                    continue
                x_e = x.index_select(0, idx_b)          # (B_used, L, D)
                delta_e = expert(x_e) * self.lora_scaling
                if self.ortho_only:
                    delta_e = _orthogonal_global(base_out.index_select(0, idx_b), delta_e, dim=-1)
                residual.index_copy_(0, idx_b, delta_e.to(residual.dtype))
            return (base_out + self.gamma * residual).to(x.dtype)

        # MoCLE path: top-k per *batch item*, broadcast to sequence
        topk_idx, topk_w = topk      # (B, K), (B, K)
        topk_idx = topk_idx.to(x.device)
        topk_w = topk_w.to(x.device)

        residual = torch.zeros_like(x)

        for e_id, expert in enumerate(self.experts):
            # (B, K) -> which samples picked this expert
            mask_bk = (topk_idx == e_id)
            if not mask_bk.any():
                continue

            used_mask = mask_bk.any(dim=1)                   # (B,)
            idx_b = used_mask.nonzero(as_tuple=False).squeeze(-1)  # (B_used,)
            x_e = x.index_select(0, idx_b)                   # (B_used, L, D)

            delta_e = expert(x_e) * self.lora_scaling        # (B_used, L, D)

            if self.ortho_only:
                delta_e = _orthogonal_global(
                    base_out.index_select(0, idx_b),
                    delta_e,
                    dim=-1,
                )

            # weight per sample, broadcast over (L, D)
            # torch.argmax on bool is not supported on newer torch versions.
            k_pos = mask_bk[idx_b].to(torch.int64).argmax(dim=1)
            w_e = topk_w[idx_b, k_pos]  # (B_used,)
            delta_e = delta_e * w_e.view(-1, 1, 1)

            residual.index_add_(0, idx_b, delta_e.to(residual.dtype))

        out = (base_out + self.gamma * residual).to(x.dtype)
        return out
