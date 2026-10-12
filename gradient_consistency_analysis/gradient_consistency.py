import math
import torch
from diffusers import FluxPipeline


@torch.no_grad()
def _flatten_like(params):
    # create a flat buffer with correct total size, but no data
    numel = sum(p.numel() for p in params)
    return torch.zeros(numel, device=params[0].device)

def _flatten_grads(grads):
    return torch.cat([g.reshape(-1) for g in grads])

def select_params(model, predicate):
    """Return list of params to analyze (e.g., a single expert's weights)."""
    params = []
    for n, p in model.named_parameters():
        if predicate(n, p):
            params.append(p)
    if not params:
        raise ValueError("No parameters matched the predicate.")
    return params

@torch.no_grad()
def project_gaussian(g_flat: torch.Tensor, k: int = 1024, seed: int = 0, chunk: int = 16384):
    """
    y = R^T g, R^T ~ N(0, 1/k) with shape [k, D], generated on-the-fly in column chunks.
    Memory: O(k + k*chunk). Time: O(k*D).
    Deterministic for a given (seed, chunk, k, D).
    """
    device = g_flat.device
    dtype  = g_flat.dtype
    y = torch.zeros(k, device=device, dtype=dtype)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    D = g_flat.numel()
    done = 0
    scale = 1.0 / (k ** 0.5)
    while done < D:
        n = min(chunk, D - done)
        # R_block_T: [k, n]
        R_block_T = torch.randn(k, n, device=device, dtype=dtype, generator=gen) * scale
        g_block = g_flat.narrow(0, done, n)
        # y += R_block_T @ g_block
        y.add_(R_block_T.matmul(g_block))
        done += n
    return y

def principal_dir_from_grads_svd(grads):
    """
    grads: list[Tensor(P,)]
    Returns: g_dir (P,), u_max (N,), lam_max (float)
    """
    U, S, Vh = torch.linalg.svd(torch.stack(grads, dim=1).float(), full_matrices=False)
    g_dir = U[:, 0]
    g_dir = g_dir / (g_dir.norm() + 1e-12)
    lam_max = (S[0]**2).item()
    u_max = Vh[0].T
    return g_dir, u_max, lam_max

def per_sample_grad_stream(model, params_S, loss_fn, router, vae_scale_factor, model_input, target, batch):
    """
    Returns a list of flattened per-sample grads.
    """
    device = next(model.parameters()).device
    model.eval()

    keep_whole = {"txt_ids", "img_ids", "return_dict"}
    B = batch["hidden_states"].shape[0]

    grads_cache = []
    for i in range(B):
        model.zero_grad(set_to_none=True)


        sample = {}
        for k, v in batch.items():
            if k in keep_whole:
                sample[k] = v
            elif torch.is_tensor(v):
                sample[k] = v[i:i+1].to(device)
            else:

                sample[k] = v[i:i+1]

        # route only this sample
        prev_idx = router.current_expert_indices
        try:
            router.current_expert_indices = prev_idx[i:i+1]

            pred = model(**sample)[0]
            pred = FluxPipeline._unpack_latents(
                pred,
                height=model_input.shape[2] * vae_scale_factor,
                width=model_input.shape[3] * vae_scale_factor,
                vae_scale_factor=vae_scale_factor,
            )

            loss_i = loss_fn(m=pred, t=target[i:i+1])
            grads_list = torch.autograd.grad(
                loss_i, params_S, retain_graph=False, create_graph=False, allow_unused=True
            )

        finally:
            router.current_expert_indices = prev_idx

        # flatten (replace None with zeros of matching shape)
        flat = torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads_list, params_S)
        ]).detach()
        flat_proj = project_gaussian(flat, 1024)
        grads_cache.append(flat_proj)

    return grads_cache

def get_top_direction(model, params_S, loss_fn, router, *, vae_scale_factor, model_input, target, model_kwargs):
    grads_cache = per_sample_grad_stream(model, params_S, loss_fn, router, vae_scale_factor, model_input, target, model_kwargs)
    if not grads_cache:
        raise ValueError("Empty stream: no samples.")
    return principal_dir_from_grads_svd(grads_cache)



def cosine(g_all, g_subset):
    return float(torch.dot(g_all, g_subset) / (g_all.norm() * g_subset.norm() + 1e-12))


def is_expert_e(name, p, e=0):
    return f"experts.{e}." in name



# 4) Compute directions and GC
# params_S = select_params(model, lambda n,p: is_expert_e(n,p,e=0))
# g_all = power_iteration_top_direction(model, params_S, make_all_batches, loss_fn, router_mask_fn=None)
# g_sub = power_iteration_top_direction(model, params_S, lambda: make_subset_batches_for_expert(0), loss_fn, router_mask_fn=None)
# gc = float(cosine(g_all, g_sub))
# print("GC (expert 0, routed subset vs all):", gc)


