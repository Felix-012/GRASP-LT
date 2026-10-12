"""
Self-guided minority sampling adapted to flow matching (Flux).

Reference: Um & Ye, "Self-Guided Generation of Minority Samples Using Diffusion
Models", ECCV 2024 (arXiv:2407.11555). The original method targets DDPM with
ε-prediction; we adapt it to Flux's flow-matching velocity prediction.

Adaptation summary
------------------
DDPM Tweedie:    x0_hat = (x_t - sqrt(1-alpha_bar)*eps) / sqrt(alpha_bar)
Flux flow-match: x0_hat = x_t - sigma_t * v_theta(x_t, t, c)

Per MG-active step we:
  1. Run the model with grad enabled to obtain v_outer = v_theta(x_t, t, c)
     and the differentiable Tweedie estimate x0_outer = x_t - sigma_t * v_outer.
  2. Re-noise x0_outer (with stop-gradient) to a perturbed sigma_p > sigma_t:
       x_tp = (1 - sigma_p) * sg(x0_outer) + sigma_p * eps'.
  3. Re-denoise (no grad): v_inner = v_theta(x_tp, t_p, c),
     x0_inner = x_tp - sigma_p * v_inner.
  4. L = || x0_outer - sg(x0_inner) ||_2^2.
  5. g = grad(L, x_t).
  6. Velocity correction: v_corrected = v - (mg_scale / sigma_t) * g
     (so the Euler step  x_{t-1} = x_t + dt * v   gains an additive term
     proportional to +mg_scale * (-dt / sigma_t) * g, i.e., the additive-mean
     correction the original DDPM paper applies to the posterior mean).

We skip MG outside [mg_t_min, mg_t_max] to avoid unstable Tweedie at very low
sigma and to avoid wasted compute at the noise-dominated end.

Compatibility with Flux's distilled CFG: Flux dev uses a `guidance` embedding
(scalar conditioning) rather than dual-pass CFG, so MG composes by simply
adding the correction term to the already-conditioned velocity.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def mg_is_active(step_idx: int, sigma_t: float, args) -> bool:
    """Decide whether to apply MG at this denoising step."""
    if not args.use_minority_guidance:
        return False
    if step_idx % max(1, args.mg_inter_rate) != 0:
        return False
    if sigma_t < args.mg_t_min or sigma_t > args.mg_t_max:
        return False
    return True


def _normalize_grad(g: torch.Tensor) -> torch.Tensor:
    """Per-sample max-abs normalization (matches `use_normed_grad=True` in Um et al.)."""
    flat = g.flatten(1)
    max_abs = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / max_abs).reshape_as(g)


def compute_mg_velocity_correction(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_t_packed: torch.Tensor,
    t_norm: torch.Tensor,
    sigma_t: float,
    sigma_p: float,
    eps_generator: Optional[torch.Generator],
    mg_scale: float,
    use_normed_grad: bool = True,
) -> torch.Tensor:
    """Compute the velocity correction term -(mg_scale/sigma_t)*g from
    self-guided minority sampling.

    Args
    ----
    velocity_fn: takes (x_packed, t_norm) and returns the model's velocity
                 prediction in packed form. Must be differentiable w.r.t. x.
    x_t_packed:  current latents (packed) at sigma_t. Detached when entering.
    t_norm:      timestep in [0, 1] for the model's expected scaling, shape [B].
    sigma_t:     scalar current sigma.
    sigma_p:     scalar perturbation sigma (> sigma_t).
    eps_generator: torch generator for reproducible re-noising.
    mg_scale:    weight w on the gradient correction.
    use_normed_grad: per-sample max-abs normalize the gradient (paper default).

    Returns
    -------
    v_correction: tensor with the same shape as x_t_packed; ADD this to the
                  baseline velocity prediction before the Euler step.
    """
    # Outer pass with autograd; the caller is responsible for enabling
    # gradient checkpointing on the transformer to keep peak activation memory
    # bounded. Only x_t carries grad: transformer params are frozen by the
    # caller, so the autograd graph holds checkpointed activations only.
    # Outer (grad) pass and inner (no-grad) pass. PyTorch's allocator with
    # expandable_segments=True (set via PYTORCH_CUDA_ALLOC_CONF in the launch
    # script) handles fragmentation; we deliberately do NOT call
    # torch.cuda.empty_cache() here, since each call would synchronize the
    # device and add 1-5s of overhead per MG step at near-full memory.
    x_t_grad = x_t_packed.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        v_outer = velocity_fn(x_t_grad, t_norm)
        x0_outer = x_t_grad - sigma_t * v_outer
        # Inner Tweedie target under no_grad inside the enable_grad scope so
        # the inner forward does not build any autograd graph.
        with torch.no_grad():
            x0_outer_sg = x0_outer.detach()
            eps_prime = torch.randn(
                x0_outer_sg.shape,
                device=x0_outer_sg.device,
                dtype=x0_outer_sg.dtype,
                generator=eps_generator,
            )
            x_tp = (1.0 - sigma_p) * x0_outer_sg + sigma_p * eps_prime
            t_p_norm = torch.full_like(t_norm, sigma_p)
            v_inner = velocity_fn(x_tp, t_p_norm)
            x0_inner = (x_tp - sigma_p * v_inner).detach()
        diff = x0_outer.float() - x0_inner.float()
        loss = diff.pow(2).flatten(1).sum(dim=1).mean()
        g = torch.autograd.grad(loss, x_t_grad, retain_graph=False, create_graph=False)[0]

    g = g.detach()
    if use_normed_grad:
        g = _normalize_grad(g)

    # Velocity correction: the Euler step is x_{t-1} = x_t + dt * v_tilde
    # with dt = sigma_next - sigma_t < 0. To inject +mg_scale * g into the
    # posterior mean, we set v_tilde = v - (mg_scale / sigma_t) * g.
    v_correction = -(mg_scale / max(sigma_t, 1e-6)) * g
    return v_correction.to(x_t_packed.dtype)


def compute_sigma_p(sigma_t: float, p_ratio: float) -> float:
    """Map sigma_t to a perturbation sigma_p > sigma_t.

    p_ratio behaves as an additive offset in normalized sigma space, clamped to
    [sigma_t, 0.999]. We chose this form (rather than the paper's
    `t_p = p_ratio * T_max`) because Flux's continuous t in [0, 1] does not
    have a fixed train horizon to anchor the multiplicative form.
    """
    return float(min(0.999, max(sigma_t + p_ratio, sigma_t * 1.05)))
