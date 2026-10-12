import argparse
from pathlib import Path
from utils.seeding import noise_seed
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from accelerate import Accelerator
from accelerate.utils import set_seed, broadcast_object_list, gather_object

import h5py

from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import set_peft_model_state_dict

from custom_datasets.embeddings import EmbeddingDataset
from lora_adaption import initialize_lora_adapters
from moe_layers import copy_to_moe
from routing import LabelRouter, TextLabelRouter, MoCLERouterBasic
from utils.checkpoint_utils import load_sharded_state_dict
from minority_guidance import (
    compute_mg_velocity_correction,
    compute_sigma_p,
    mg_is_active,
)


# -------------------------
# HDF5 utilities (rank-0)
# -------------------------

def _ensure_h5_ready(h5_path: str, total: int, H: int, W: int):
    """
    Open/create an HDF5 file with datasets:
      - /images: uint8 [T, H, W, 3], chunked, compressed, maxshape=(None,H,W,3)
      - /present: uint8 [T], marks which positions have been written; maxshape=(None,)
      - /meta attrs: height, width
    Ensures shape[0] >= total (resizes if needed).
    Returns: h5py.File handle (opened in append mode).
    """
    # Do not use libver="latest": Helma's login Python currently writes HDF5
    # 2.0 layout messages that the BeyondFID container's HDF5 1.14 stack cannot
    # read. The default bounds keep generated H5s compatible with evaluation.
    f = h5py.File(h5_path, "a")

    if "images" not in f:
        # LZF compression instead of gzip-4: ~8-10x faster compression on the
        # rank-0 write path, modest size penalty (~50% larger than gzip-4).
        # The previous gzip-4 path took ~200ms per 1024x1024 chunk and
        # bottlenecked the whole gen loop on rank 0; lzf brings that to ~20ms.
        # lzf is universally read-supported by h5py, so older gzip-compressed
        # outputs remain readable by any downstream eval code.
        imgs = f.create_dataset(
            "images",
            shape=(total, H, W, 3),
            maxshape=(None, H, W, 3),
            dtype="uint8",
            chunks=(1, H, W, 3),
            compression="lzf",
            shuffle=True,
        )
        pres = f.create_dataset(
            "present",
            shape=(total,),
            maxshape=(None,),
            dtype="uint8",
            chunks=True,
        )
        pres[...] = 0
        # minimal meta
        f["meta"] = np.array([0], dtype=np.uint8)
        f["meta"].attrs["height"] = H
        f["meta"].attrs["width"] = W
    else:
        imgs = f["images"]
        pres = f["present"]
        # Ensure H/W compatibility by using what's already in file
        H0, W0 = imgs.shape[1], imgs.shape[2]
        if (H0, W0) != (H, W):
            # Use existing dataset's H/W for consistency
            H, W = H0, W0

        # Ensure capacity >= total
        if imgs.shape[0] < total:
            new_len = total
            imgs.resize((new_len, H, W, 3))
            pres.resize((new_len,))
            # newly allocated 'present' are zero by default, but make it explicit
            pres[imgs.shape[0]:new_len] = 0

    return f




def _write_batch_to_h5_rank0(f: h5py.File, positions, pil_images, H: int, W: int):
    """
    Writes a batch of PIL images at given global positions.
    Resizes images to (W,H) if necessary.
    """
    imgs = f["images"]
    pres = f["present"]

    # Dataset H/W as truth
    Hf, Wf = imgs.shape[1], imgs.shape[2]
    H, W = Hf, Wf

    # Ensure capacity (should already be sufficient)
    max_p = max(positions) if positions else -1
    if max_p >= imgs.shape[0]:
        new_len = max_p + 1
        imgs.resize((new_len, H, W, 3))
        pres.resize((new_len,))

    for p, im in zip(positions, pil_images):
        img = im if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im))
        if img.size != (W, H):
            img = img.resize((W, H), Image.BICUBIC)
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        imgs[p, ...] = arr
        pres[p] = 1


# -------------------------
# Argument parsing
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--metadata_csv", type=str)
    parser.add_argument("--prompt_key", type=str)
    parser.add_argument("--id_key", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--router_type", type=str, choices=["no", "set", "label", "textlabel", "mocle"], default="no")
    parser.add_argument("--bitmaps_path", type=str)
    parser.add_argument("--id2cui_path", type=str)
    parser.add_argument("--cui2sty_path", type=str)
    parser.add_argument("--medoid_path", type=str)
    parser.add_argument("--label_path", type=str)
    parser.add_argument("--moe_type", type=str, choices=["default", "delta", "mocle"], default="default")
    parser.add_argument("--moe_target_blocks", type=int, default=2)
    parser.add_argument("--ortho_only", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--bottleneck", type=int, default=None)
    parser.add_argument("--single_blocks", nargs='+', default=None)
    parser.add_argument("--dual_blocks", nargs='+', default=None)
    parser.add_argument("--n_single", type=int, default=None)
    parser.add_argument("--n_dual", type=int, default=None)
    parser.add_argument("--rank", type=int, default=4,
                        help=("The dimension of the LoRA update matrices."))
    parser.add_argument("--lora_alpha", type=int, default=4,
                        help="LoRA alpha to be used for additional scaling.")
    parser.add_argument("--lora_blocks", type=str, default=None,
                        help=("The transformer blocks to apply LoRA training on, comma separated "
                              "(e.g. '12,30')."))
    parser.add_argument("--precomputed_embeddings_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_images", type=int, required=True)
    parser.add_argument("--test_mode_expert", type=int, default=-1)

    # --- Minimal signature change: add a single optional argument for HDF5 path ---
    parser.add_argument("--h5_path", type=str, default=None,
                        help="If set, store all images in this single HDF5 file. "
                             "Existing *.jpg in --output are imported and deleted automatically.")

    # Optional seed for determinism equivalence
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--seed_mode", type=str, choices=["per_sample", "per_batch"], default="per_sample",
                        help="per_sample: every image gets its own noise seed (seed + position). "
                             "per_batch: all positions of one batch-sized block share "
                             "a seed (seed + position // batch_size).")
    parser.add_argument("--features_path", type=str, default=None)
    parser.add_argument("--centers_path", type=str, default=None)

    # Minority guidance (Um & Ye, ECCV 2024 — adapted for flow matching).
    parser.add_argument("--use_minority_guidance", action="store_true",
                        help="Enable self-guided minority sampling at inference time.")
    parser.add_argument("--mg_scale", type=float, default=0.4,
                        help="Weight w on the gradient correction. Paper default 0.4 (DDPM).")
    parser.add_argument("--mg_inter_rate", type=int, default=5,
                        help="Apply MG every Nth step (paper default 5).")
    parser.add_argument("--mg_p_ratio", type=float, default=0.2,
                        help="Additive offset in sigma space for the perturbation timestep "
                             "(sigma_p = clamp(sigma_t + mg_p_ratio, sigma_t*1.05, 0.999)).")
    parser.add_argument("--mg_t_min", type=float, default=0.2,
                        help="Skip MG when sigma < mg_t_min (Tweedie unstable at very low sigma).")
    parser.add_argument("--mg_t_max", type=float, default=0.95,
                        help="Skip MG when sigma > mg_t_max.")
    parser.add_argument("--mg_no_norm_grad", action="store_true",
                        help="Disable per-sample max-abs gradient normalization.")
    parser.add_argument("--mg_num_inference_steps", type=int, default=25,
                        help="Override num_inference_steps for the custom MG sampling loop.")
    parser.add_argument("--mg_guidance_scale", type=float, default=5.0,
                        help="Flux distilled guidance embedding value used during MG sampling.")

    return parser.parse_args()


def flux_minority_sample(
    pipe,
    prompt_embeds,
    pooled_prompt_embeds,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    generators,
    args,
):
    """Custom Flux denoising loop with self-guided minority correction.

    Replaces `pipe(prompt_embeds=..., ...)` for the MG path. Returns a list of
    PIL images of length `prompt_embeds.shape[0]`.
    """
    device = pipe._execution_device if hasattr(pipe, "_execution_device") else next(pipe.transformer.parameters()).device
    dtype = pipe.transformer.dtype

    batch_size = prompt_embeds.shape[0]
    num_channels_latents = pipe.transformer.config.in_channels // 4
    vae_scale_factor = pipe.vae_scale_factor if hasattr(pipe, "vae_scale_factor") else 8

    # Pack-aligned latent shape: matches FluxPipeline.prepare_latents.
    latent_h = 2 * (int(height) // (vae_scale_factor * 2))
    latent_w = 2 * (int(width) // (vae_scale_factor * 2))

    # Stack per-sample seeds via per-element generators (slow but matches the
    # reproducibility convention used elsewhere in this script).
    init_latents = []
    for g in generators:
        init_latents.append(
            torch.randn(
                (1, num_channels_latents, latent_h, latent_w),
                device=device, dtype=dtype, generator=g,
            )
        )
    latents = torch.cat(init_latents, dim=0)

    latents = pipe._pack_latents(
        latents, batch_size=batch_size,
        num_channels_latents=num_channels_latents,
        height=latent_h, width=latent_w,
    )

    latent_image_ids = pipe._prepare_latent_image_ids(
        batch_size, latent_h // 2, latent_w // 2, device, dtype,
    )

    # Build text-id tensor (pad-zeros, matching diffusers' behavior).
    text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)

    # Set timesteps with the same default Flux schedule the pipe uses.
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    scheduler_kwargs = {}
    if getattr(pipe.scheduler.config, "use_dynamic_shifting", False):
        image_seq_len = latents.shape[1]
        base_image_seq_len = pipe.scheduler.config.get("base_image_seq_len", 256)
        max_image_seq_len = pipe.scheduler.config.get("max_image_seq_len", 4096)
        base_shift = pipe.scheduler.config.get("base_shift", 0.5)
        max_shift = pipe.scheduler.config.get("max_shift", 1.15)
        m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
        b = base_shift - m * base_image_seq_len
        scheduler_kwargs["mu"] = image_seq_len * m + b
    pipe.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device, **scheduler_kwargs)
    timesteps = pipe.scheduler.timesteps

    transformer = pipe.transformer

    # Memory hygiene for the MG path: at 1024x1024 the MG forward pass would
    # otherwise hold activations through all 57 Flux blocks under autograd.
    # (a) Freeze parameters so we don't accumulate optimizer/grad metadata.
    #     We only need the gradient w.r.t. x_t, never w.r.t. params.
    # (b) Enable per-block gradient checkpointing. Flux's checkpointing path
    #     only triggers when torch.is_grad_enabled() is True, so non-MG
    #     forwards (wrapped in torch.no_grad()) are unaffected.
    if args.use_minority_guidance:
        for p in transformer.parameters():
            p.requires_grad_(False)
        try:
            transformer.enable_gradient_checkpointing()
        except Exception:
            pass

    # Distilled guidance embedding (Flux-dev convention).
    if transformer.config.guidance_embeds:
        guidance = torch.full((batch_size,), guidance_scale, device=device, dtype=dtype)
    else:
        guidance = None

    def _velocity_fn(x_packed, t_norm):
        return transformer(
            hidden_states=x_packed,
            timestep=t_norm,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

    # Use the first generator to drive MG re-noising; per-sample reproducibility
    # is preserved because the MG noise tensor's deterministic shape and the
    # outer per-sample x_t already encode the per-sample state.
    mg_gen = generators[0] if len(generators) > 0 else None

    for i, t in enumerate(timesteps):
        sigma_t = float(pipe.scheduler.sigmas[i].item())
        t_norm = (t.float() / 1000.0).expand(batch_size).to(latents.dtype)

        with torch.no_grad():
            v = _velocity_fn(latents, t_norm)

        if mg_is_active(i, sigma_t, args):
            sigma_p = compute_sigma_p(sigma_t, args.mg_p_ratio)
            v_corr = compute_mg_velocity_correction(
                velocity_fn=_velocity_fn,
                x_t_packed=latents,
                t_norm=t_norm,
                sigma_t=sigma_t,
                sigma_p=sigma_p,
                eps_generator=mg_gen,
                mg_scale=args.mg_scale,
                use_normed_grad=not args.mg_no_norm_grad,
            )
            v = v + v_corr

        latents = pipe.scheduler.step(v, t, latents, return_dict=False)[0]

    # Decode to images.
    latents = pipe._unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.no_grad():
        decoded = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
    images = pipe.image_processor.postprocess(decoded, output_type="pil")
    return images


def iter_indices_images(gathered):
    # sanity check
    if len(gathered) % 2 != 0:
        raise ValueError(f"Expected even-length alternating list, got {len(gathered)}")

    def _unwrap(x):
        # some gather APIs wrap payloads like [payload]
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return x[0]
        return x

    for i in range(0, len(gathered), 2):
        idxs = _unwrap(gathered[i])      # e.g. [0, 8, 16, ...]
        imgs = _unwrap(gathered[i + 1])  # e.g. [PIL.Image, PIL.Image, ...]
        if not isinstance(idxs, (list, tuple)): idxs = [idxs]  # single-item payload over-unwrapped to scalar
        if not isinstance(imgs, (list, tuple)): imgs = [imgs]
        if len(idxs) != len(imgs):
            raise ValueError(f"length mismatch: {len(idxs)} indices vs {len(imgs)} images at pair {i//2}")
        yield idxs, imgs


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    # Deterministic runs (same base on all ranks; per-position seeds ensure uniqueness)
    set_seed(args.seed)

    out_dir = Path(f"{args.output}")
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load CSV / prompts / ids
    df = pd.read_csv(args.metadata_csv)
    if args.precomputed_embeddings_path is None:
        prompts = df[args.prompt_key].tolist()
    ids = df[args.id_key].tolist()
    N = len(ids)

    embeddings = None
    if args.precomputed_embeddings_path is not None:
        embeddings = EmbeddingDataset(args.precomputed_embeddings_path)

    # Load pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    pipe.to(device)

    # Router setup
    router = None
    if args.router_type != "no":
        if args.router_type == "label":
            router = LabelRouter(args.label_path, ids)
            router.test_mode(args.test_mode_expert)
        elif args.router_type == "textlabel":
            embeds = embeddings.load_ppe()
            router = TextLabelRouter(label_path=args.label_path,
                                     ids=ids,
                                     centers_path=args.centers_path,
                                     embeddings=embeds)
        elif args.router_type == "mocle":
            if args.label_path is not None:
                router = MoCLERouterBasic(
                    cluster_assign_path=args.label_path,
                    pooled_embeddings=None,
                    cluster_centers_path=args.centers_path,
                    ids=ids,
                )
            else:
                if embeddings is None:
                    raise ValueError(
                        "router_type='mocle' in centroid mode requires --precomputed_embeddings_path "
                        "or provide --label_path for lookup mode."
                    )
                embeds = embeddings.load_ppe()
                router = MoCLERouterBasic(
                    cluster_assign_path=None,
                    pooled_embeddings=embeds,
                    cluster_centers_path=args.centers_path,
                    ids=ids,
                )


    # Transformer setup (LoRA/MoE) and checkpoint loading
    local_dir = args.checkpoint
    if router is None:
        new_tf = FluxTransformer2DModel.from_pretrained(
            local_dir, subfolder="transformer", use_safetensors=True,
            torch_dtype=pipe.transformer.dtype, local_files_only=True
        )
    elif args.use_lora:
        transformer_ = new_tf = copy_to_moe(
            pipe.transformer, router, args.moe_type, n=args.moe_target_blocks,
            target_mixture=None, num_experts=getattr(router, "num_experts", 1),
            ortho_only=args.ortho_only,
            n_dual=args.n_dual,
            n_single=args.n_single,
            dual_blocks=args.dual_blocks,
            single_blocks=args.single_blocks,
            bottleneck=args.bottleneck
        )
        transformer_, _ = initialize_lora_adapters(args, transformer_)
        lora_state_dict = FluxPipeline.lora_state_dict(local_dir)
        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                accelerator.print(
                    f"Loading adapter weights led to unexpected keys not found in the model: {unexpected_keys}"
                )
    else:
        new_tf = copy_to_moe(
            pipe.transformer, router, args.moe_type, n=args.moe_target_blocks,
            target_mixture=None, num_experts=getattr(router, "num_experts", 1),
            ortho_only=args.ortho_only,
            n_dual=args.n_dual,
            n_single=args.n_single,
            dual_blocks=args.dual_blocks,
            single_blocks=args.single_blocks,
            bottleneck=args.bottleneck
        )
        state_dict = new_tf.state_dict()
        load_state_dict = load_sharded_state_dict(local_dir, subfolder="transformer")
        for k, v in load_state_dict.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                state_dict[k] = v
            else:
                raise ValueError(f"Wrong dimensions for state dict key {k}: "
                                 f"expected {tuple(state_dict[k].shape) if k in state_dict else 'N/A'} vs {tuple(v.shape)}")
        new_tf.load_state_dict(state_dict)

    pipe.register_modules(transformer=new_tf.to(pipe.dtype))
    pipe.to(device)

    # Fixed generation size (match original script)
    H = 1024
    W = 1024

    # HDF5 file (rank 0 writer)
    f = None
    if args.h5_path and rank == 0:
        f = _ensure_h5_ready(args.h5_path, total=args.num_images, H=H, W=W)
    accelerator.wait_for_everyone()

    # Determine resume index start_global based on existing data
    if rank == 0:
        if args.h5_path:
            # Use 'present' to find last
            pres = f["present"][...] if f is not None and "present" in f else None
            if pres is not None and np.any(pres == 1):
                last_global = int(np.max(np.where(pres == 1)[0]))
            else:
                last_global = -1
        else:
            # Legacy: scan filesystem
            last_global = -1
            for p in out_dir.glob("*.jpg"):
                stem = p.stem
                if "_" in stem:
                    a, b = stem.split("_", 1)
                    if a.isdigit() and b.isdigit():
                        total_idx = int(a) * N + int(b)
                        last_global = max(last_global, total_idx)
                elif stem.isdigit():
                    last_global = max(last_global, int(stem))
        start_global = last_global + 1
    else:
        start_global = None

    # Broadcast start_global to everyone (Accelerate keyword per your version)
    obj_list = [start_global]
    broadcast_object_list(obj_list, from_process=0)
    start_global = obj_list[0]
    if rank == 0:
        accelerator.print(f"Starting from global index {start_global}")

    total = args.num_images
    bs = args.batch_size if args.batch_size else 1


    # Noise seed per global position p (see --seed_mode).
    def seed_for_pos(p: int) -> int:
        # per_batch: blocks of `batch_size` consecutive positions share one seed; per_sample: one seed per image.
        return noise_seed(p, args.seed, args.seed_mode, start=start_global, batch_size=bs)


    # Shard global positions across ranks by congruence class
    all_positions = list(range(start_global, total))
    my_positions = [p for p in all_positions if (p - start_global) % world_size == rank]

    for i in range(0, len(my_positions), bs):
        pos_batch = my_positions[i: i + bs]
        if not pos_batch:
            continue

        batch_indices = [p % N for p in pos_batch]
        batch_cycles = [p // N for p in pos_batch]

        # Router per local batch
        if router is not None:
            router.set_current_ids([ids[j] for j in batch_indices])
            router.set_current_expert_indices(router.get_experts(), device=device)

        # Build per-sample generators (preserve original seed equivalence)
        gens = []
        for p in pos_batch:
            g = torch.Generator(device=device)
            g.manual_seed(seed_for_pos(p))
            gens.append(g)

        if args.use_minority_guidance:
            # Custom Flux denoising loop with self-guided minority correction.
            # MG requires per-step grad through the model, so this path cannot
            # use pipe(...) directly. Encode prompts here when not precomputed.
            if embeddings is None:
                with torch.no_grad():
                    pe, ppe, _ = pipe.encode_prompt(
                        prompt=[prompts[j] for j in batch_indices],
                        prompt_2=None,
                        device=device,
                    )
                prompt_embeds = pe.to(device=device, dtype=pipe.dtype)
                pooled_prompt_embeds = ppe.to(device=device, dtype=pipe.dtype)
            else:
                outs = [embeddings[j] for j in batch_indices]
                prompt_embeds = torch.stack([o.prompt_embeds for o in outs]).to(device=device, dtype=pipe.dtype)
                pooled_prompt_embeds = torch.stack([o.pooled_prompt_embeds for o in outs]).to(device=device, dtype=pipe.dtype)
            local_images = flux_minority_sample(
                pipe=pipe,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                height=H, width=W,
                num_inference_steps=args.mg_num_inference_steps,
                guidance_scale=args.mg_guidance_scale,
                generators=gens,
                args=args,
            )
        else:
            with torch.no_grad():
                if embeddings is None:
                    local_images = pipe(
                        prompt=[prompts[j] for j in batch_indices],
                        num_inference_steps=25,
                        guidance_scale=5,
                        height=H, width=W,
                        generator=gens,
                    ).images
                else:
                    outs = [embeddings[j] for j in batch_indices]
                    prompt_embeds = torch.stack([o.prompt_embeds for o in outs]).to(device=device, dtype=pipe.dtype)
                    pooled_prompt_embeds = torch.stack([o.pooled_prompt_embeds for o in outs]).to(device=device, dtype=pipe.dtype)
                    local_images = pipe(
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        num_inference_steps=25,
                        guidance_scale=5,
                        height=H, width=W,
                        generator=gens,
                    ).images

        # Gather (positions, images) to rank 0 for writing
        # NOTE: gather_object returns the list on all ranks; we only act on rank 0.
        gathered = gather_object((pos_batch, local_images))

        if rank == 0:
            # Flatten gathered batches from all ranks
            positions_all, images_all = [], []
            for positions_r, images_r in iter_indices_images(gathered):
                positions_all.extend(positions_r)
                images_all.extend(images_r)

            if args.h5_path:
                _write_batch_to_h5_rank0(f, positions_all, images_all, H, W)
                accelerator.print(f"[rank 0] wrote {len(positions_all)} images to HDF5")
            else:
                # Fallback: write files (not recommended on HPC due to inode limits)
                for p, im in zip(positions_all, images_all):
                    cyc, idx = divmod(p, N)
                    save_path = out_dir / f"{cyc}_{idx}.jpg"
                    im.save(save_path.as_posix())
                    accelerator.print(f"[rank 0] Saved {save_path}")

    # Finalize HDF5 (rank 0)
    if rank == 0 and args.h5_path and f is not None:
        f.flush()
        try:
            # Best-effort fsync of underlying file descriptor
            os.fsync(f.fid.get_vfd_handle())
        except Exception:
            pass
        f.close()

    accelerator.print(f"[rank {rank}] Done.")


if __name__ == "__main__":
    main()
