import os
from contextlib import nullcontext

import numpy as np
import torch
import wandb
from diffusers.training_utils import free_memory
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from transformers import PretrainedConfig


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = ""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "flux",
        "flux-diffusers",
        "template:sd",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))



def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    logger,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    # pre-calculate  prompt embeds, pooled prompt embeds, text ids because t5 does not support autocast
    # Precompute with the pipeline's own encoder so shapes/dtypes match:
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            args.validation_prompt,
            prompt_2=args.validation_prompt
        )

    images = []
    for _ in range(args.num_validation_images):
        with autocast_ctx:
            out = pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=6.0,
                num_inference_steps=50,
                height=1024, width=1024,
                generator=generator
            )
            images.append(out.images[0])

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision,token=os.environ["HUGGINGFACE_HUB_TOKEN"]
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None
):
    if prompt is not None:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
    else:
        batch_size = len(text_input_ids)
    if tokenizer is not None and text_input_ids is None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    if not torch.is_tensor(text_input_ids):
        if isinstance(text_input_ids, list):
            text_input_ids = torch.stack(text_input_ids)
        else:
            text_input_ids = torch.tensor(text_input_ids)


    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length=256, device="cuda"):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            text_ids = text_ids.to(device)
        return prompt_embeds, pooled_prompt_embeds, text_ids


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids

def load_text_encoders(args, class_one, class_two):
    token = os.environ["HUGGINGFACE_HUB_TOKEN"]
    common_kwargs = {
        "revision": args.revision,
        "variant": args.variant,
        "token": token,
    }

    def _load_with_retry(model_cls, subfolder: str):
        try:
            return model_cls.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder=subfolder,
                **common_kwargs,
            )
        except Exception as first_exc:
            # Shared-cache / partial-download issues can cause transient config-weight mismatch.
            # Retry once with forced re-download before failing hard.
            print(
                f"[WARN] load_text_encoders: first load failed for subfolder={subfolder}: "
                f"{first_exc}. Retrying with force_download=True."
            )
            return model_cls.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder=subfolder,
                force_download=True,
                **common_kwargs,
            )

    text_encoder_one = _load_with_retry(class_one, "text_encoder")
    text_encoder_two = _load_with_retry(class_two, "text_encoder_2")
    return text_encoder_one, text_encoder_two


def moe_weight_orthogonality(experts, selector=lambda e: e.head.weight, eps=1e-8):
    """
    experts : iterable of expert modules
    selector: fn(expert) -> weight tensor to use (e.g., final linear weight)
    returns: scalar penalty encouraging inter-expert weight orthogonality
    """
    vecs = []
    for e in experts:
        W = selector(e)                 # e.g., (D_out, D_in)
        w = W.reshape(-1)               # (P,)
        w = w / (w.norm() + eps)        # scale-invariant
        vecs.append(w)

    if len(vecs) <= 1:
        return vecs[0].new_tensor(0.0) if vecs else torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')

    V = torch.stack(vecs, dim=0)        # (E, P)
    G = V @ V.t()                        # (E, E) cosine Gram across experts
    E = G.size(0)
    I = torch.eye(E, device=G.device, dtype=G.dtype)
    offdiag = G - I                      # remove diagonal (=1)
    # mean squared off-diagonal (bounded in [0, 1])
    pen = (offdiag.pow(2).sum()) / (E * (E - 1))
    return pen
