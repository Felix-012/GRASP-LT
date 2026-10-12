#!/usr/bin/env python
# coding=utf-8

import copy
import itertools
import logging
import os

from gradient_consistency_analysis.gca_train_helpers import (
    append_label_pair_rows,
    append_projected_alignment_rows,
    compute_label_pair_projected_cosine_for_step,
    compute_projected_alignment_for_step,
    extract_labels_str_from_batch,
    init_label_pair_csv,
    init_projected_alignment_csv,
    plot_projected_alignment_csv,
    run_gradient_consistency_analysis,
)

import re
import shutil
import uuid
from pathlib import Path

import diffusers
import h5py
import math
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel, )
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, T5TokenizerFast, T5EncoderModel, CLIPTextModel

from analyzer import ExpertAnalyzer
from custom_datasets.imagenet_data import get_dataloader
from custom_datasets.mapping import ToIndex
from custom_datasets.mimic import encode_and_save_text_embeddings
from lora_adaption import initialize_lora_adapters, get_lora_save_model_hook, get_lora_load_model_hook, \
    get_lora_params_to_optimize
from moe_layers import copy_to_moe, _copy_transfer, DeltaFeedForwardMOE, FeedForwardMOE, DeltaProjOutFFMOE, MoCLELoRAFF
from routing import Router, TextClusterRouter, ClusterRouter, GradientRouter, LabelRouter, FakeLabelRouter, \
    TextLabelRouter, MoCLERouterBasic
from target_mixture.target_mixture import TargetMixture
from utils.checkpoint_utils import load_sharded_state_dict
from utils.train_parser import parse_args
from utils.train_utils import import_model_class_from_model_name_or_path, encode_prompt, tokenize_prompt, \
    log_validation, save_model_card, compute_text_embeddings, load_text_encoders
from ccua import (
    FeatureGrabber,
    broadcast_null_embeddings,
    compute_alignment_loss,
    compute_unsupervised_contrastive_loss,
    load_null_embeddings,
)

if is_wandb_available():
    pass

logger = get_logger(__name__)



def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    find_unused = True
    static_graph = False
    if args.router_type in ["no", "fake"] or args.batch_buffer_size >= 3:
        find_unused = False
        static_graph=True
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused,
                                           gradient_as_bucket_view=True,
                                           static_graph=static_graph)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to in (None, "none") else args.report_to,  # --report_to=none: no tracker
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        token=os.environ["HUGGINGFACE_HUB_TOKEN"]
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        token=os.environ["HUGGINGFACE_HUB_TOKEN"]
    )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", token=os.environ["HUGGINGFACE_HUB_TOKEN"]
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(args, text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        token=os.environ["HUGGINGFACE_HUB_TOKEN"]
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        token=os.environ["HUGGINGFACE_HUB_TOKEN"]
    )

    train_dataloader, train_dataset = get_dataloader(args,
                                                     split="train",
                                                     base_path=args.instance_data_dir,
                                                     batch_size=args.train_batch_size,
                                                     name=args.dataset_name)

    router = None

    if args.router_type == "gradient":
        router = GradientRouter()
    elif args.router_type == "cluster":
        router = ClusterRouter(768, 64)
    elif args.router_type == "clip-cluster":
        router = TextClusterRouter(768, 64)
        if os.path.exists(args.precomputed_embeddings_path):
            with h5py.File(args.precomputed_embeddings_path, 'r') as f:
                logger.info("loading embeddings...")
                text_embeddings = torch.from_numpy(np.asarray(f['embeddings']))
                logger.info("embeddings loaded.")
        else:
            logger.info("precomputing embeddings...")
            text_embeddings = encode_and_save_text_embeddings(text_encoder_one,
                                                              tokenizer_one,
                                                              train_dataset,
                                                              args.precomputed_embeddings_path)
            logger.info("embeddings precomputed."
                        "")
        router.set_routing(text_embeddings)
    elif args.router_type == "label":
        router = LabelRouter(label_path=args.label_path,
                             ids=train_dataset.ids,
                             permute_seed=(args.router_label_permute_seed if args.router_label_permute_seed >= 0 else None))
        router.test_mode(args.test_mode_expert)
    elif args.router_type == "fake":
        router = FakeLabelRouter(label_path=args.label_path,
                             ids=train_dataset.ids,
                             permute_seed=(args.router_label_permute_seed if args.router_label_permute_seed >= 0 else None))
    elif args.router_type == "textlabel":
        router = TextLabelRouter(label_path=args.label_path,
                                 ids=train_dataset.ids,
                                 features_path=args.features_path,
                                 centers_path=args.centers_path)
    elif args.router_type == "mocle":
        router = MoCLERouterBasic(cluster_assign_path=args.label_path,
                                  cluster_centers_path=args.centers_path,
                                  ids=train_dataset.ids,
                                  static_init=args.mocle_grasp_init)
    elif args.router_type != "no":
        logger.warning("Unrecognized router type, falling back to default router (always routes to index 0)")
        router = Router()

    if args.use_target_mixture:
        target_mixture = TargetMixture()
        if args.reference_batch_size > 0:
            target_mixture.sample_reference_batches_by_labels((train_dataset.latents, train_dataset.embeddings),
                                                              router.labels,
                                                              args.reference_batch_size,
                                                              vae.config.scaling_factor,
                                                              vae.config.shift_factor)
    else:
        target_mixture = None


    if router is not None and args.router_type != "fake":
        transformer = copy_to_moe(transformer, router, args.moe_type,
                                  n=args.moe_target_blocks,
                                  target_mixture=target_mixture,
                                  num_experts=router.num_experts,
                                  copy_weights=args.copy_weights,
                                  ortho_only=args.ortho_only,
                                  n_dual=args.n_dual,
                                  n_single=args.n_single,
                                  dual_blocks=args.dual_blocks,
                                  single_blocks=args.single_blocks,
                                  bottleneck=args.bottleneck
                                  )
        root = Path.cwd().resolve()
        run_id = uuid.uuid4().hex[:8]
        def slugify(x) -> str:
            # Safe for filenames
            return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(x)).strip("-")
        reports_dir = root / "reports"
        fname = (
            f"moe={slugify(args.moe_type)}_"
            f"router={slugify(args.router_type)}_"
            f"targetmix={int(bool(args.use_target_mixture))}_"
            f"buf={slugify(args.batch_buffer_size)}_"
            f"freeze={int(bool(args.freeze))}_"
            f"run-{run_id}.log"
        )
        report_path = reports_dir / fname

        analyzer = ExpertAnalyzer(num_experts=router.num_experts,
                                  logger=logger,
                                  report_path=report_path,
                                  run_id=run_id)

    if args.use_lora:
        transformer.requires_grad_(False)
    elif args.freeze:
        transformer.requires_grad_(False)

        for m in transformer.modules():
            if isinstance(m, DeltaFeedForwardMOE) or isinstance(m, MoCLELoRAFF):
                # make the delta expert trainable
                for p in m.parameters():
                    p.requires_grad = True
                if hasattr(m, "base") and m.base is not None:
                    m.base.requires_grad_(False)

            elif isinstance(m, FeedForwardMOE):
                for p in m.parameters():
                    p.requires_grad = True
            elif isinstance(m, DeltaProjOutFFMOE):
                for p in m.parameters():
                    p.requires_grad = True
                if hasattr(m, "base") and m.base is not None:
                    m.base.requires_grad_(False)
    else:
        transformer.requires_grad_(True)
    vae.requires_grad_(False)
    if args.train_text_encoder:
        text_encoder_one.requires_grad_(True)
        text_encoder_two.requires_grad_(True)
    else:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)


    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    if args.precomputed_latents_path is None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    if args.precomputed_embeddings_path is None:
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    else:
        text_encoder_one.to(dtype=weight_dtype)
        text_encoder_two.to(dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.use_lora:
        transformer, transformer_lora_config = initialize_lora_adapters(args, transformer)

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), FluxTransformer2DModel):
                    if args.router_type=="mocle":
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"),
                                                            safe_serialization=False)
                    else:
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"))
                elif isinstance(unwrap_model(model), (CLIPTextModel, T5EncoderModel)):
                    if isinstance(unwrap_model(model), CLIPTextModel):
                        hidden_size = unwrap_model(model).config.hidden_size
                        if hidden_size == 768:
                            unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder"))
                        elif hidden_size == 1280:
                            unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_2"))
                    else:
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_3"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(unwrap_model(model), FluxTransformer2DModel):
                load_model = None
                if router is not None and args.router_type != "fake":
                    model = copy_to_moe(model, router, args.moe_type,
                                        n=args.moe_target_blocks,
                                        target_mixture=None, num_experts=router.num_experts,
                                        n_dual=args.n_dual,
                                        n_single=args.n_single,
                                        dual_blocks=args.dual_blocks,
                                        single_blocks=args.single_blocks,
                                        bottleneck=args.bottleneck
                                        )
                    state_dict = model.state_dict()
                    load_state_dict = load_sharded_state_dict(input_dir, subfolder="transformer")
                    for k, v in load_state_dict.items():
                        if k in state_dict and state_dict[k].shape == v.shape:
                            state_dict[k] = v
                        else:
                            raise ValueError("Wrong dimensions for state dict")
                    model.load_state_dict(state_dict)
                else:
                    load_model = FluxTransformer2DModel.from_pretrained(input_dir, subfolder="transformer",
                                                                        token=os.environ["HUGGINGFACE_HUB_TOKEN"])
                    model.load_state_dict(load_model.state_dict())
            elif isinstance(unwrap_model(model), (CLIPTextModel, T5EncoderModel)):
                try:
                    load_model = CLIPTextModel.from_pretrained(input_dir, subfolder="text_encoder",
                                                                             token=os.environ["HUGGINGFACE_HUB_TOKEN"])
                    model.load_state_dict(load_model.state_dict())
                except Exception:
                    try:
                        load_model = CLIPTextModel.from_pretrained(input_dir, subfolder="text_encoder_2",
                                                                                 token=os.environ[
                                                                                     "HUGGINGFACE_HUB_TOKEN"])
                        model.load_state_dict(load_model.state_dict())
                    except Exception:
                        try:
                            load_model = T5EncoderModel.from_pretrained(input_dir, subfolder="text_encoder_3",
                                                                        token=os.environ["HUGGINGFACE_HUB_TOKEN"])
                            model.load_state_dict(load_model.state_dict())
                        except Exception:
                            raise ValueError(f"Couldn't load the model of type: ({type(model)}).")
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")
            if load_model is not None:
                del load_model

    if not args.use_lora:
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    else:
        accelerator.register_save_state_pre_hook(get_lora_save_model_hook(args,
                                                                          accelerator,
                                                                          transformer,
                                                                          text_encoder_one))
        accelerator.register_load_state_pre_hook(get_lora_load_model_hook(args,
                                                                          logger,
                                                                          accelerator,
                                                                          transformer,
                                                                          text_encoder_one,
                                                                          text_encoder_two,
                                                                          transformer_lora_config,
                                                                          text_encoder_cls_one,
                                                                          text_encoder_cls_two))
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_lora:
        params_to_optimize,transformer_lora_parameters, = get_lora_params_to_optimize(args, transformer)

    else:
        transformer_parameters_with_lr = {"params": transformer.parameters(), "lr": args.learning_rate}
        if args.train_text_encoder:
            # different learning rate for text encoder and unet
            text_parameters_one_with_lr = {
                "params": text_encoder_one.parameters(),
                "weight_decay": args.adam_weight_decay_text_encoder,
                "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
            }
            text_parameters_two_with_lr = {
                "params": text_encoder_two.parameters(),
                "weight_decay": args.adam_weight_decay_text_encoder,
                "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
            }

            params_to_optimize = [
                transformer_parameters_with_lr,
                text_parameters_one_with_lr,
                text_parameters_two_with_lr,
            ]
        else:
            params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

    if args.router_type == "clip-cluster":
        router.create_lookup(train_dataset.get_indices())

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            text_encoder_two,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            text_encoder_two,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        (transformer,
         optimizer,
         train_dataloader,
         lr_scheduler) = accelerator.prepare(
            transformer,
            optimizer,
            train_dataloader,
            lr_scheduler
        )





    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "flux"
        accelerator.init_trackers(tracker_name)

    # Train!
    total_batch_size = (args.train_batch_size + args.batch_buffer_size) * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size + args.batch_buffer_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # -------------------------
    # Resume (with special-case for gc_mode)
    # -------------------------
    if args.resume_from_checkpoint:
        gc_mode = bool(getattr(args, "gc_mode", False))

        if args.resume_from_checkpoint != "latest":
            if gc_mode:
                # In GC mode, use the provided path directly (can be absolute or relative).
                ckpt_dir = Path(args.resume_from_checkpoint).expanduser()
                ckpt_dir = ckpt_dir.resolve() if ckpt_dir.exists() else ckpt_dir  # resolve when possible
                path = ckpt_dir.name  # for logging + global_step parsing
            else:
                # Non-GC mode keeps the old behavior: treat it as a directory name under output_dir.
                path = os.path.basename(args.resume_from_checkpoint)
                ckpt_dir = Path(args.output_dir) / path
        else:
            # "latest" still means: search under output_dir (same behavior as before)
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            ckpt_dir = (Path(args.output_dir) / path) if path is not None else None

        # Existence check
        if ckpt_dir is None or not Path(ckpt_dir).exists():
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {ckpt_dir}")
            try:
                accelerator.load_state(str(ckpt_dir))
            except ValueError as ve:
                if args.freeze:
                    logger.info(
                        "Failed to load checkpoint as MOE; converting a base transformer to MOE and loading its weights..."
                    )

                    target_model = accelerator.unwrap_model(transformer)

                    with accelerator.main_process_first():
                        conv_dir = Path(args.output_dir) / "_auto_converted_from_base"
                        conv_dir.mkdir(parents=True, exist_ok=True)
                        # Make the file name specific to the source checkpoint to avoid collisions
                        conv_path = conv_dir / f"moe_transformer_from_{str(path).replace('/', '_')}.pt"

                        if accelerator.is_main_process and not conv_path.exists():
                            base = FluxTransformer2DModel.from_pretrained(
                                str(ckpt_dir),
                                subfolder="transformer",
                                token=os.environ.get("HUGGINGFACE_HUB_TOKEN", None),
                            )
                            base = copy_to_moe(
                                base,
                                router,
                                args.moe_type,
                                n=args.moe_target_blocks,
                                target_mixture=None,
                                num_experts=router.num_experts,
                                n_dual=args.n_dual,
                                n_single=args.n_single,
                                dual_blocks=args.dual_blocks,
                                single_blocks=args.single_blocks,
                                bottleneck=args.bottleneck,
                            )
                            torch.save(base.state_dict(), conv_path)

                    accelerator.wait_for_everyone()
                    state_dict = torch.load(conv_path, map_location="cpu")

                    missing, unexpected = target_model.load_state_dict(state_dict, strict=False)

                    accelerator.print(
                        f"Loaded converted base weights from {conv_path.name}. "
                        f"Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}"
                    )

                    del state_dict
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise ve

            # Parse step from the checkpoint folder name (e.g., "checkpoint-1234")
            # NOTE: if you pass a custom dir that doesn't match this naming, this will fail.
            global_step = int(str(path).split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # -------------------------
    # Gradient consistency mode (exit after analysis)
    # -------------------------
    if getattr(args, "gc_mode", False):
        if accelerator.is_main_process:
            logger.info("***** Running GRADIENT CONSISTENCY analysis mode *****")

        tokenizers_local = [tokenizer_one, tokenizer_two]
        text_encoders_local = [text_encoder_one, text_encoder_two]

        run_gradient_consistency_analysis(
            accelerator=accelerator,
            transformer=transformer,
            text_encoder_one=text_encoder_one,
            text_encoder_two=text_encoder_two,
            vae=vae,
            noise_scheduler_copy=noise_scheduler_copy,
            train_dataloader=train_dataloader,
            router=router,
            args=args,
            weight_dtype=weight_dtype,
            tokenizers=tokenizers_local,
            text_encoders=text_encoders_local,
        )
        accelerator.end_training()
        return

    gc_train_mode = bool(getattr(args, "gc_train_mode", False))
    gc_train_every_steps = max(1, int(getattr(args, "gc_train_every_steps", 1)))
    gc_train_max_steps = int(getattr(args, "gc_train_max_steps", 1000))
    gc_train_out = getattr(args, "gc_train_out", None) or os.path.join(
        args.output_dir, "projected_gradient_alignment.csv"
    )
    gc_train_plot = getattr(args, "gc_train_plot", None) or os.path.join(
        args.output_dir, "projected_gradient_alignment.png"
    )
    gc_train_rows_written = 0
    gc_train_label_pair_mode = bool(getattr(args, "gc_train_label_pair_mode", False))
    gc_train_label_pair_out = getattr(args, "gc_train_label_pair_out", None) or os.path.join(
        args.output_dir, "projected_label_pair_cosine.csv"
    )
    gc_train_extended_metrics = bool(getattr(args, "gc_train_extended_metrics", False))
    gc_train_label_pair_rows_written = 0
    if gc_train_label_pair_mode and not gc_train_mode and accelerator.is_main_process:
        logger.warning(
            "gc_train_label_pair_mode is enabled but gc_train_mode is disabled; "
            "label-pair tracking will not run."
        )

    if gc_train_mode and accelerator.is_main_process:
        resumed_training = initial_global_step > 0
        if resumed_training and os.path.exists(gc_train_out):
            logger.info(
                "gc_train_mode: resuming with existing alignment CSV, appending rows to "
                f"{gc_train_out}"
            )
        else:
            init_projected_alignment_csv(gc_train_out)
        logger.info("***** Running training with projected gradient alignment tracking *****")
        logger.info(f"  Alignment CSV = {gc_train_out}")
        logger.info(f"  Alignment plot = {gc_train_plot}")
        logger.info(f"  Alignment cadence = every {gc_train_every_steps} steps")
        logger.info(
            "  Alignment max steps = "
            f"{gc_train_max_steps if gc_train_max_steps > 0 else 'unlimited (uses max_train_steps)'}"
        )
        if gc_train_extended_metrics:
            logger.info("  Extended gc_train metrics = enabled")
        if gc_train_label_pair_mode:
            if resumed_training and os.path.exists(gc_train_label_pair_out):
                logger.info(
                    "gc_train_mode: resuming with existing label-pair CSV, appending rows to "
                    f"{gc_train_label_pair_out}"
                )
            else:
                init_label_pair_csv(gc_train_label_pair_out)
            logger.info("  Label-pair cosine tracking = enabled")
            logger.info(f"  Label-pair CSV = {gc_train_label_pair_out}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # ---- CCUA setup ---------------------------------------------------------
    ccua_null_prompt = None
    ccua_null_pooled = None
    ccua_grabber = None
    if args.use_ccua:
        if args.router_type != "fake":
            raise ValueError(
                "CCUA currently supports only --router_type=fake (full-FT class-cond). "
                f"Got --router_type={args.router_type}."
            )
        if args.ccua_null_embeddings_path is None:
            raise ValueError(
                "CCUA requires --ccua_null_embeddings_path. "
                "Run scripts/precompute_null_embedding.py first."
            )
        ccua_null_prompt, ccua_null_pooled = load_null_embeddings(
            args.ccua_null_embeddings_path,
            device=accelerator.device,
            dtype=weight_dtype,
        )
        # Hook the last single-stream transformer block under the null pass.
        ccua_grabber = FeatureGrabber()
        ccua_grabber.attach(unwrap_model(transformer).single_transformer_blocks[-1])
        if accelerator.is_main_process:
            logger.info(
                f"CCUA enabled: alpha={args.ccua_alpha}, gamma={args.ccua_gamma}, "
                f"tau={args.ccua_tau}, t_weight_ucl={args.ccua_t_weight_ucl}, "
                f"pool_features={args.ccua_pool_features}"
            )
    # -------------------------------------------------------------------------

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            # set top parameter requires_grad = True for gradient checkpointing works
            unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            if args.router_type in ["gradient", "clip-cluster", "set", "label", "fake", "textlabel", "mocle"]:
                router.set_current_ids(batch[ToIndex.ID.value])
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one])
            with accelerator.accumulate(models_to_accumulate):
                if not args.precomputed_latents_path:
                    images = torch.stack(batch[ToIndex.IMG.value]).to(device=accelerator.device, dtype=vae.dtype)
                else:
                    images = batch[ToIndex.IMG.value]

                prompts = batch[ToIndex.CAPTION.value]

                if not args.train_text_encoder:
                    if args.precomputed_embeddings_path is None:
                        prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                            prompts, text_encoders, tokenizers,
                            max_sequence_length=args.max_sequence_length,
                            device=accelerator.device
                        )
                    else:
                        eos = batch[ToIndex.EMBEDDING.value]
                        prompt_embeds = torch.stack([eo.prompt_embeds for eo in eos]) \
                            .to(device=accelerator.device, dtype=weight_dtype)
                        pooled_prompt_embeds = torch.stack([eo.pooled_prompt_embeds for eo in eos]) \
                            .to(device=accelerator.device, dtype=weight_dtype)
                        try:
                            text_ids =eos[0].text_ids.unsqueeze(0).expand(args.max_sequence_length, -1) \
                                .to(device=accelerator.device)
                        except AttributeError:
                            text_ids = torch.zeros(size=(args.max_sequence_length, 3), dtype=weight_dtype, device=accelerator.device)
                else:
                    tokens_one = tokenize_prompt(tokenizer_one, prompts, max_sequence_length=77)
                    tokens_two = tokenize_prompt(
                        tokenizer_two, prompts, max_sequence_length=args.max_sequence_length
                    )
                    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=[None, None],
                        text_input_ids_list=[tokens_one, tokens_two],
                        max_sequence_length=args.max_sequence_length,
                        device=accelerator.device,
                        prompt=prompts,
                    )
                # Convert images to latent space
                if not args.precomputed_latents_path:
                    model_input = vae.encode(images).latent_dist.sample()
                else:
                    model_input = torch.stack([image.latent_dist.sample() for image in images])
                    # keep the batch dimension (a bare .squeeze() dropped it for batch size 1); identical for batch >= 2
                    model_input = model_input.reshape(model_input.shape[0], *model_input.shape[-3:]).to(transformer.device)
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype, device=accelerator.device)

                resample_buffer = torch.zeros((args.batch_buffer_size, *model_input.shape[1:]),
                                              dtype=model_input.dtype, device=model_input.device)

                model_input = torch.cat((model_input, resample_buffer))

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)

                noisy_model_input = ((1.0 - sigmas) * model_input + sigmas * noise)

                if args.batch_buffer_size != 0:
                    noisy_model_input = noisy_model_input[:-args.batch_buffer_size]
                if args.use_target_mixture:
                    target_mixture.update(variance=sigmas ** 2,
                                          latents=model_input,
                                          noisy_latents=noisy_model_input)
                if router is not None:
                    router.set_current_expert_indices(router.get_experts(), device=accelerator.device)
                if target_mixture and args.batch_buffer_size > 0:
                    x = torch.tensor(len(batch[0]) * [0], device=accelerator.device, dtype=model_input.dtype)
                    latents_by_expert, embeddings_by_expert, pooled_embeddings_by_expert = \
                    _copy_transfer(x,
                                   router.current_expert_indices, target_mixture,
                                   num_experts=router.num_experts,
                                   total_additions=args.batch_buffer_size,
                                   random_sampling=args.random_sampling,
                                   router=router,
                                   dataset=train_dataset,
                                   vae_factors = (vae.config.scaling_factor, vae.config.shift_factor))
                    index = noisy_model_input.shape[0]
                    for expert in range(router.num_experts):
                        if latents_by_expert.get(expert) is not None:
                            num_samples = latents_by_expert[expert].shape[0]
                            model_input[index:index+num_samples] = latents_by_expert[expert]
                            index = index + num_samples
                            prompt_embeds_expert = embeddings_by_expert[expert]
                            pooled_prompt_embeds_expert = pooled_embeddings_by_expert[expert]
                            prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_expert))
                            pooled_prompt_embeds = torch.cat((pooled_prompt_embeds, pooled_prompt_embeds_expert))
                            router.current_expert_indices = torch.cat((router.current_expert_indices, torch.tensor([expert] * num_samples).to(router.current_expert_indices.device)))

                    noisy_model_input = ((1.0 - sigmas) * model_input + sigmas * noise)

                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                # handle guidance
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

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

                null_kwargs = None
                model_pred_uncond_ref = None
                if args.use_ccua:
                    null_prompt_b, null_pooled_b = broadcast_null_embeddings(
                        ccua_null_prompt, ccua_null_pooled, batch_size=model_input.shape[0]
                    )
                    null_text_ids = torch.zeros(
                        size=(null_prompt_b.shape[1], 3),
                        dtype=text_ids.dtype,
                        device=text_ids.device,
                    )
                    null_kwargs = dict(batch_kwargs)
                    null_kwargs["pooled_projections"] = null_pooled_b
                    null_kwargs["encoder_hidden_states"] = null_prompt_b
                    null_kwargs["txt_ids"] = null_text_ids
                    with torch.no_grad():
                        model_pred_uncond_ref = transformer(**null_kwargs)[0].detach()

                # Predict the noise residual
                model_pred_packed = transformer(**batch_kwargs)[0]
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

                model_pred = FluxPipeline._unpack_latents(
                    model_pred_packed,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )

                if router is not None and args.router_type != "fake":
                    labels = list(batch[ToIndex.LABEL.value])
                    labels.extend(["Resampled"]*args.batch_buffer_size)
                    analyzer.update(router.current_expert_indices, step, labels)

                ccua_log = None
                if args.use_ccua:
                    timesteps_norm = (timesteps.float() / 1000.0).to(accelerator.device)
                    l_fm = loss.mean()
                    l_fm_log = l_fm.detach()

                    model_pred_cond_ref = model_pred_packed.detach()
                    l_al_cond = compute_alignment_loss(
                        model_pred_packed,
                        model_pred_uncond_ref,
                        timesteps_norm,
                    )
                    accelerator.backward(l_fm + args.ccua_gamma * l_al_cond)

                    del model_pred, model_pred_packed, model_pred_uncond_ref, l_fm, l_al_cond
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    ccua_grabber.feat = None
                    ccua_grabber.enabled = True
                    try:
                        model_pred_uncond_packed = transformer(**null_kwargs)[0]
                        null_features = ccua_grabber.feat
                    finally:
                        ccua_grabber.enabled = False
                    if null_features is None:
                        raise RuntimeError("CCUA feature hook did not capture null-pass features.")
                    l_al = compute_alignment_loss(
                        model_pred_cond_ref,
                        model_pred_uncond_packed,
                        timesteps_norm,
                    )
                    l_ucl_t_weight = timesteps_norm if args.ccua_t_weight_ucl else None
                    l_ucl = compute_unsupervised_contrastive_loss(
                        null_features,
                        tau=args.ccua_tau,
                        pool=args.ccua_pool_features,
                        timesteps_norm=l_ucl_t_weight,
                    )
                    accelerator.backward(args.ccua_gamma * l_al + args.ccua_alpha * l_ucl)
                    loss = l_fm_log + args.ccua_gamma * l_al.detach() + args.ccua_alpha * l_ucl.detach()
                    ccua_log = {
                        "ccua/l_fm": l_fm_log,
                        "ccua/l_al": l_al.detach(),
                        "ccua/l_ucl": l_ucl.detach(),
                    }
                    # Free the captured features so they don't linger across steps.
                    ccua_grabber.feat = None
                else:
                    loss = loss.mean()
                    aux = getattr(router, 'last_aux_loss', None) if router is not None else None
                    globals()['_router_aux_last'] = float(aux.detach()) if aux is not None else 0.0
                    router_aux_log = float(aux.detach()) if aux is not None else 0.0
                    if args.router_aux_loss_coef > 0.0 and aux is not None and aux.requires_grad:
                        loss = loss + args.router_aux_loss_coef * aux
                    accelerator.backward(loss)
                if args.use_lora:
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(
                                transformer_lora_parameters
                            )
                            if args.train_text_encoder
                            else transformer_lora_parameters
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                else:
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(
                                transformer.parameters(),
                                text_encoder_one.parameters(),
                                text_encoder_two.parameters(),
                            )
                            if args.train_text_encoder
                            else transformer.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if gc_train_mode and accelerator.sync_gradients:
                    tracked_step = global_step + 1
                    should_track = (
                        tracked_step % gc_train_every_steps == 0
                        and (gc_train_max_steps <= 0 or tracked_step <= gc_train_max_steps)
                    )
                    if should_track:
                        try:
                            labels_str = extract_labels_str_from_batch(batch)
                            alignment_rows = compute_projected_alignment_for_step(
                                model_unwrapped=unwrap_model(transformer),
                                router=router,
                                args=args,
                                step=tracked_step,
                                labels_str=labels_str,
                                model_input=model_input,
                                target=target,
                                weighting=weighting,
                                batch_kwargs=batch_kwargs,
                                vae_scale_factor=vae_scale_factor,
                                train_loss=float(loss.detach().item()),
                                lr=float(lr_scheduler.get_last_lr()[0]),
                            )
                            if accelerator.is_main_process and alignment_rows:
                                append_projected_alignment_rows(gc_train_out, alignment_rows)
                                gc_train_rows_written += len(alignment_rows)
                            if gc_train_label_pair_mode:
                                pair_rows = compute_label_pair_projected_cosine_for_step(
                                    model_unwrapped=unwrap_model(transformer),
                                    router=router,
                                    args=args,
                                    step=tracked_step,
                                    labels_str=labels_str,
                                    model_input=model_input,
                                    target=target,
                                    weighting=weighting,
                                    batch_kwargs=batch_kwargs,
                                    vae_scale_factor=vae_scale_factor,
                                    train_loss=float(loss.detach().item()),
                                    lr=float(lr_scheduler.get_last_lr()[0]),
                                )
                                if accelerator.is_main_process and pair_rows:
                                    append_label_pair_rows(gc_train_label_pair_out, pair_rows)
                                    gc_train_label_pair_rows_written += len(pair_rows)
                        except Exception as gc_train_error:
                            if accelerator.is_main_process:
                                logger.warning(
                                    f"gc_train_mode: failed to compute alignment at step {tracked_step}: "
                                    f"{gc_train_error}"
                                )

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        if os.path.isdir(save_path):
                            if args.overwrite_existing_checkpoints:
                                logger.info(
                                    f"Overwriting existing checkpoint directory at {save_path} "
                                    f"(step={global_step})."
                                )
                                shutil.rmtree(save_path)
                            else:
                                raise FileExistsError(
                                    f"Checkpoint directory already exists at {save_path}. "
                                    "Use --overwrite_existing_checkpoints to replace it."
                                )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.router_aux_loss_coef > 0.0:
                logs["router/aux_loss"] = globals().get("_router_aux_last", 0.0)
            if ccua_log is not None:
                logs.update({k: v.item() for k, v in ccua_log.items()})
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
        if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
            if router is not None:
                router.set_current_ids(["7c15331e-ae04cf36-a53644c9-cd0fd7a9-54b8c4b0"])
                router.set_current_expert_indices(router.get_experts(), device=accelerator.device)
            if accelerator.is_main_process:
                # create pipeline
                if not args.train_text_encoder:
                    text_encoder_one, text_encoder_two = load_text_encoders(args, text_encoder_cls_one, text_encoder_cls_two)
                    text_encoder_one.to(weight_dtype)
                    text_encoder_two.to(weight_dtype)
                pipeline = FluxPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=unwrap_model(text_encoder_one),
                    text_encoder_2=unwrap_model(text_encoder_two),
                    transformer=unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    token=os.environ["HUGGINGFACE_HUB_TOKEN"]
                )
                pipeline_args = {"prompt": args.validation_prompt}
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                    torch_dtype=weight_dtype,
                    logger=logger
                )

                if not args.train_text_encoder:
                    del text_encoder_one, text_encoder_two
                    free_memory()

                images = None
                del pipeline

        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_two = unwrap_model(text_encoder_two)
            pipeline = FluxPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                text_encoder=text_encoder_one,
                text_encoder_2=text_encoder_two,
                transformer=transformer,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                token=os.environ["HUGGINGFACE_HUB_TOKEN"]
            )
        else:
            pipeline = FluxPipeline.from_pretrained(
                args.pretrained_model_name_or_path, transformer=transformer, token=os.environ["HUGGINGFACE_HUB_TOKEN"]
            )

        # save the pipeline
        try:
            pipeline.save_pretrained(args.output_dir)
        except RuntimeError as e:
            # MoCLE-style shared parameters can trigger this in safetensors export.
            if "Some tensors share memory" in str(e):
                logger.warning(
                    "Shared tensors detected during final save to %s; retrying with safe_serialization=False.",
                    args.output_dir,
                )
                pipeline.save_pretrained(args.output_dir, safe_serialization=False)
            else:
                raise

        # Final inference (only when validation images are requested). Reloading the exported pipeline is not
        # needed otherwise, and MoE exports with shared parameters (MoCLE / learned gates, saved with
        # safe_serialization=False) cannot be reloaded as a plain FluxPipeline; generation uses checkpoint-<step>.
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            try:
                pipeline = FluxPipeline.from_pretrained(
                    args.output_dir,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype
                )
            except (OSError, ValueError) as e:
                logger.warning("Skipping final validation: could not reload the exported pipeline (%s)", e)
                args.num_validation_images = 0
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {"prompt": args.validation_prompt}
            images = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                epoch=epoch,
                is_final_validation=True,
                torch_dtype=weight_dtype,
                logger=logger
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                instance_prompt=args.instance_prompt,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    if gc_train_mode and accelerator.is_main_process:
        if gc_train_rows_written > 0:
            plot_projected_alignment_csv(gc_train_out, gc_train_plot)
            logger.info(
                f"gc_train_mode: wrote {gc_train_rows_written} alignment rows to {gc_train_out} "
                f"and plot to {gc_train_plot}"
            )
        else:
            logger.warning(
                f"gc_train_mode: no alignment rows were written. CSV path: {gc_train_out}"
            )
        if gc_train_label_pair_mode:
            if gc_train_label_pair_rows_written > 0:
                logger.info(
                    f"gc_train_label_pair_mode: wrote {gc_train_label_pair_rows_written} rows "
                    f"to {gc_train_label_pair_out}"
                )
            else:
                logger.warning(
                    f"gc_train_label_pair_mode: no rows were written. CSV path: {gc_train_label_pair_out}"
                )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
