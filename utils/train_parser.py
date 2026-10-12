import argparse
import os


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=str(os.environ["MODEL_NAME"]),
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
    "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=256,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")

    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ["OUTPUT_DIR"],
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=4200, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--overwrite_existing_checkpoints",
        action="store_true",
        help="If set, overwrite an existing checkpoint directory when saving a checkpoint at the same step.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--router_type", type=str, default="no", choices=["no", "clip-cluster", "gradient", "cluster", "set", "label", "fake", "textlabel", "mocle"])
    parser.add_argument("--precomputed_latents_path", type=str, default=None)
    parser.add_argument("--precomputed_embeddings_path", type=str, default=None)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument(
        "--lora_blocks",
        type=str,
        default=None,
        help=(
            "The transformer blocks to apply LoRA training on. Please specify the block numbers in a comma seperated manner."
            'E.g. - "--lora_blocks 12,30" will result in lora training of transformer blocks 12 and 30. For more examples refer to https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_SD3.md'
        ),
    )
    parser.add_argument("--instance_prompt", type=str, default=None)
    parser.add_argument("--bitmaps_path", type=str, default=None)
    parser.add_argument("--id2cui_path", type=str, default=None)
    parser.add_argument("--cui2sty_path", type=str, default=None)
    parser.add_argument("--medoid_path", type=str, default=None)
    parser.add_argument("--label_path", type=str, default=None)
    parser.add_argument(
        "--router_label_permute_seed",
        type=int,
        default=-1,
        help="If >=0, shuffle label-router assignments across samples with this seed (semantic permutation control).",
    )
    parser.add_argument("--use_target_mixture", action="store_true")
    parser.add_argument("--copy_weights", action="store_true")
    parser.add_argument("--moe_type", type=str, choices=["default", "delta", "mocle"])
    parser.add_argument("--moe_target_blocks", type=int, default=None)
    parser.add_argument("--ortho_only", action="store_true")
    parser.add_argument("--fill_empty", action="store_true")
    parser.add_argument("--reference_batch_size", type=int, default=0)
    parser.add_argument("--batch_buffer_size", type=int, default=0)
    parser.add_argument("--var_eq_weight", type=float, default=0)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--random_sampling", action="store_true")
    parser.add_argument("--bottleneck", type=int,default=None)
    parser.add_argument("--single_blocks", nargs='+', default=None)
    parser.add_argument("--dual_blocks", nargs='+', default=None)
    parser.add_argument("--n_single", type=int, default=None)
    parser.add_argument("--n_dual", type=int, default=None)
    parser.add_argument("--test_mode_expert", type=int, default=-1)
    parser.add_argument("--features_path", type=str, default=None)
    parser.add_argument("--centers_path", type=str, default=None)
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.0,
                        help="Coefficient for the learned router load-balancing "
                             "auxiliary loss. 0.0 disables it and reproduces the "
                             "submitted MoE-GRASP behaviour exactly.")
    parser.add_argument("--mocle_grasp_init", action="store_true",
                        help="Initialize the learned MoCLE gate to reproduce the fixed "
                             "cluster->expert (GRASP) assignment at ~0.9 confidence, then "
                             "let it finetune (static-initialized-then-finetuned hybrid).")
    parser.add_argument("--dataset_name", type=str, default="mimic-lt")

    parser.add_argument("--gc_mode", action="store_true",
                        help="Run gradient consistency analysis instead of training.")
    parser.add_argument("--gc_num_batches", type=int, default=25,
                        help="How many training batches to analyze. <=0 means one full epoch.")
    parser.add_argument("--gc_proj_k", type=int, default=1024,
                        help="Dimensionality of Gaussian projection for gradients.")
    parser.add_argument("--gc_proj_seed", type=int, default=0,
                        help="Seed for Gaussian projection.")
    parser.add_argument("--gc_max_samples_per_expert", type=int, default=32,
                        help="Cap per-expert subset size (for speed).")
    parser.add_argument("--gc_min_subset_size", type=int, default=2,
                        help="Minimum subset size required for GC subgroup metrics.")
    parser.add_argument("--gc_out", type=str, default=None,
                        help="Optional path to write CSV results. Default: <output_dir>/gradient_consistency.csv")
    parser.add_argument("--gc_include_gate", action="store_true",
                        help="Also analyze MoCLE router gate parameters (in addition to expert params).")

    parser.add_argument("--gc_grouping", choices=["routed", "labels"], default="routed", type=str)
    parser.add_argument(
        "--gc_train_mode",
        action="store_true",
        help="Run projected gradient alignment tracking during normal training.",
    )
    parser.add_argument(
        "--gc_train_every_steps",
        type=int,
        default=1,
        help="Compute projected gradient alignment every N optimization steps in gc_train_mode.",
    )
    parser.add_argument(
        "--gc_train_max_steps",
        type=int,
        default=1000,
        help="Maximum number of optimization steps to track in gc_train_mode (<=0 means unlimited).",
    )
    parser.add_argument(
        "--gc_train_max_samples_per_expert",
        type=int,
        default=32,
        help="Cap the number of active samples per expert used for projected alignment during training.",
    )
    parser.add_argument(
        "--gc_train_min_subset_size",
        type=int,
        default=2,
        help="Minimum samples required for active and dominant subsets during gc_train_mode.",
    )
    parser.add_argument(
        "--gc_train_out",
        type=str,
        default=None,
        help="Optional path to write per-step projected alignment CSV. Default: <output_dir>/projected_gradient_alignment.csv",
    )
    parser.add_argument(
        "--gc_train_plot",
        type=str,
        default=None,
        help="Optional path to write per-run projected alignment plot. Default: <output_dir>/projected_gradient_alignment.png",
    )
    parser.add_argument(
        "--gc_train_extended_metrics",
        action="store_true",
        help=(
            "Also compute extended gc_train metrics per row: conflict ratio, EVR, principal-direction alignment "
            "and intra/inter gradient cosine diagnostics."
        ),
    )
    parser.add_argument(
        "--gc_train_label_pair_mode",
        action="store_true",
        help="Track projected gradient cosine for label pairs during training.",
    )
    parser.add_argument(
        "--gc_train_label_pair_out",
        type=str,
        default=None,
        help="Optional path to write per-step label-pair projected gradient cosine CSV. "
             "Default: <output_dir>/projected_label_pair_cosine.csv",
    )
    parser.add_argument(
        "--gc_train_label_pair_min_subset_size",
        type=int,
        default=2,
        help="Minimum samples per label to include it in label-pair projected cosine tracking.",
    )
    parser.add_argument(
        "--gc_train_label_pair_max_samples_per_label",
        type=int,
        default=32,
        help="Cap samples per label used to estimate projected label gradients.",
    )
    parser.add_argument(
        "--gc_train_label_pair_max_labels",
        type=int,
        default=0,
        help="If >0, keep only top-K labels by batch frequency for label-pair tracking.",
    )
    parser.add_argument(
        "--gc_train_label_pair_param_group",
        choices=["expert", "trainable"],
        type=str,
        default="expert",
        help="Parameter subset for label-pair projected gradients: expert-only or all trainable params.",
    )

    # ---- CCUA (Liang et al. 2025, arXiv:2507.09052) baseline ----
    parser.add_argument(
        "--use_ccua",
        action="store_true",
        help="Enable CCUA training-time loss: L = L_fm + alpha*L_ucl + gamma*L_al.",
    )
    parser.add_argument(
        "--ccua_alpha",
        type=float,
        default=0.5,
        help="Weight on the unsupervised contrastive (L_ucl) term. Paper does not report a value; 0.5 is a starting point.",
    )
    parser.add_argument(
        "--ccua_gamma",
        type=float,
        default=0.5,
        help="Weight on the alignment (L_al) term. Paper does not report a value; 0.5 is a starting point.",
    )
    parser.add_argument(
        "--ccua_tau",
        type=float,
        default=0.1,
        help="Temperature for the InfoNCE contrastive loss (paper: 0.1).",
    )
    parser.add_argument(
        "--ccua_uncond_dropout",
        type=float,
        default=0.1,
        help="Probability of replacing the conditional pass with the unconditional pass during training (paper: 0.1, used for CFG compatibility).",
    )
    parser.add_argument(
        "--ccua_t_weight_ucl",
        action="store_true",
        help="If set, weight the L_ucl contrastive loss by t (paper does this on TinyImageNet-LT/Places-LT, not on ImageNet-LT).",
    )
    parser.add_argument(
        "--ccua_null_embeddings_path",
        type=str,
        default=None,
        help="Path to a precomputed null/empty-prompt embedding (.h5 with prompt_embeds, pooled_prompt_embeds for the empty string). If unset, CCUA will refuse to run.",
    )
    parser.add_argument(
        "--ccua_pool_features",
        action="store_true",
        help="If set, pool encoder features (mean over tokens) before the InfoNCE loss; default is to flatten all tokens.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
