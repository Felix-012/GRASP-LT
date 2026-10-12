import torch
from accelerate import DistributedType
from accelerate.utils.other import is_compiled_module
from diffusers import FluxTransformer2DModel, FluxPipeline
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch import nn


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def _collect_expert_modules(transformer, blocks=None):
    """
    Return fully qualified module names under 'experts' that are LoRA-compatible.
    - blocks: optional iterable of ints (e.g., [17, 18]) to restrict to specific transformer blocks
    """

    targets = []
    for name, module in transformer.named_modules():
        if "experts" not in name:
            continue
        if blocks is not None:
            if not any(f"transformer_blocks.{b}." in name for b in blocks):
                continue
        if ".ff." not in name and ".ff_context." not in name:
            continue

        if isinstance(module, nn.Linear):
            targets.append(name)

    return sorted(set(targets))


def initialize_lora_adapters(args, transformer):
    # Parse optional filters from args
    blocks = None
    if getattr(args, "lora_blocks", None):
        blocks = [int(b.strip()) for b in str(args.lora_blocks).split(",") if b.strip()]

    # Auto-discover all expert modules
    target_modules = _collect_expert_modules(transformer, blocks=blocks)

    if not target_modules:
        raise ValueError(
            "No LoRA targets found that contain 'experts'. "
            f"Checked blocks={blocks}. "
            "Print `list(n for n,_ in transformer.named_modules())` to inspect names."
        )

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    for n, p in transformer.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)

    return transformer, transformer_lora_config


def get_lora_save_model_hook(args, accelerator, transformer, text_encoder_one):
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    model = unwrap_model(accelerator, model)
                    if args.upcast_before_saving:
                        model = model.to(torch.float32)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )
    return save_model_hook

def get_lora_load_model_hook(args, logger, accelerator, transformer, text_encoder_one, text_encoder_two,
                             transformer_lora_config, text_encoder_cls_one, text_encoder_cls_two):
    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(accelerator,model), type(unwrap_model(accelerator,transformer))):
                    transformer_ = unwrap_model(accelerator,model)
                elif isinstance(unwrap_model(accelerator,model), type(unwrap_model(accelerator,text_encoder_one))):
                    text_encoder_one_ = unwrap_model(accelerator,model)
                elif isinstance(unwrap_model(accelerator,model), type(unwrap_model(accelerator,text_encoder_two))):
                    text_encoder_two_ = unwrap_model(accelerator, model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

        else:
            transformer_ = FluxTransformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer"
            )
            transformer_.add_adapter(transformer_lora_config)
            if args.train_text_encoder:
                text_encoder_one_ = text_encoder_cls_one.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="text_encoder"
                )
                text_encoder_two_ = text_encoder_cls_two.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="text_encoder_2"
                )

        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        if args.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)
    return load_model_hook

def get_lora_params_to_optimize(args, transformer):
    if args.mixed_precision == "fp16" or args.mixed_precision == "bf16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    return params_to_optimize, transformer_lora_parameters