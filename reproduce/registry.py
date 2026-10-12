"""Registry of every configuration reported in the paper.

Each entry reproduces the flags of the launch scripts that produced the reported
numbers (see docs/REPRODUCE.md for the table-row mapping). Paths are relative to
the data root ($GRASP_DATA); output folders are managed by reproduce.cli.

Fields
  dataset        "mimic-lt" or "imagenet-lt"
  rows           where the configuration appears in the paper
  train          flags for train_flux.py (None for generation-only entries)
  base           generation-only entries: which trained method's checkpoint to sample from
  emb_train      prompt-embedding file used for training
  emb_gen        prompt-embedding file used for generation, exactly as in the paper
  emb_gen_fixed  test-prompt embeddings to use with --fix_gen_embeddings (see docs/GENERATION_EMBEDDINGS.md)
  gen            flags for evaluation/generate_images.py
  batch          per-GPU training batch size (paper: 8 GPUs)
  stages         list of (max_train_steps, lr_warmup_steps); several stages resume in the same folder
  seed_mode      noise layout used when generating the reported row (evaluation/generate_images.py --seed_mode)
  legacy_seed0   training seed actually used for generator seed 0 (several original scripts did not pass
                 --seed and fell back to the parser default 4200); None means "same as generator seed"
  seeds          generator seeds reported in the paper
  multi_gpu      True if a single 80 GB GPU cannot hold the job (full fine-tuning)
  gen_batch      generation batch size
  train_hours    training wall-clock on the paper setup (8 x H100), used to split training into
                 resumable Slurm jobs that fit the walltime limit
"""

PAPER_SEEDS = (0, 5868, 9090)
# Base noise seed of generation; the generator seeds above select the trained model.
PAPER_NOISE_SEED = 0

MIMIC_COMMON_TRAIN = [
    "--learning_rate=5e-5",
    "--gradient_checkpointing",
    "--mixed_precision=bf16",
    "--use_8bit_adam",
    "--lr_scheduler=cosine",
    "--precomputed_latents_path={data}/mimic-lt/image_latents_flux-lt.h5",
    "--instance_data_dir={data}/mimic-lt/",
]
IMAGENET_COMMON_TRAIN = [
    "--learning_rate=5e-5",
    "--gradient_checkpointing",
    "--mixed_precision=bf16",
    "--use_8bit_adam",
    "--lr_scheduler=cosine",
    "--precomputed_latents_path={data}/imagenet-lt/imagenet_train_latents_flux-lt.h5",
    "--instance_data_dir={data}/imagenet-lt/",
    "--dataset_name=imagenet-lt",
]
MIMIC_GEN = [
    "--metadata_csv={data}/mimic-lt/mimic-cxr-lt_test.csv",
    "--prompt_key=impression",
    "--id_key=dicom_id",
]
IMAGENET_GEN = [
    "--metadata_csv={data}/imagenet-lt/imagenet_lt_test.csv",
    "--prompt_key=Finding Labels",     # unused when precomputed embeddings are given (the paper scripts passed "impression")
    "--id_key=id",
]

LABELS_TRAIN = "{data}/mimic-lt/labels_gt.txt"
LABELS_TEST = "{data}/mimic-lt/test/labels_gt.txt"
TEXT_LABELS = "{data}/mimic-lt/text_cluster/train/labels.txt"
TEXT_CENTERS = "{data}/mimic-lt/text_cluster/train/centroids.npz"
IN_TEXT_LABELS = "{data}/imagenet-lt/text_cluster/train/labels.txt"
IN_TEXT_CENTERS = "{data}/imagenet-lt/text_cluster/train/centroids.npz"

EMB_IMPRESSION_TRAIN = "{data}/mimic-lt/embeddings_flux-lt.h5"          # report impression (train)
EMB_IMPRESSION_TEST = "{data}/mimic-lt/embeddings_flux-lt_test.h5"      # report impression (test)
EMB_LABEL_TRAIN = "{data}/mimic-lt/embeddings_mimic-lt_labels_train.h5"   # finding label text (train)
EMB_LABEL_TEST = "{data}/mimic-lt/embeddings_mimic-lt_labels_test.h5"     # finding label text (test)
EMB_LABELTXT_TRAIN = "{data}/mimic-lt/embeddings_mimic-lt_label_txt_train.h5"
EMB_LABELTXT_TEST = "{data}/mimic-lt/embeddings_mimic-lt_label_txt_test.h5"
EMB_IN_TRAIN = "{data}/imagenet-lt/embeddings_imagenet-lt_flux_train.h5"
EMB_IN_TEST = "{data}/imagenet-lt/embeddings_imagenet-lt_flux_test.h5"

LOWDIM = ["--bottleneck=16", "--n_dual=19", "--n_single=38"]
DIM768 = ["--bottleneck=768", "--n_dual=19", "--n_single=38"]
HIGHDIM = ["--moe_target_blocks=4"]          # full-width (4d) experts in the last four dual-stream blocks
MG = ["--use_minority_guidance", "--mg_scale=0.4", "--mg_inter_rate=5", "--mg_p_ratio=0.2",
      "--mg_t_min=0.2", "--mg_t_max=0.95", "--mg_num_inference_steps=25", "--mg_guidance_scale=5.0"]
CCUA = ["--use_ccua", "--ccua_alpha=0.5", "--ccua_gamma=0.5", "--ccua_tau=0.1", "--ccua_uncond_dropout=0.1",
        "--ccua_pool_features", "--ccua_null_embeddings_path={data}/mimic-lt/null_embedding.h5"]


def _m(**kw):
    d = dict(dataset="mimic-lt", rows=(), train=None, base=None, emb_train=EMB_IMPRESSION_TRAIN,
             emb_gen=EMB_IMPRESSION_TEST, emb_gen_fixed=None, gen=[], batch=8, stages=[(10000, 100)],
             seed_mode="per_sample", legacy_seed0=None, seeds=PAPER_SEEDS, multi_gpu=False, gen_batch=8,
             train_hours=12, note="")
    d.update(kw)
    return d


METHODS = {
    # ------------------------------------------------------------- MIMIC-CXR-LT baselines
    "ft_class": _m(
        rows=("Tab. 1-3: Full Fine-Tune (class)",),
        train=MIMIC_COMMON_TRAIN + ["--router_type=fake", f"--label_path={LABELS_TRAIN}"],
        emb_train=EMB_LABEL_TRAIN, emb_gen=EMB_LABEL_TRAIN, emb_gen_fixed=EMB_LABEL_TEST,
        gen=["--router_type=no"], seed_mode="per_batch", legacy_seed0=4200, multi_gpu=True, train_hours=30),
    "ft_text_class": _m(
        rows=("Tab. 1-3: Full Fine-Tune (text+class)",),
        train=MIMIC_COMMON_TRAIN + ["--router_type=fake", f"--label_path={LABELS_TRAIN}"],
        emb_train=EMB_LABELTXT_TRAIN, emb_gen=EMB_LABELTXT_TRAIN, emb_gen_fixed=EMB_LABELTXT_TEST,
        gen=["--router_type=no"], seed_mode="per_batch", legacy_seed0=4200, multi_gpu=True, train_hours=30),
    "mocle": _m(
        rows=("Tab. 1-3: MoCLE",),
        train=MIMIC_COMMON_TRAIN + ["--router_type=mocle", "--moe_type=mocle", "--freeze", "--n_dual=19",
                                    f"--label_path={TEXT_LABELS}", f"--centers_path={TEXT_CENTERS}"],
        gen=["--router_type=mocle", "--moe_type=mocle", "--n_dual=19", f"--centers_path={TEXT_CENTERS}"],
        seed_mode="per_batch", legacy_seed0=4200,
        note="LoRA rank 64 (code default when --bottleneck is not given)."),
    "ccua": _m(
        rows=("Tab. 1-3: CCUA (20k steps)",),
        train=MIMIC_COMMON_TRAIN + ["--router_type=fake", f"--label_path={LABELS_TRAIN}"] + CCUA,
        emb_train=EMB_LABEL_TRAIN, emb_gen=EMB_LABEL_TRAIN, emb_gen_fixed=EMB_LABEL_TEST,
        gen=["--router_type=no"], batch=2, stages=[(10000, 100), (20000, 0)], multi_gpu=True, train_hours=28, seed_mode="per_batch",
        note="Trained 10k steps, then resumed in the same folder to 20k steps without warm-up."),
    "adapter_768": _m(
        rows=("Tab. 1-3: Adapter (768d)",),
        train=MIMIC_COMMON_TRAIN + ["--router_type=label", "--moe_type=delta", "--freeze", f"--label_path={LABELS_TRAIN}",
                                    "--test_mode_expert=0"] + DIM768,
        gen=["--router_type=label", "--moe_type=delta", f"--label_path={LABELS_TEST}", "--test_mode_expert=0"] + DIM768,
        seed_mode="per_batch", legacy_seed0=4200),
    # ------------------------------------------------------------- MIMIC-CXR-LT GRASP
    "grasp_768": _m(
        rows=("Tab. 1-3: GRASP (768d)",),
        train=MIMIC_COMMON_TRAIN + ["--router_type=label", "--moe_type=delta", "--freeze", f"--label_path={LABELS_TRAIN}"] + DIM768,
        gen=["--router_type=label", "--moe_type=delta", f"--label_path={LABELS_TEST}"] + DIM768,
        seed_mode="per_batch", legacy_seed0=4200),
    "grasp_lowdim": _m(
        rows=("Tab. 1-3: GRASP-LowDim",),
        train=MIMIC_COMMON_TRAIN + ["--router_type=label", "--moe_type=delta", "--freeze", f"--label_path={LABELS_TRAIN}"] + LOWDIM,
        gen=["--router_type=label", "--moe_type=delta", f"--label_path={LABELS_TEST}"] + LOWDIM,
        seed_mode="per_batch"),
    "grasp_highdim": _m(
        rows=("Tab. 1-3: GRASP-HighDim",),
        train=MIMIC_COMMON_TRAIN + ["--router_type=label", "--moe_type=delta", "--freeze", f"--label_path={LABELS_TRAIN}"] + HIGHDIM,
        gen=["--router_type=label", "--moe_type=delta", f"--label_path={LABELS_TEST}"] + HIGHDIM,
        seed_mode="per_batch", legacy_seed0=4200),
    # ------------------------------------------------------------- learned routing (MoE-GRASP)
    "moe_grasp_lowdim": _m(
        rows=("Tab. 1-3: MoE-GRASP-LowDim",),
        train=MIMIC_COMMON_TRAIN + ["--router_type=mocle", "--moe_type=delta", "--freeze", f"--label_path={LABELS_TRAIN}"] + LOWDIM,
        gen=["--router_type=mocle", "--moe_type=delta", f"--label_path={LABELS_TEST}"] + LOWDIM,
        seed_mode="per_batch"),
    "moe_grasp_highdim": _m(
        rows=("Tab. 1-3: MoE-GRASP-HighDim",),
        train=MIMIC_COMMON_TRAIN + ["--router_type=mocle", "--moe_type=delta", "--freeze", f"--label_path={LABELS_TRAIN}"] + HIGHDIM,
        gen=["--router_type=mocle", "--moe_type=delta", f"--label_path={LABELS_TEST}"] + HIGHDIM,
        seed_mode="per_batch"),
}

# ------------------------------------------------------------- minority guidance (generation only)
METHODS.update({
    "vanilla_mg": _m(rows=("Tab. 1-2: Vanilla + MG",), base="ft_class",
                     emb_gen=EMB_LABEL_TRAIN, emb_gen_fixed=EMB_LABEL_TEST, gen=["--router_type=no"] + MG),
    "grasp_lowdim_mg": _m(rows=("Tab. 1-2: GRASP-LowDim + MG",), base="grasp_lowdim",
                          gen=METHODS["grasp_lowdim"]["gen"] + MG),
    "grasp_highdim_mg": _m(rows=("Tab. 1-2: GRASP-HighDim + MG",), base="grasp_highdim",
                           gen=METHODS["grasp_highdim"]["gen"] + MG),
})
for _k in ("vanilla_mg", "grasp_lowdim_mg", "grasp_highdim_mg"):
    METHODS[_k]["gen_batch"] = 4   # minority guidance back-propagates through the model; batch 4 fits 80 GB

# ------------------------------------------------------------- ImageNet-LT (Tab. 2, right block)
_IN = dict(dataset="imagenet-lt", emb_train=EMB_IN_TRAIN, emb_gen=EMB_IN_TEST, seed_mode="per_sample")
METHODS.update({
    "imagenet_ft": _m(rows=("Tab. 2: ImageNet-LT Full Fine-Tune (class)",), multi_gpu=True, train_hours=30,
                      train=IMAGENET_COMMON_TRAIN + ["--router_type=no"], gen=["--router_type=no"],
                      **{**_IN, "legacy_seed0": 4200}),
    "imagenet_mocle": _m(rows=("Tab. 2: ImageNet-LT MoCLE",),
                         train=IMAGENET_COMMON_TRAIN + ["--router_type=mocle", "--moe_type=mocle", "--freeze", "--n_dual=19",
                                                        f"--label_path={IN_TEXT_LABELS}", f"--centers_path={IN_TEXT_CENTERS}"],
                         gen=["--router_type=mocle", "--moe_type=mocle", "--n_dual=19", f"--centers_path={IN_TEXT_CENTERS}"], **_IN),
    "imagenet_adapter_768": _m(rows=("Tab. 2: ImageNet-LT Adapter (768d)",),
                               train=IMAGENET_COMMON_TRAIN + ["--router_type=label", "--moe_type=delta", "--freeze",
                                                              f"--label_path={IN_TEXT_LABELS}", "--test_mode_expert=0"] + DIM768,
                               gen=["--router_type=textlabel", "--moe_type=delta", f"--centers_path={IN_TEXT_CENTERS}",
                                    "--test_mode_expert=0"] + DIM768, **_IN),
    "imagenet_grasp_768": _m(rows=("Tab. 2: ImageNet-LT GRASP (768d)",),
                             train=IMAGENET_COMMON_TRAIN + ["--router_type=label", "--moe_type=delta", "--freeze",
                                                            f"--label_path={IN_TEXT_LABELS}"] + DIM768,
                             gen=["--router_type=textlabel", "--moe_type=delta", f"--centers_path={IN_TEXT_CENTERS}"] + DIM768, **_IN),
    "imagenet_grasp_lowdim": _m(rows=("Tab. 2: ImageNet-LT GRASP-LowDim",),
                                train=IMAGENET_COMMON_TRAIN + ["--router_type=label", "--moe_type=delta", "--freeze",
                                                               f"--label_path={IN_TEXT_LABELS}"] + LOWDIM,
                                gen=["--router_type=textlabel", "--moe_type=delta", f"--centers_path={IN_TEXT_CENTERS}"] + LOWDIM, **_IN),
    "imagenet_grasp_highdim": _m(rows=("Tab. 2: ImageNet-LT GRASP-HighDim",),
                                 train=IMAGENET_COMMON_TRAIN + ["--router_type=label", "--moe_type=delta", "--freeze",
                                                                f"--label_path={IN_TEXT_LABELS}"] + HIGHDIM,
                                 gen=["--router_type=textlabel", "--moe_type=delta", f"--centers_path={IN_TEXT_CENTERS}"] + HIGHDIM,
                                 **{**_IN, "legacy_seed0": 4200}),
    "imagenet_moe_grasp_lowdim": _m(rows=("Tab. 2: ImageNet-LT MoE-GRASP-LowDim",),
                                    train=IMAGENET_COMMON_TRAIN + ["--router_type=mocle", "--moe_type=delta", "--freeze",
                                                                   f"--label_path={IN_TEXT_LABELS}", f"--centers_path={IN_TEXT_CENTERS}"] + LOWDIM,
                                    gen=["--router_type=mocle", "--moe_type=delta", f"--centers_path={IN_TEXT_CENTERS}"] + LOWDIM,
                                    **{**_IN, "seed_mode": "per_sample"}),
})

# Methods whose checkpoint is shared with a generation-only entry.
GEN_ONLY = {k for k, v in METHODS.items() if v["train"] is None}


def get(name):
    if name not in METHODS:
        raise KeyError(f"unknown method '{name}'. Known: {', '.join(sorted(METHODS))}")
    return METHODS[name]
