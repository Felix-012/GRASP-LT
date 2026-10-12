# Reproducing the paper

Every configuration of the main results (Tables 1–3) is an entry of `reproduce/registry.py`
(`python -m reproduce.cli list`).
Each entry carries the flags of the launch scripts that produced the reported numbers, the generator seeds,
the sampling settings of the run and the training seed that was actually used.

Run files call the CLI for a group of rows; set `PROFILE`, `LAUNCHER`, `STAGE` and `SEEDS` as needed:

```bash
LAUNCHER=slurm runs/table1-3_mimic.sh                 # submits train -> generate -> metrics -> classify (afterok chained)
PROFILE=single_gpu STAGE=train runs/table2_imagenet.sh
EXTRA=--dry_run runs/table1-2_minority_guidance.sh     # only write the job scripts to $GRASP_OUT/jobs
```

On Slurm, cluster specifics go into environment variables, for example on a cluster with an `h100`
partition and typed GPUs: `GRASP_SLURM_ARGS="--partition=h100 --account=myproj" GRASP_GPU_TYPE=h100`.
Each stage becomes one job; `STAGE=all` chains them with `afterok`, and `--after <jobid>` lets a
generation-only method (e.g. the +MG rows) wait for its base training job. Job scripts and logs are kept in
`$GRASP_OUT/<profile>/jobs/` (one immutable script per submission).

Training runs longer than one job's walltime (full fine-tuning and CCUA at the paper budget, every method
on `single_gpu`) are submitted as several chained parts that resume from the latest checkpoint
(`afterany`); the number follows from the method's `train_hours`, the profile's `time_factor` and
`slurm_time`, and `--train_repeats N` overrides it. A part whose checkpoint already reached the stage's
`max_train_steps` skips that stage, so splitting never changes the schedule. Training keeps the last three
checkpoints (`--checkpoints_total_limit=3`; a full fine-tuning checkpoint is several tens of GB).

To evaluate an existing checkpoint without training, place or symlink it at
`$GRASP_OUT/<profile>/checkpoints/<dataset>/<method>/seed<S>/checkpoint-10000` and use `--stage eval`
(generate → metrics → classify), e.g. for a checkpoint trained elsewhere:

```bash
python -m reproduce.cli run --launcher slurm --method grasp_lowdim --stage eval
```

After the runs, collect and compare (pass `--profile` for non-paper budgets):

```bash
python tools/collect_metrics.py        # -> $GRASP_OUT/<profile>/tables/metrics_{runs,summary}.csv
python tools/collect_classifier.py     # -> $GRASP_OUT/<profile>/tables/classifier_{runs,summary}.csv
python tools/compare_to_paper.py       # flags values outside 2 reported SDs
```

## Table → run file

| paper | run file | methods |
|---|---|---|
| Tables 1–3 (MIMIC-CXR-LT) | `runs/table1-3_mimic.sh` | `ft_class ft_text_class mocle ccua adapter_768 grasp_768 grasp_lowdim grasp_highdim moe_grasp_lowdim moe_grasp_highdim real` |
| Tables 1–2 (+MG block) | `runs/table1-2_minority_guidance.sh` | `vanilla_mg grasp_lowdim_mg grasp_highdim_mg` (generation only; train the base models first) |
| Table 2 (ImageNet-LT) | `runs/table2_imagenet.sh` | `imagenet_*` |

Code of the further analyses in the paper: the Table 4 partition
variants (`partitions/maps.json`, `partitions/build_label_files.py`; train with the GRASP-LowDim flags and
the corresponding routing file), the static-initialized gate (`train_flux.py --mocle_grasp_init`), the
gradient-consistency analysis of Figure 4 (`gradient_consistency_analysis/`, enabled through the `--gc_train_*`
options of `train_flux.py`), the qualitative panels of Figure 3 (`custom_datasets/rebuttal_qual*.py`,
`utils/build_qualitative_figure.py`) and the analysis scripts in `analysis/`.

## What the stages do

| stage | entry point | paper setting |
|---|---|---|
| train | `train_flux.py` | FLUX.1-dev frozen (adapters) or fully fine-tuned; 10k steps (CCUA: 10k + resumed to 20k), lr 5e-5 cosine, warm-up 100, 8-bit AdamW, bf16, per-GPU batch 8 (CCUA 2) on 8 GPUs |
| generate | `evaluation/generate_images.py` | 50,000 images per generator seed from the 23,550 MIMIC-CXR-LT test prompts (cycled), 25 steps, guidance 5, 1024², batch 8 (MG: 4) |
| metrics | `evaluation/beyond_fid.py` | FID, coverage (k = 10) and adjusted IRS in DINOv2 and CXR feature space, overall and per label |
| classify | `downstream/classifier_ddp.py` | DenseNet-121, 100 epochs, batch 64, lr 3e-4, 512², hflip, classifier seeds 0/1/2, trained only on generated images, evaluated on NIH-CXR-LT (16,506 images) |

Statistics reported in the paper: `analysis/breadth_stats.py` (class recovery, exact paired test),
`analysis/mcnemar.py`, `analysis/bootstrap_classifier_difference.py`, and
`utils/compute_wilcoxon_classifier.py` (paired Wilcoxon over generator × classifier seed cells), all on
`$GRASP_OUT/<profile>/tables/classifier_runs.csv` (default profile `paper`).

## Expected numbers

`expected/paper_values.csv` holds every value of Tables 1–3.
Determinism of the metrics on fixed images (checked by recomputing a paper run from its original
images): FID and DINOv2 IRS reproduce exactly, CXR IRS to within ~.003. Precision, recall, density and
coverage are computed by BeyondFID on a random equal-size subsample of the larger set, which the paper runs
left unseeded; recomputation therefore moves all-label coverage by ~.001 and label-wise coverage of the
rarest labels by up to ~.3 (macro coverage by ~.01–.02). `evaluation/beyond_fid.py` now seeds this draw
(`--metric_seed`, default 0), so new runs are repeatable.

A few anchors (MIMIC-CXR-LT):

| method | CXR FID (all) | DINOv2 FID (all) | macro-F1 | #nz |
|---|---:|---:|---:|---:|
| GRASP-LowDim | .01 | 91.4 | .077 | 9 |
| GRASP-HighDim | .02 | 67.3 | .080 | 8 |
| MoE-GRASP-LowDim | .013 | 97.7 | .073 | 7 |
| MoCLE | .04 | 87.5 | .069 | 5 |
| Real data | – | – | .079 | 3 |

Reproductions are not bit-identical (non-deterministic GPU kernels, data-loader order across hardware);
`compare_to_paper.py` flags values outside two reported standard deviations.
