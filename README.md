# GRASP: Guided Residual Adapters with Sample-wise Partitioning

Official code for **GRASP** (NeurIPS 2026): a static, conditioning-space partition routes every
training sample of a text-to-image flow-matching transformer (FLUX.1-dev) to a group-specific residual
adapter, so that rare conditions always update their own expert. The flow-matching objective and the
sampler are unchanged.

> Felix Nützel, Mischa Dombrowski, Bernhard Kainz. *GRASP: Guided Residual Adapters with Sample-wise
> Partitioning.* NeurIPS 2026. [paper](TODO) · [project page](TODO)

The run files reproduce every method on MIMIC-CXR-LT (full fine-tuning,
MoCLE, CCUA, shared adapter, GRASP variants, learned-gate MoE-GRASP, minority guidance), the ImageNet-LT
block and the downstream NIH-CXR-LT classifiers. There are four budget configurations, from the paper's 8-GPU setup
down to a minutes-long smoke test.

## Contents

| | |
|---|---|
| [Quick start](#quick-start) | install, check, smoke test |
| [docs/DATA.md](docs/DATA.md) | obtaining and preparing MIMIC-CXR-LT, NIH-CXR-LT and ImageNet-LT |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | one command per main table (Tables 1-3), plus the expected numbers |
| [docs/PARTITIONS.md](docs/PARTITIONS.md) | the GRASP partition, its exact label-to-expert lookup, the sensitivity variants |
| [docs/COMPUTE.md](docs/COMPUTE.md) | hardware profiles and GPU-hours |
| [docs/GENERATION_EMBEDDINGS.md](docs/GENERATION_EMBEDDINGS.md) | prompt embeddings used at generation time for the label-conditioned baselines |

## Repository layout

```
train_flux.py, routing.py, moe_layers.py, ccua.py, minority_guidance.py   training code (unchanged from the paper runs)
lora_adaption.py, analyzer.py, target_mixture/   adapter injection, expert-usage logging, per-step expert/latent bookkeeping (used by train_flux.py)
custom_datasets/  MIMIC-CXR-LT / ImageNet-LT datasets, latent and text-embedding precomputation
knowledge_graph/  label clustering used by the router and the partition builders
data/             checksums of the prepared data files (tools/verify_data.py)
evaluation/       generate_images.py (sampling), beyond_fid*.py (FID, coverage, IRS; overall + label-wise)
downstream/       classifier_ddp.py (DenseNet-121 trained on synthetic images, evaluated on NIH-CXR-LT)
gradient_consistency_analysis/   gradient-consistency analysis (Figure 4; hooks used by train_flux.py)
partitions/       maps.json (exact label->expert maps), builders for routing files, METIS / text-cluster rebuild
reproduce/        registry of every paper configuration, budget profiles, the CLI that plans and runs them
runs/             run files for Tables 1-3 (MIMIC-CXR-LT, +MG block, ImageNet-LT)
scripts/          data preparation, GCA launcher
tools/            result collection, comparison to the paper, data verification, tiny-subset builder
analysis/         statistics (Wilcoxon, class-recovery test, bootstrap) used in the paper
expected/         the numbers reported in the paper (for tools/compare_to_paper.py)
environment/      pinned requirements, conda env, Apptainer and Docker recipes, environment check
```

## Quick start

```bash
# 1) environment (exact pins of the paper container: Python 3.11, torch 2.9.0 + CUDA 12.8)
conda env create -f environment/environment.yml && conda activate grasp
#    or: apptainer build grasp.sif environment/grasp.def && export GRASP_CONTAINER=$PWD/grasp.sif
export HUGGINGFACE_HUB_TOKEN=...          # FLUX.1-dev is gated: accept its license on the Hub first
export HF_HOME=/fast/disk/hf_cache        # recommended on clusters (FLUX.1-dev is ~33 GB); must be visible
                                          # inside the container (a symlinked ~/.cache often is not)
python environment/check_env.py
python -m pytest                          # CPU-only checks: flags, partitions, sampling (seconds)

# 2) data (docs/DATA.md): MIMIC-CXR-JPG + reports (PhysioNet), LongTailCXR splits, ChestX-ray14
export GRASP_DATA=/path/to/data GRASP_OUT=/path/to/outputs
scripts/prepare_mimic.sh && scripts/prepare_nih.sh
python tools/verify_data.py               # routing files must match the paper byte for byte

# 3) smoke test: 10 training steps, 32 images, one classifier epoch on a tiny data subset (one GPU)
python tools/make_tiny_subset.py --src $GRASP_DATA --dst /tmp/grasp_tiny
GRASP_DATA=/tmp/grasp_tiny PROFILE=smoke runs/table1-3_mimic.sh   # or any run file

# 4) reproduce a table (docs/REPRODUCE.md)
LAUNCHER=slurm runs/table1-3_mimic.sh             # paper budget, 2 nodes x 4 GPUs
PROFILE=single_gpu runs/table1-3_mimic.sh         # same optimisation on one 80 GB GPU
python tools/collect_metrics.py && python tools/collect_classifier.py && python tools/compare_to_paper.py
```

Every run file is a thin wrapper around the CLI, which you can also call directly:

```bash
python -m reproduce.cli list                                                   # all configurations
python -m reproduce.cli plan --method grasp_lowdim --stage train --seed 0      # print the exact commands
python -m reproduce.cli run  --method grasp_lowdim --profile single_gpu        # train, generate, metrics, classify
```

## Budgets

| profile | hardware | what it reproduces |
|---|---|---|
| `paper` | 2 nodes × 4 × 80 GB GPUs | the paper setup exactly |
| `single_gpu` | 1 × 80 GB GPU | the same optimisation (gradient accumulation keeps the effective batch); ~8× longer. Full fine-tuning and CCUA need the `paper` profile |
| `reduced` | 1 × 80 GB GPU, about a day per method | 20% of the steps, 5k images, one seed: trends, not the paper numbers |
| `smoke` | 1 GPU | checks that the whole pipeline runs (minutes) |

See [docs/COMPUTE.md](docs/COMPUTE.md) for GPU-hours per stage.

## Citation

```bibtex
@inproceedings{nuetzel2026grasp,
  title     = {{GRASP}: Guided Residual Adapters with Sample-wise Partitioning},
  author    = {N{\"u}tzel, Felix and Dombrowski, Mischa and Kainz, Bernhard},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

Code: see [LICENSE](LICENSE). FLUX.1-dev weights are subject to the FLUX.1 [dev] Non-Commercial License.
MIMIC-CXR is distributed by PhysioNet under its credentialed data use agreement; this repository ships no
patient data and no data derived from reports (see [docs/DATA.md](docs/DATA.md)).
