# Compute

Paper runs used NVIDIA H100 80 GB GPUs (some later controls H200), PyTorch bf16. GPU-hours = wall-clock × GPUs.

## Per run (paper profile, one generator seed)

| stage | wall-clock | GPU-hours |
|---|---|---:|
| adapter training (GRASP, Adapter, MoE-GRASP, MoCLE), 10k steps, 8 GPUs | 7–12 h | 56–96 |
| full fine-tuning (FT), 10k steps, 8 GPUs | ~30 h | ~240 |
| CCUA, 10k + 10k steps at per-GPU batch 2, 8 GPUs | ~15 h + ~13 h | ~225 |
| generation, 50,000 images at 1024², 8 GPUs | ~10–12 h | 80–96 |
| generation with minority guidance (batch 4) | longer, ~130 per seed | ~130 |
| metrics (BeyondFID, both feature spaces), 1 GPU | 1–3 h | 1–3 |
| downstream classifiers, 3 classifier seeds, 8 GPUs | ~1.5 h | ~10–20 |

Reproducing Tables 1–3 (three generator seeds per row) takes roughly 12,000–13,000 GPU-hours, estimated
from the per-run figures above: about 8,400 for MIMIC-CXR-LT including minority guidance and the
downstream classifiers, and about 4,100 for ImageNet-LT.

## Profiles

| profile | GPUs | relative to paper |
|---|---|---|
| `paper` | 2 nodes × 4 | identical |
| `single_gpu` | 1 × 80 GB | same effective batch (gradient accumulation ×8), ≈ 8× wall-clock; classifier also accumulates ×8. FT / CCUA not supported |
| `reduced` | 1 × 80 GB | 20% of the training steps, 5,000 images (the classifier then uses the first 5,000 test prompts), seed 0, one classifier seed, 30 epochs: ≈ 1 day per adapter method |
| `smoke` | 1 GPU | 10 steps, 32 images, 1 classifier epoch (use with `tools/make_tiny_subset.py`) |
| `smoke_2node` | 2 nodes × 4 GPUs | the smoke budget on the paper topology, to test the multi-node Slurm launch |

Memory: adapter training of FLUX.1-dev with precomputed latents and text embeddings, gradient checkpointing
and 8-bit AdamW fits one 80 GB GPU at per-GPU batch 8; generation fits at batch 8 (batch 4 with minority
guidance). Smaller GPUs are only expected to work for the smoke profile with batch 1.
