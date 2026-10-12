"""Hardware / budget profiles.

paper       exactly the paper setup: 2 nodes x 4 GPUs (8 x H100 80 GB), per-GPU batch 8 (effective 64),
            50,000 images per generator seed, 3 generator seeds x 3 classifier seeds, 100 classifier epochs.
single_gpu  the same optimisation on one 80 GB GPU: gradient accumulation keeps the effective batch at 64,
            so results match the paper up to non-determinism, but wall-clock is ~8x longer.
            Full fine-tuning and CCUA do not fit one GPU and are refused.
reduced     a one-day budget on one 80 GB GPU: 20% of the training steps, 5,000 images, one generator
            seed, one classifier seed, 30 classifier epochs (trained on the images of the first 5,000 test prompts).
            Reproduces trends, NOT the paper numbers.
smoke       a minutes-long end-to-end check of the pipeline (10 steps, 32 images, 1 classifier epoch).
smoke_2node the smoke budget on the paper topology (2 nodes x 4 GPUs), to test the multi-node Slurm launch.

time_factor is the training wall-clock relative to the paper setup; with the per-method estimate
(registry `train_hours`) it decides how many resumable Slurm jobs a training run is split into.
"""

PROFILES = {
    "paper": dict(nodes=2, gpus_per_node=4, grad_accum=1, step_scale=1.0, num_images=50000,
                  seeds=None, cls_seeds=(0, 1, 2), cls_epochs=100, cls_batch=64, cls_accum=1, checkpointing_steps=1000,
                  batch_override=None, allow_multi_gpu_only=True, slurm_time="24:00:00", time_factor=1.0,
                  checkpoints_total_limit=3),
    "single_gpu": dict(nodes=1, gpus_per_node=1, grad_accum=8, step_scale=1.0, num_images=50000,
                       seeds=None, cls_seeds=(0, 1, 2), cls_epochs=100, cls_batch=64, cls_accum=8, checkpointing_steps=1000,
                       batch_override=None, allow_multi_gpu_only=False, slurm_time="24:00:00", time_factor=8.0,
                       checkpoints_total_limit=3),
    "reduced": dict(nodes=1, gpus_per_node=1, grad_accum=8, step_scale=0.2, num_images=5000,
                    seeds=(0,), cls_seeds=(0,), cls_epochs=30, cls_batch=64, cls_accum=8, checkpointing_steps=500,
                    batch_override=None, allow_multi_gpu_only=False, slurm_time="24:00:00", time_factor=1.6,
                    checkpoints_total_limit=3),
    "smoke": dict(nodes=1, gpus_per_node=1, grad_accum=1, step_scale=None, smoke_steps=10, num_images=32,
                  seeds=(0,), cls_seeds=(0,), cls_epochs=1, cls_batch=8, cls_accum=1, checkpointing_steps=10,
                  batch_override=1, allow_multi_gpu_only=False, slurm_time="01:00:00", time_factor=0.0,
                  checkpoints_total_limit=None),
    "smoke_2node": dict(nodes=2, gpus_per_node=4, grad_accum=1, step_scale=None, smoke_steps=10, num_images=32,
                        seeds=(0,), cls_seeds=(0,), cls_epochs=1, cls_batch=8, cls_accum=1, checkpointing_steps=10,
                        batch_override=1, allow_multi_gpu_only=False, slurm_time="01:00:00", time_factor=0.0,
                        checkpoints_total_limit=None),
}


def get(name):
    if name not in PROFILES:
        raise KeyError(f"unknown profile '{name}'. Known: {', '.join(PROFILES)}")
    return PROFILES[name]


def slurm_hours(profile):
    """Walltime of one Slurm job of this profile, in hours (from "D-HH:MM:SS" or "HH:MM:SS")."""
    t = profile["slurm_time"]
    days, _, t = t.rpartition("-")
    h, m, sec = (int(x) for x in t.split(":"))
    return (int(days) if days else 0) * 24 + h + m / 60 + sec / 3600


def scaled_stages(stages, profile):
    """Training stages [(max_steps, warmup), ...] under a profile."""
    if profile.get("step_scale") is None:                    # smoke
        n = profile["smoke_steps"]
        out = []
        for i, (_, w) in enumerate(stages):
            out.append((n * (i + 1), min(w, 2)))
        return out
    return [(max(1, int(round(s * profile["step_scale"]))), int(round(w * profile["step_scale"])))
            for s, w in stages]
