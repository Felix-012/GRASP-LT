#!/usr/bin/env python3
"""Plan and run every experiment of the GRASP paper.

    python -m reproduce.cli list
    python -m reproduce.cli plan  --method grasp_lowdim --stage train --seed 0 --profile paper
    python -m reproduce.cli run   --method grasp_lowdim --stage all   --profile single_gpu
    python -m reproduce.cli run   --method grasp_lowdim --stage all   --launcher slurm
    python -m reproduce.cli run   --method real --stage classify          # real-data reference classifier

Stages: train, generate, metrics, classify, all (= the four in order; generation-only methods skip train),
eval (= generate, metrics, classify from an existing checkpoint, e.g. one trained elsewhere and placed or
symlinked at $GRASP_OUT/<profile>/checkpoints/<dataset>/<method>/seed<S>/checkpoint-<steps>).

Environment: GRASP_DATA (data root, see docs/DATA.md), GRASP_OUT (outputs, default ./outputs),
MODEL_NAME (default black-forest-labs/FLUX.1-dev), HUGGINGFACE_HUB_TOKEN, HF_HOME, optional
GRASP_CONTAINER / GRASP_METRICS_CONTAINER (Apptainer images; commands are wrapped in `apptainer exec --nv`),
GRASP_SLURM_ARGS (extra #SBATCH options, e.g. "--partition=h100 --account=xyz"), GRASP_GPU_TYPE
(e.g. h100 -> --gres=gpu:h100:N), GRASP_REPORT_TO (tensorboard | wandb | none).

Outputs are written to $GRASP_OUT/<profile>/<kind>/<dataset>/<method>/<variant>/seed<S>/, where kind is
checkpoints | generated | metrics | classifier and variant is the seed mode (plus "-testemb" with
--fix_gen_embeddings). Checkpoints do not depend on the variant.
"""
import argparse
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from reproduce import profiles as P
from reproduce import registry as R

REPO = Path(__file__).resolve().parents[1]
STAGES = ("train", "generate", "metrics", "classify")
EXIT_UNSUPPORTED = 3


# ----------------------------------------------------------------------------- paths
def env_paths(profile_name):
    data = os.environ.get("GRASP_DATA")
    if not data:
        sys.exit("GRASP_DATA is not set (see docs/DATA.md)")
    out = Path(os.environ.get("GRASP_OUT", str(REPO / "outputs"))).expanduser().resolve() / profile_name
    return str(Path(data).expanduser().resolve()), out


def fmt(items, data):
    return [x.format(data=data) for x in items]


def variant(m, seed_mode, fix_emb, noise_seed=R.PAPER_NOISE_SEED):
    return (seed_mode + ("-testemb" if (fix_emb and m.get("emb_gen_fixed")) else "")
            + (f"-noise{noise_seed}" if noise_seed != R.PAPER_NOISE_SEED else ""))


def run_dir(out, kind, m, method, seed, var=None):
    d = Path(out) / kind / m["dataset"] / method
    if var:
        d = d / var
    return d / f"seed{seed}"


def train_seed(m, seed):
    return m["legacy_seed0"] if (seed == 0 and m["legacy_seed0"] is not None) else seed


# ----------------------------------------------------------------------------- launchers
def accelerate_prefix(profile, port=29500):
    nproc = profile["nodes"] * profile["gpus_per_node"]
    if profile["nodes"] > 1:
        return ["accelerate", "launch", "--num_machines", "$SLURM_JOB_NUM_NODES", "--num_processes", str(nproc),
                "--machine_rank", "$SLURM_NODEID", "--main_process_ip", "$MASTER_ADDR",
                "--main_process_port", str(port), "--mixed_precision", "bf16"]
    return ["accelerate", "launch", "--num_machines", "1", "--num_processes", str(nproc),
            "--main_process_port", str(port), "--mixed_precision", "bf16"]


def container(cmd, var="GRASP_CONTAINER"):
    img = os.environ.get(var)
    return (["apptainer", "exec", "--nv", img] + cmd) if img else cmd


def q(tokens):
    """Shell-quote every token except bare $VARS (expanded by the job shell)."""
    return " ".join(t if (t.startswith("$") and " " not in t) else shlex.quote(t) for t in tokens)


def unsupported(msg):
    print(f"SKIP {msg}", file=sys.stderr)
    raise SystemExit(EXIT_UNSUPPORTED)


# ----------------------------------------------------------------------------- command builders
def build_train(method, m, seed, profile, data, out):
    if m["train"] is None:
        return []
    if m["multi_gpu"] and not profile["allow_multi_gpu_only"]:
        unsupported(f"{method}: full fine-tuning needs more than one 80 GB GPU (paper: 8 x H100); "
                    "use --profile paper, or generate from a checkpoint trained elsewhere.")
    ckpt = run_dir(out, "checkpoints", m, method, seed)
    batch = profile["batch_override"] or m["batch"]
    cmds = []
    for steps, warmup in P.scaled_stages(m["stages"], profile):
        args = fmt(m["train"], data) + [
            f"--precomputed_embeddings_path={m['emb_train'].format(data=data)}",
            f"--output_dir={ckpt}",
            f"--train_batch_size={batch}",
            f"--gradient_accumulation_steps={profile['grad_accum']}",
            f"--max_train_steps={steps}",
            f"--lr_warmup_steps={warmup}",
            f"--checkpointing_steps={min(profile['checkpointing_steps'], steps)}",
            "--resume_from_checkpoint=latest",
            f"--seed={train_seed(m, seed)}",
            f"--report_to={os.environ.get('GRASP_REPORT_TO', 'tensorboard')}",
        ]
        if profile.get("checkpoints_total_limit"):
            args.append(f"--checkpoints_total_limit={profile['checkpoints_total_limit']}")
        # skip_reached: a stage whose checkpoint already reached max_train_steps is not resumed again, so a
        # training run split into several Slurm jobs (train_repeats) never trains past its schedule
        cmds.append(dict(env={"OUTPUT_DIR": str(ckpt)}, skip_reached=(ckpt, steps),
                         cmd=container(accelerate_prefix(profile) + ["train_flux.py"] + args), mkdir=[ckpt]))
    return cmds


def train_repeats(m, profile):
    """Number of chained, resumable Slurm jobs a training run needs to fit the profile's walltime."""
    hours = m.get("train_hours", 12) * profile.get("time_factor", 1.0)
    return max(1, math.ceil(hours / (0.9 * P.slurm_hours(profile))))


def checkpoint_for(method, m, seed, profile, out):
    base = m["base"] or method
    bm = R.get(base)
    last = P.scaled_stages(bm["stages"], profile)[-1][0]
    return run_dir(out, "checkpoints", bm, base, seed) / f"checkpoint-{last}"


def build_generate(method, m, seed, profile, data, out, seed_mode, fix_emb, noise_seed=R.PAPER_NOISE_SEED):
    gdir = run_dir(out, "generated", m, method, seed, variant(m, seed_mode, fix_emb, noise_seed))
    if fix_emb and not m.get("emb_gen_fixed"):
        print(f"note: {method} already generates from the test-prompt embeddings", file=sys.stderr)
    emb = (m["emb_gen_fixed"] if (fix_emb and m.get("emb_gen_fixed")) else m["emb_gen"]).format(data=data)
    base_gen = R.MIMIC_GEN if m["dataset"] == "mimic-lt" else R.IMAGENET_GEN
    args = fmt(base_gen, data) + fmt(m["gen"], data) + [
        f"--checkpoint={checkpoint_for(method, m, seed, profile, out)}",
        f"--output={gdir}",
        f"--h5_path={gdir}/all_images.h5",
        f"--batch_size={m['gen_batch']}",
        f"--num_images={profile['num_images']}",
        f"--precomputed_embeddings_path={emb}",
        f"--seed={noise_seed}",   # noise seed; the generator seed only selects the trained model
        f"--seed_mode={seed_mode}",
    ]
    return [dict(env={}, cmd=container(accelerate_prefix(profile, 29501) + ["evaluation/generate_images.py"] + args),
                 mkdir=[gdir])]


def build_metrics(method, m, seed, profile, data, out, seed_mode, fix_emb, noise_seed=R.PAPER_NOISE_SEED):
    var = variant(m, seed_mode, fix_emb, noise_seed)
    gdir = run_dir(out, "generated", m, method, seed, var)
    mdir = run_dir(out, "metrics", m, method, seed, var)
    script = "evaluation/beyond_fid.py" if m["dataset"] == "mimic-lt" else "evaluation/beyond_fid_imagenet.py"
    cache = Path(out) / "feature_cache" / m["dataset"]   # real-image features are computed once and reused
    args = [f"--gen_path={gdir}/all_images.h5", f"--results_filename={mdir}/{method}_k10.json",
            f"--output_path={cache}", f"--data_root={data}", f"--num_synth={profile['num_images']}"]
    return [dict(env={}, cmd=container(["python", script] + args, "GRASP_METRICS_CONTAINER"), mkdir=[mdir, cache],
                 single_gpu=True)]


def build_classify(method, m, seed, profile, data, out, seed_mode, fix_emb, noise_seed=R.PAPER_NOISE_SEED):
    if m["dataset"] != "mimic-lt":
        return []
    if method == "real":
        cdir = Path(out) / "classifier" / "mimic-lt" / "real" / "real" / "seed0"
        train = [f"--train_roots={data}/mimic-lt/mimic-lt_test.h5"]
    else:
        var = variant(m, seed_mode, fix_emb, noise_seed)
        cdir = run_dir(out, "classifier", m, method, seed, var)
        train = [f"--train_roots={run_dir(out, 'generated', m, method, seed, var)}/all_images.h5"]
    args = train + [
        f"--train_csvs={data}/mimic-lt/mimic-cxr-lt_test.csv",
        f"--eval_h5={data}/nih-cxr-lt/classifier_eval.h5",
        f"--output_dir={cdir}", "--model=densenet121", f"--epochs={profile['cls_epochs']}",
        f"--batch_size={profile['cls_batch']}", "--num_workers=8", "--lr=3e-4", "--weight_decay=1e-4", "--image_size=512",
        "--seeds", *map(str, profile["cls_seeds"]), "--hflip", f"--accum_steps={profile['cls_accum']}"]
    return [dict(env={}, cmd=container(accelerate_prefix(profile, 29502) + ["downstream/classifier_ddp.py"] + args),
                 mkdir=[cdir])]


REAL = dict(dataset="mimic-lt", train=None, base=None, seeds=(0,), seed_mode="per_sample")


def plan(method, stage, seed, profile_name, seed_mode=None, fix_emb=False, noise_seed=None):
    profile = P.get(profile_name)
    data, out = env_paths(profile_name)
    if method == "real":
        return [("classify", build_classify("real", REAL, 0, profile, data, out, None, False))]
    m = R.get(method)
    seed_mode = seed_mode or m["seed_mode"]
    noise_seed = R.PAPER_NOISE_SEED if noise_seed is None else noise_seed
    stages = STAGES if stage == "all" else STAGES[1:] if stage == "eval" else (stage,)
    jobs = []
    for st in stages:
        if st == "train":
            jobs.append((st, build_train(method, m, seed, profile, data, out)))
        elif st == "generate":
            jobs.append((st, build_generate(method, m, seed, profile, data, out, seed_mode, fix_emb, noise_seed)))
        elif st == "metrics":
            jobs.append((st, build_metrics(method, m, seed, profile, data, out, seed_mode, fix_emb, noise_seed)))
        elif st == "classify":
            jobs.append((st, build_classify(method, m, seed, profile, data, out, seed_mode, fix_emb, noise_seed)))
    return [(st, cmds) for st, cmds in jobs if cmds]


def as_script(cmds, nodes):
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(REPO))}",
             f"export PYTHONPATH={shlex.quote(str(REPO))}${{PYTHONPATH:+:$PYTHONPATH}}",
             'export MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-dev}"', "export HF_HUB_DISABLE_XET=1",
             "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]
    if nodes > 1:
        lines.append('export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)')
    for c in cmds:
        for d in c["mkdir"]:
            lines.append(f"mkdir -p {shlex.quote(str(d))}")
        for k, v in c["env"].items():
            lines.append(f"export {k}={shlex.quote(v)}")
        if c.get("skip_reached"):
            d, steps = c["skip_reached"]
            lines.append(f'latest=$(ls -d {shlex.quote(str(d))}/checkpoint-* 2>/dev/null | sed "s/.*checkpoint-//" '
                         f'| sort -n | tail -1 || true)')
            lines.append(f'if [ "${{latest:-0}}" -ge {steps} ]; then echo "checkpoint-$latest >= {steps}: stage done"; '
                         f'else {q(c["cmd"])}; fi')
        else:
            lines.append(q(c["cmd"]))
    return "\n".join(lines) + "\n"


def sbatch_header(name, profile, log_dir):
    gpus = profile["gpus_per_node"]
    gtype = os.environ.get("GRASP_GPU_TYPE")
    h = ["#!/usr/bin/env bash", f"#SBATCH --job-name={name}", f"#SBATCH --nodes={profile['nodes']}",
         "#SBATCH --ntasks-per-node=1", f"#SBATCH --gres=gpu:{gtype + ':' if gtype else ''}{gpus}",
         f"#SBATCH --time={profile['slurm_time']}", f"#SBATCH --output={log_dir}/%x.%j.out"]
    h += [f"#SBATCH {a}" for a in shlex.split(os.environ.get("GRASP_SLURM_ARGS", ""))]
    return "\n".join(h) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list methods and profiles")
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--method", required=True)
        p.add_argument("--stage", default="all", choices=STAGES + ("all", "eval"))
        p.add_argument("--seed", type=int, action="append", help="generator seed(s); default: the paper seeds")
        p.add_argument("--profile", default="paper", choices=list(P.PROFILES))
        p.add_argument("--seed_mode", choices=["per_sample", "per_batch"],
                       help="override the noise layout of the reported row")
        p.add_argument("--noise_seed", type=int,
                       help="base noise seed of generation (default: 0, as in every paper run; the generator seed "
                            "--seed only selects the trained model)")
        p.add_argument("--fix_gen_embeddings", action="store_true",
                       help="generate from the test-prompt embeddings (docs/GENERATION_EMBEDDINGS.md)")
        if name == "run":
            p.add_argument("--launcher", default="local", choices=["local", "slurm"])
            p.add_argument("--dry_run", action="store_true", help="write the job scripts but do not execute them")
            p.add_argument("--train_repeats", type=int,
                           help="slurm: submit the train stage as N chained, resumable jobs (default: from the "
                                "method's train_hours, the profile's time_factor and its walltime)")
            p.add_argument("--after", help="slurm: job id the first submitted stage depends on (e.g. the base "
                                           "training job of a generation-only method)")
    a = ap.parse_args()

    if a.cmd == "list":
        print("Methods:")
        for k in sorted(R.METHODS):
            m = R.METHODS[k]
            kind = "gen-only from " + m["base"] if m["train"] is None else "train"
            print(f"  {k:32s} {m['dataset']:12s} {kind:28s} {'; '.join(m['rows'])}")
        print("  real                             mimic-lt     classifier on real images")
        print("Profiles:", ", ".join(P.PROFILES))
        return

    profile = P.get(a.profile)
    if a.cmd == "run" and a.launcher == "local" and profile["nodes"] > 1 and not a.dry_run:
        unsupported(f"profile '{a.profile}' spans {profile['nodes']} nodes; use --launcher slurm, "
                    "or --profile single_gpu for one machine.")
    if a.method == "real":
        seeds = [0]
    else:
        m = R.get(a.method)
        seeds = a.seed or list(profile["seeds"] or m["seeds"])
    _, out = env_paths(a.profile)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    for seed in seeds:
        jobs = plan(a.method, a.stage, seed, a.profile, a.seed_mode, a.fix_gen_embeddings, a.noise_seed)
        prev_job = getattr(a, "after", None)  # slurm: each stage waits for the previous one (afterok)
        mode = "real" if a.method == "real" else variant(R.get(a.method), a.seed_mode or R.get(a.method)["seed_mode"],
                                                          a.fix_gen_embeddings,
                                                          R.PAPER_NOISE_SEED if a.noise_seed is None else a.noise_seed)
        for st, cmds in jobs:
            tag = f"{a.method}_{st}_s{seed}_{mode}"
            prof_st = dict(profile, nodes=1, gpus_per_node=1) if any(c.get("single_gpu") for c in cmds) else profile
            if a.cmd == "plan":
                print(f"# ---- {tag} ({a.profile})")
                print(as_script(cmds, prof_st["nodes"]))
                continue
            jdir = out / "jobs"
            if a.launcher == "slurm" and any(c.isspace() for c in str(jdir)):
                sys.exit(f"GRASP_OUT must not contain whitespace for Slurm (#SBATCH --output cannot be quoted): {jdir}")
            jdir.mkdir(parents=True, exist_ok=True)
            # every submission gets its own immutable script (queued jobs never see a later overwrite)
            script = jdir / f"{tag}.{stamp}.sh"
            script.write_text(as_script(cmds, prof_st["nodes"] if a.launcher == "slurm" else 1))
            script.chmod(0o755)
            wrapped = None
            if a.launcher == "slurm":
                # one task per node; each node starts its own accelerate process (machine_rank = $SLURM_NODEID)
                wrapped = script.with_suffix(".sbatch")
                wrapped.write_text(sbatch_header(tag, prof_st, jdir) + f"srun bash {shlex.quote(str(script))}\n")
            if a.dry_run:
                print(f"wrote {script}" + (f" and {wrapped}" if wrapped else ""))
            elif a.launcher == "slurm":
                n_sub = (a.train_repeats or train_repeats(R.get(a.method), profile)) if st == "train" else 1
                for rep in range(n_sub):
                    # repeats of a training run wait for the previous part however it ended (walltime)
                    kind = "afterany" if rep > 0 else "afterok"
                    dep = [f"--dependency={kind}:{prev_job}"] if prev_job else []
                    proc = subprocess.run(["sbatch", "--parsable", *dep, str(wrapped)], capture_output=True, text=True)
                    if proc.returncode != 0:
                        sys.exit(f"sbatch rejected {wrapped}:\n{proc.stderr.strip()}\n(cluster options go into "
                                 "GRASP_SLURM_ARGS / GRASP_GPU_TYPE; some clusters require whole nodes for multi-node jobs)")
                    res = proc.stdout.strip()
                    part = f" [part {rep + 1}/{n_sub}]" if n_sub > 1 else ""
                    print(f"submitted {tag}{part}: job {res.split(';')[0]}" + (f" ({kind} {prev_job})" if dep else ""))
                    prev_job = res.split(";")[0]
            else:
                print(f"running {script}")
                subprocess.run(["bash", str(script)], check=True)


if __name__ == "__main__":
    main()
