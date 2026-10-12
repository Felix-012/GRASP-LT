"""Every flag emitted by the CLI must exist in the argparse definition of its target script."""
import os
import re
from pathlib import Path

import pytest

from reproduce import cli, profiles, registry

REPO = Path(__file__).resolve().parents[1]


def accepted(path):
    src = (REPO / path).read_text()
    return set(re.findall(r"add_argument\(\s*['\"](--[\w-]+)['\"]", src))


TARGETS = {
    "train_flux.py": accepted("utils/train_parser.py"),
    "evaluation/generate_images.py": accepted("evaluation/generate_images.py"),
    "evaluation/beyond_fid.py": accepted("evaluation/beyond_fid.py"),
    "evaluation/beyond_fid_imagenet.py": accepted("evaluation/beyond_fid_imagenet.py"),
    "downstream/classifier_ddp.py": accepted("downstream/classifier_ddp.py"),
}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("GRASP_DATA", "/data")
    monkeypatch.setenv("GRASP_OUT", str(tmp_path))


@pytest.mark.parametrize("method", sorted(registry.METHODS) + ["real"])
def test_flags_exist(method):
    for _, cmds in cli.plan(method, "all", 0, "paper"):
        for c in cmds:
            script = next(t for t in c["cmd"] if t.endswith(".py"))
            for tok in c["cmd"][c["cmd"].index(script) + 1:]:
                if tok.startswith("--"):
                    flag = tok.split("=")[0]
                    assert flag in TARGETS[script], f"{method}: {flag} not accepted by {script}"


@pytest.mark.parametrize("method", sorted(registry.METHODS))
def test_generation_reads_the_trained_checkpoint(method):
    m = registry.METHODS[method]
    gen = dict(cli.plan(method, "generate", 0, "paper"))["generate"][0]["cmd"]
    ckpt = next(t for t in gen if t.startswith("--checkpoint=")).split("=", 1)[1]
    base = m["base"] or method
    last = profiles.scaled_stages(registry.METHODS[base]["stages"], profiles.get("paper"))[-1][0]
    assert ckpt.endswith(f"/{base}/seed0/checkpoint-{last}")


def test_ccua_two_stages_resume_in_one_folder():
    stages = dict(cli.plan("ccua", "train", 0, "paper"))["train"]
    outs = {next(t for t in s["cmd"] if t.startswith("--output_dir=")) for s in stages}
    steps = [next(t for t in s["cmd"] if t.startswith("--max_train_steps=")) for s in stages]
    assert len(outs) == 1 and steps == ["--max_train_steps=10000", "--max_train_steps=20000"]


def test_legacy_training_seed():
    cmd = dict(cli.plan("grasp_highdim", "train", 0, "paper"))["train"][0]["cmd"]
    assert "--seed=4200" in cmd
    cmd = dict(cli.plan("grasp_lowdim", "train", 0, "paper"))["train"][0]["cmd"]
    assert "--seed=0" in cmd


def test_eval_stage_skips_training():
    assert [st for st, _ in cli.plan("moe_grasp_lowdim", "eval", 0, "paper")] == ["generate", "metrics", "classify"]


def test_long_training_is_split_into_resumable_jobs():
    assert cli.train_repeats(registry.get("ft_class"), profiles.get("paper")) == 2
    assert cli.train_repeats(registry.get("ccua"), profiles.get("paper")) == 2
    assert cli.train_repeats(registry.get("grasp_lowdim"), profiles.get("paper")) == 1
    assert cli.train_repeats(registry.get("grasp_lowdim"), profiles.get("smoke")) == 1


def test_training_stages_skip_when_already_reached(tmp_path):
    import subprocess
    train = dict(cli.plan("ccua", "train", 0, "paper"))["train"]
    script = cli.as_script(train, 1)
    assert script.count("stage done") == 2 and "--checkpoints_total_limit=3" in script
    # the guard runs under `set -euo pipefail` with and without existing checkpoints
    ckpt = train[0]["skip_reached"][0]
    guard = [l for l in script.splitlines() if l.startswith("latest=")][0]
    for made in ([], ["checkpoint-10000"]):
        for d in made:
            (tmp_path / d).mkdir()
        g = guard.replace(str(ckpt), str(tmp_path))
        out = subprocess.run(["bash", "-c", f"set -euo pipefail; {g}; echo ${{latest:-0}}"],
                             capture_output=True, text=True, check=True).stdout.split()
        assert out[-1] == ("10000" if made else "0")


def test_generation_noise_seed_is_fixed_like_the_paper():
    gen = dict(cli.plan("grasp_lowdim", "generate", 5868, "paper"))["generate"][0]["cmd"]
    assert "--seed=0" in gen and any("seed5868/checkpoint-10000" in t for t in gen)
    other = dict(cli.plan("grasp_lowdim", "generate", 5868, "paper", noise_seed=5868))["generate"][0]["cmd"]
    assert "--seed=5868" in other and any("per_batch-noise5868" in t for t in other)


def test_seed_mode_groups():
    assert registry.METHODS["ccua"]["seed_mode"] == "per_batch"
    assert all(m["seed_mode"] == "per_sample" for k, m in registry.METHODS.items() if m["dataset"] == "imagenet-lt")


def test_seed_modes():
    assert registry.METHODS["grasp_lowdim"]["seed_mode"] == "per_batch"
    assert registry.METHODS["moe_grasp_lowdim"]["seed_mode"] == "per_batch"
    assert registry.METHODS["grasp_lowdim_mg"]["seed_mode"] == "per_sample"
    gen = dict(cli.plan("grasp_lowdim", "generate", 0, "paper", seed_mode="per_sample"))["generate"][0]["cmd"]
    assert "--seed_mode=per_sample" in gen


def test_outputs_are_separated_by_profile_and_variant(tmp_path):
    a = dict(cli.plan("ft_class", "generate", 0, "paper"))["generate"][0]["cmd"]
    b = dict(cli.plan("ft_class", "generate", 0, "paper", fix_emb=True))["generate"][0]["cmd"]
    c = dict(cli.plan("grasp_lowdim", "generate", 0, "reduced"))["generate"][0]["cmd"]
    out = lambda cmd: next(t for t in cmd if t.startswith("--output=")).split("=", 1)[1]
    assert out(a).endswith("/paper/generated/mimic-lt/ft_class/per_batch/seed0")
    assert out(b).endswith("/paper/generated/mimic-lt/ft_class/per_batch-testemb/seed0")
    assert "/reduced/" in out(c)
    assert any("embeddings_mimic-lt_labels_test.h5" in t for t in b)
    assert all(Path(out(x)).is_absolute() for x in (a, b, c))


def test_local_multi_node_is_refused(monkeypatch):
    monkeypatch.setattr("sys.argv", ["cli", "run", "--method", "grasp_lowdim", "--profile", "paper", "--launcher", "local"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == cli.EXIT_UNSUPPORTED
