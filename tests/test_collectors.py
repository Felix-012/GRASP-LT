import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def fake_run(path, fid_all, labels):
    sec = {"overall_prdc_test": {"coverage": 0.9}, "overall_irs_adjusted": 0.5, "overall_fid_test": {"fid": fid_all}}
    for name, (cov, irs, fid) in labels.items():
        sec[f"label={name}_prdc_test"] = {"coverage": cov}
        sec[f"label={name}_irs_adjusted"] = irs
        sec[f"label={name}_fid_test"] = {"fid": fid}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cxr": sec, "dinov2": sec}))


def test_sentinels_are_ignored_and_sds_follow_the_paper(tmp_path):
    root = tmp_path / "paper" / "metrics" / "mimic-lt" / "grasp_lowdim" / "per_batch"
    fake_run(root / "seed0" / "grasp_lowdim_k10.json", 10.0, {"A": (0.5, 0.4, 100.0), "B": (-1, -1, -1)})
    fake_run(root / "seed5868" / "grasp_lowdim_k10.json", 12.0, {"A": (0.7, 0.6, 120.0), "B": (0.2, 0.2, 50.0)})
    subprocess.run([sys.executable, str(REPO / "tools" / "collect_metrics.py"), "--out", str(tmp_path)], check=True)
    row = next(csv.DictReader(open(tmp_path / "paper" / "tables" / "metrics_summary.csv")))
    assert float(row["all_fid"]) == 11.0 and abs(float(row["all_fid_sd"]) - 1.0) < 1e-9      # population SD
    assert row["n_labels_fid"] == "1" and int(row["n_invalid"]) >= 3                        # label B dropped
    assert float(row["macro_fid"]) == 110.0


def test_label_invalid_in_every_seed_is_counted(tmp_path):
    root = tmp_path / "paper" / "metrics" / "mimic-lt" / "grasp_lowdim" / "per_batch"
    for s, v in (("seed0", 10.0), ("seed1", 12.0)):
        fake_run(root / s / "grasp_lowdim_k10.json", v, {"A": (0.5, 0.4, 100.0), "B": (-1, -1, -1)})
    subprocess.run([sys.executable, str(REPO / "tools" / "collect_metrics.py"), "--out", str(tmp_path)], check=True)
    row = next(csv.DictReader(open(tmp_path / "paper" / "tables" / "metrics_summary.csv")))
    assert int(row["n_invalid"]) == 6      # label B: 3 metrics x 2 seeds
