#!/usr/bin/env python3
"""Compare collected results with the numbers reported in the paper.

    python tools/collect_metrics.py && python tools/collect_classifier.py
    python tools/compare_to_paper.py [--tolerance 2.0]

Covers the main results (Tables 1-3, expected/paper_values.csv), which the run files in runs/ reproduce.

For each reported value with a reported SD, the reproduction is flagged when it deviates by more than
--tolerance reported SDs (and by more than the reported rounding). Values are matched per method; by
default the settings of each reported row are used (reproduce/registry.py).
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce import registry as R  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def load(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def decimals(s):
    return len(s.split(".")[1]) if "." in s else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("GRASP_OUT", "outputs"))
    ap.add_argument("--profile", default="paper", help="profile folder under --out (reproduce/profiles.py)")
    ap.add_argument("--seed_mode", default=None)
    ap.add_argument("--tolerance", type=float, default=2.0)
    a = ap.parse_args()
    a.out = str(Path(a.out).expanduser().resolve() / a.profile)
    t = Path(a.out) / "tables"
    metrics = {(r["dataset"], r["method"], r["seed_mode"], r["feature"]): r for r in load(t / "metrics_summary.csv")}
    cls = {(r["model"], r["seed_mode"]): r for r in load(t / "classifier_summary.csv")}
    shown = flagged = 0
    for e in load(HERE / "expected" / "paper_values.csv"):
        method = e["method"]
        mode = a.seed_mode or (R.METHODS[method]["seed_mode"] if method in R.METHODS else "per_sample")
        got = None
        if e["table"] in ("T1", "T2"):
            r = metrics.get((e["dataset"], method, mode, e["feature"]))
            got = float(r[e["metric"]]) if r else None
        elif e["table"] == "T3":
            r = cls.get((method, "real" if method == "real" else mode))
            if r:
                got = float(r["macro_f1"] if e["metric"] == "macro_f1" else r["nz"])
        if got is None:
            continue
        if got != got:                       # NaN: the value could not be computed (e.g. too few images)
            shown += 1
            flagged += 1
            print(f"!! {e['table']} {method:24s} {e['feature']:10s} {e['metric']:10s} paper {e['value']:>8s} reproduced NaN")
            continue
        want = float(e["value"])
        sdv = float(e["sd"]) if e["sd"] else 0.0
        slack = max(a.tolerance * sdv, 0.5 * 10 ** -decimals(e["value"]))
        bad = abs(got - want) > slack
        shown += 1
        flagged += bad
        print(f"{'!!' if bad else 'ok'} {e['table']} {method:24s} {e['feature']:10s} {e['metric']:10s} "
              f"paper {e['value']:>8s}{'±' + e['sd'] if e['sd'] else '':8s} reproduced {got:.4g}")
    print(f"\n{shown} values compared, {flagged} outside tolerance" if shown else "no matching results yet")


if __name__ == "__main__":
    main()
