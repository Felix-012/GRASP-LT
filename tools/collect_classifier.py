#!/usr/bin/env python3
"""Collect downstream classifier results (best_metrics.json) into one CSV.

Reads  $GRASP_OUT/<profile>/classifier/mimic-lt/<method>/<seed_mode>/seed<S>/**/seed_<c>/best_metrics.json
       $GRASP_OUT/<profile>/classifier/mimic-lt/real/real/**/seed_<c>/best_metrics.json
Writes $GRASP_OUT/<profile>/tables/classifier_runs.csv   one row per (method, generator seed, classifier seed),
                                               columns as used by analysis/*.py
       $GRASP_OUT/<profile>/tables/classifier_summary.csv macro-F1 mean/SD (population SD over runs) and #nz,
                                               the number of classes with nonzero mean F1 (Tab. 3)
"""
import argparse
import csv
import json
import os
import re
import statistics as st
from collections import defaultdict
from pathlib import Path


def norm(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("GRASP_OUT", "outputs"))
    ap.add_argument("--profile", default="paper", help="profile folder under --out (reproduce/profiles.py)")
    a = ap.parse_args()
    a.out = str(Path(a.out).expanduser().resolve() / a.profile)
    root = Path(a.out) / "classifier" / "mimic-lt"
    rows = []
    for f in sorted(root.glob("**/best_metrics.json")):
        rel = f.relative_to(root).parts
        method, seed_mode = rel[0], rel[1]
        gseed = next((p[4:] for p in rel if p.startswith("seed") and not p.startswith("seed_")), "0")
        cseed = next((p for p in rel if p.startswith("seed_")), "seed_000")
        d = json.loads(f.read_text())
        agg, pc = d["aggregates"], d["per_class"]
        r = dict(model=method, seed_mode=seed_mode, group=f"seed{gseed}", seed=cseed, macro_f1=agg["macro_f1"],
                 balanced_acc=agg.get("balanced_acc"), json_path=str(f))
        for label, v in sorted(pc.items()):
            k = norm(label)
            r[f"{k}_support"], r[f"{k}_pred"], r[f"{k}_f1"] = v.get("support"), v.get("pred"), v["f1"]
        rows.append(r)
    if not rows:
        raise SystemExit(f"no best_metrics.json under {root}")
    tdir = Path(a.out) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(tdir / "classifier_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["seed_mode"])].append(r)
    summary = []
    for (model, mode), rs in sorted(by.items()):
        f1cols = [c for c in fields if c.endswith("_f1") and c != "macro_f1"]
        means = {c: st.mean(float(r[c]) for r in rs if r.get(c) not in (None, "")) for c in f1cols if any(r.get(c) not in (None, "") for r in rs)}
        mf = [float(r["macro_f1"]) for r in rs]
        summary.append(dict(model=model, seed_mode=mode, runs=len(rs), macro_f1=st.mean(mf),
                            macro_f1_sd=st.pstdev(mf) if len(mf) > 1 else 0.0,
                            nz=sum(v > 0 for v in means.values()), n_classes=len(means),
                            no_finding_f1=means.get("no_finding_f1")))
    with open(tdir / "classifier_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {tdir/'classifier_runs.csv'} ({len(rows)} runs) and {tdir/'classifier_summary.csv'}")


if __name__ == "__main__":
    main()
