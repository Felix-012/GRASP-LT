#!/usr/bin/env python3
"""Collect BeyondFID results into one CSV per feature space, aggregated like the paper tables.

Reads  $GRASP_OUT/<profile>/metrics/<dataset>/<method>/<seed_mode>/seed<S>/<method>_k10.json
Writes $GRASP_OUT/<profile>/tables/metrics_runs.csv     (one row per run and feature space)
       $GRASP_OUT/<profile>/tables/metrics_summary.csv  (mean over generator seeds)

Aggregation (as in the paper):
  all-label metrics: mean over generator seeds; population SD over seeds ("_sd"; "_sd_sample" also given)
  label-macro metrics: mean over the labels of the seed-averaged label values; SD across labels
  ("_sd_labels") and population SD across seeds of the per-seed macro ("_sd_seeds").
  The paper reports population SDs; macro SDs are across labels, except for the MoE-GRASP and +MG rows
  (across seeds).
BeyondFID writes -1 when a value cannot be computed (e.g. too few samples of a label); such values are
treated as missing and counted in "n_invalid".
"""
import argparse
import csv
import json
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

FEATURES = ("cxr", "dinov2")


def valid(x):
    return x is not None and x == x and x >= 0      # BeyondFID uses -1 for "not computable"; drop NaN too


def label_values(sec, suffix, field=None):
    """Valid label-wise values {label: value} and the number of invalid (sentinel / missing) entries."""
    out, n_bad = {}, 0
    for k, v in sec.items():
        if k.startswith("label=") and k.endswith(suffix):
            name = k[len("label="):-len(suffix)]
            try:
                val = float(v[field] if field else v)
            except (TypeError, KeyError, ValueError):
                n_bad += 1
                continue
            if valid(val):
                out[name] = val
            else:
                n_bad += 1
    return out, n_bad


def overall(sec, key, field=None):
    try:
        val = float(sec[key][field] if field else sec[key])
    except (TypeError, KeyError, ValueError):
        return None
    return val if valid(val) else None


def per_run(path):
    d = json.loads(Path(path).read_text())
    rows = []
    for feat in FEATURES:
        if feat not in d:
            continue
        s = d[feat]
        (lc, ic), (li, ii), (lf, if_) = (label_values(s, "_prdc_test", "coverage"),
                                         label_values(s, "_irs_adjusted"), label_values(s, "_fid_test", "fid"))
        all_vals = [overall(s, "overall_prdc_test", "coverage"), overall(s, "overall_irs_adjusted"),
                    overall(s, "overall_fid_test", "fid")]
        rows.append(dict(feature=feat, all_cov=all_vals[0], all_irs=all_vals[1], all_fid=all_vals[2],
                         lab_cov=lc, lab_irs=li, lab_fid=lf,
                         n_invalid=ic + ii + if_ + sum(v is None for v in all_vals)))
    return rows


def sd(x):
    return st.pstdev(x) if len(x) > 1 else 0.0


def sd_sample(x):
    return st.stdev(x) if len(x) > 1 else 0.0


def mean(x):
    return st.mean(x) if x else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("GRASP_OUT", "outputs"))
    ap.add_argument("--profile", default="paper", help="profile folder under --out (reproduce/profiles.py)")
    a = ap.parse_args()
    a.out = str(Path(a.out).expanduser().resolve() / a.profile)
    root = Path(a.out) / "metrics"
    runs = defaultdict(list)  # (dataset, method, seed_mode, feature) -> list of per-seed dicts
    run_rows = []
    for f in sorted(root.glob("*/*/*/seed*/*_k10.json")):
        dataset, method, seed_mode, seed = f.parts[-5], f.parts[-4], f.parts[-3], f.parts[-2]
        for r in per_run(f):
            runs[(dataset, method, seed_mode, r["feature"])].append(r)
            run_rows.append(dict(dataset=dataset, method=method, seed_mode=seed_mode, seed=seed[4:],
                                 feature=r["feature"], all_cov=r["all_cov"], all_irs=r["all_irs"], all_fid=r["all_fid"],
                                 macro_cov=mean(list(r["lab_cov"].values())), macro_irs=mean(list(r["lab_irs"].values())),
                                 macro_fid=mean(list(r["lab_fid"].values())),
                                 n_labels=min(len(r["lab_cov"]), len(r["lab_irs"]), len(r["lab_fid"]))))
    tdir = Path(a.out) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    if not run_rows:
        raise SystemExit(f"no metric files under {root}")
    with open(tdir / "metrics_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(run_rows[0]))
        w.writeheader()
        w.writerows(run_rows)

    summary = []
    for (dataset, method, seed_mode, feat), rs in sorted(runs.items()):
        row = dict(dataset=dataset, method=method, seed_mode=seed_mode, feature=feat, n_seeds=len(rs))
        for m in ("cov", "irs", "fid"):
            vals = [r[f"all_{m}"] for r in rs if r[f"all_{m}"] is not None]
            row[f"all_{m}"], row[f"all_{m}_sd"], row[f"all_{m}_sd_sample"] = mean(vals), sd(vals), sd_sample(vals)
            labels = sorted(set.intersection(*(set(r[f"lab_{m}"]) for r in rs)))
            per_label = [st.mean(r[f"lab_{m}"][l] for r in rs) for l in labels]
            per_seed_macro = [st.mean(r[f"lab_{m}"][l] for l in labels) for r in rs] if labels else []
            row[f"macro_{m}"] = mean(per_label)
            row[f"macro_{m}_sd_labels"] = sd(per_label)
            row[f"macro_{m}_sd_seeds"] = sd(per_seed_macro)
            row[f"macro_{m}_sd_seeds_sample"] = sd_sample(per_seed_macro)
            row[f"n_labels_{m}"] = len(labels)
        row["n_invalid"] = sum(r["n_invalid"] for r in rs)   # invalid entries over all seeds and labels
        summary.append(row)
    with open(tdir / "metrics_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {tdir/'metrics_runs.csv'} ({len(run_rows)} rows) and {tdir/'metrics_summary.csv'} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
