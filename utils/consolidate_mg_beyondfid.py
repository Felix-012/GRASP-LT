#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import csv
import json
import re
import statistics as stats
from pathlib import Path


METHODS = {"mg_class_cond", "mg_grasp_lowdim"}
FEATURES = ("dinov2", "cxr")


def fmean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def fsd(values: list[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


def label_macro(section: dict, suffix: str, field: str | None = None) -> tuple[float, int]:
    values = []
    for key, value in section.items():
        if not key.startswith("label=") or not key.endswith(suffix):
            continue
        if field is None and isinstance(value, int | float):
            values.append(float(value))
        elif field is not None and isinstance(value, dict) and field in value:
            values.append(float(value[field]))
    return fmean(values), len(values)


def parse_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("seed*/mg_*/*.json")):
        if path.parent.name not in METHODS:
            continue
        match = re.search(r"seed(\d+)", str(path))
        seed = int(match.group(1)) if match else -1
        method = path.parent.name
        with path.open() as handle:
            data = json.load(handle)

        for feature in FEATURES:
            section = data[feature]
            row = {"seed": seed, "method": method, "feature": feature, "json": str(path)}
            for split in ("train", "test"):
                row[f"fid_{split}"] = section[f"overall_fid_{split}"]["fid"]
                prdc = section[f"overall_prdc_{split}"]
                for key in ("precision", "recall", "density", "coverage"):
                    row[f"prdc_{split}_{key}"] = prdc[key]
            for name in ("test", "snth"):
                irs = section[f"overall_irs_{name}"]
                row[f"irs_{name}_alpha"] = irs["irs_alpha"]
                row[f"irs_{name}_inf"] = irs["irs_inf"]
            row["irs_adjusted"] = section["overall_irs_adjusted"]
            row["label_macro_fid_train"], row["n_labels_fid_train"] = label_macro(
                section, "_fid_train", "fid"
            )
            row["label_macro_fid_test"], row["n_labels_fid_test"] = label_macro(
                section, "_fid_test", "fid"
            )
            row["label_macro_irs_adjusted"], row["n_labels_irs"] = label_macro(
                section, "_irs_adjusted"
            )
            rows.append(row)
    return rows


def aggregate(rows: list[dict], metrics: list[str]) -> list[dict]:
    agg_rows = []
    for method in sorted({row["method"] for row in rows}):
        for feature in FEATURES:
            subset = [row for row in rows if row["method"] == method and row["feature"] == feature]
            if not subset:
                continue
            agg = {"method": method, "feature": feature, "n_seeds": len(subset)}
            for metric in metrics:
                values = [float(row[metric]) for row in subset]
                agg[f"{metric}_mean"] = fmean(values)
                agg[f"{metric}_std"] = fsd(values)
            agg_rows.append(agg)
    return agg_rows


def fmt_mean_std(row: dict, metric: str, digits: int = 3) -> str:
    return f"{row[f'{metric}_mean']:.{digits}f} +/- {row[f'{metric}_std']:.{digits}f}"


def print_summary(rows: list[dict], agg_rows: list[dict]) -> None:
    print("# Main Aggregate (mean +/- std over 3 seeds)")
    print(
        "method | feature | FID test | macro label FID test | "
        "PRDC test P/R/D/C | IRS adj | macro label IRS adj"
    )
    print("--- | --- | ---: | ---: | --- | ---: | ---:")
    for row in agg_rows:
        prdc = "/".join(
            fmt_mean_std(row, metric, 3)
            for metric in (
                "prdc_test_precision",
                "prdc_test_recall",
                "prdc_test_density",
                "prdc_test_coverage",
            )
        )
        print(
            f"{row['method']} | {row['feature']} | "
            f"{fmt_mean_std(row, 'fid_test', 2)} | "
            f"{fmt_mean_std(row, 'label_macro_fid_test', 2)} | "
            f"{prdc} | {fmt_mean_std(row, 'irs_adjusted', 3)} | "
            f"{fmt_mean_std(row, 'label_macro_irs_adjusted', 3)}"
        )

    print()
    print("# Per Seed Main Metrics")
    print("method | seed | feature | FID test | macro label FID test | PRDC test P/R/D/C | IRS adj")
    print("--- | ---: | --- | ---: | ---: | --- | ---:")
    for row in sorted(rows, key=lambda item: (item["method"], item["seed"], item["feature"])):
        prdc = (
            f"{row['prdc_test_precision']:.3f}/"
            f"{row['prdc_test_recall']:.3f}/"
            f"{row['prdc_test_density']:.3f}/"
            f"{row['prdc_test_coverage']:.3f}"
        )
        print(
            f"{row['method']} | {row['seed']} | {row['feature']} | "
            f"{row['fid_test']:.2f} | {row['label_macro_fid_test']:.2f} | "
            f"{prdc} | {row['irs_adjusted']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("GRASP_OUT", "outputs")) / "metrics" / "mimic-lt",
    )
    args = parser.parse_args()

    rows = parse_rows(args.root)
    if not rows:
        raise SystemExit(f"No MG BeyondFID JSONs found under {args.root}")

    columns = [
        "method",
        "seed",
        "feature",
        "fid_train",
        "fid_test",
        "label_macro_fid_train",
        "label_macro_fid_test",
        "prdc_train_precision",
        "prdc_train_recall",
        "prdc_train_density",
        "prdc_train_coverage",
        "prdc_test_precision",
        "prdc_test_recall",
        "prdc_test_density",
        "prdc_test_coverage",
        "irs_test_alpha",
        "irs_test_inf",
        "irs_snth_alpha",
        "irs_snth_inf",
        "irs_adjusted",
        "label_macro_irs_adjusted",
        "n_labels_fid_train",
        "n_labels_fid_test",
        "n_labels_irs",
        "json",
    ]
    run_csv = args.root / "mg_beyondfid_consolidated_runs.csv"
    with run_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda item: (item["method"], item["feature"], item["seed"])))

    metrics = [
        column
        for column in columns
        if column not in {"method", "seed", "feature", "json"} and not column.startswith("n_labels")
    ]
    agg_rows = aggregate(rows, metrics)
    agg_columns = ["method", "feature", "n_seeds"] + [
        name for metric in metrics for name in (f"{metric}_mean", f"{metric}_std")
    ]
    agg_csv = args.root / "mg_beyondfid_consolidated_aggregate.csv"
    with agg_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=agg_columns)
        writer.writeheader()
        writer.writerows(agg_rows)

    print(f"wrote {run_csv}")
    print(f"wrote {agg_csv}")
    print()
    print_summary(rows, agg_rows)


if __name__ == "__main__":
    main()
