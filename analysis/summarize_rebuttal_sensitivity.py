#!/usr/bin/env python3
"""Summarize partition-sensitivity metrics with the paper's aggregation rules.

The published GRASP-LowDim row is read from the paper-input CSVs:
  * all-label mean +/- population SD across three generator seeds;
  * macro mean +/- population SD across the 19 label-wise means.

Rebuttal variants have one generator seed each. Their all-label values are
therefore point estimates, while their macro +/- remains population SD across
the same 19 labels. The random-x3 summary averages corresponding label values
across the three assignments before applying the label-macro aggregation.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRIC_ROOT = ROOT / "local_results" / "rebuttal_partition_metrics"
PAPER_RUN = "grasp_lowdim"
VARIANTS = ("random0", "random1", "random2", "k3", "k5", "oneperlabel", "hmt", "hybrid")
DISPLAY = {
    "paper_grasp_lowdim": "Informed (paper GRASP-LowDim)",
    "ccua_10k": "CCUA-10k",
    "ccua_20k": "CCUA-20k",
    "random_mean": "Random ×3 mean",
    "random0": "Random 0",
    "random1": "Random 1",
    "random2": "Random 2",
    "k3": "K=3",
    "k5": "K=5",
    "oneperlabel": "One expert per label",
    "hmt": "Literal H/M/T",
    "hybrid": "Static-initialized learned gate",
}
EXPERTS = {
    "paper_grasp_lowdim": 4,
    "ccua_10k": "—",
    "ccua_20k": "—",
    "random_mean": 4,
    "random0": 4,
    "random1": 4,
    "random2": 4,
    "k3": 3,
    "k5": 5,
    "oneperlabel": 19,
    "hmt": 3,
    "hybrid": 4,
}
CSV_METRICS = {
    "coverage": "prdc_coverage_test",
    "irs": "irs_adjusted",
    "fid": "fid_test",
}
# The retained CXR paper-input CSV rounds this cell to .014±.001. Preserve the
# author-verified unrounded value used in the rebuttal evidence audit.
PAPER_ALL_OVERRIDES = {
    ("cxr", "fid"): (0.01394, 0.00086),
}


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def mean_part(value: str) -> float:
    return float(value.split("±", 1)[0])


def parse_pm(value: str) -> tuple[float, float]:
    mean, sd = value.split("±", 1)
    return float(mean), float(sd)


def paper_metrics(extractor: str) -> dict:
    overall_path = ROOT / f"{extractor}_overall.csv"
    labels_path = ROOT / f"{extractor}_labels.csv"
    with overall_path.open(newline="") as handle:
        row = next(r for r in csv.DictReader(handle) if r["run"] == PAPER_RUN)
    all_label = {name: parse_pm(row[column]) for name, column in CSV_METRICS.items()}
    for metric in CSV_METRICS:
        if (extractor, metric) in PAPER_ALL_OVERRIDES:
            all_label[metric] = PAPER_ALL_OVERRIDES[(extractor, metric)]

    with labels_path.open(newline="") as handle:
        label_rows = [r for r in csv.DictReader(handle) if r["run"] == PAPER_RUN]
    if len(label_rows) != 19:
        raise ValueError(f"{labels_path}: expected 19 {PAPER_RUN} rows, got {len(label_rows)}")
    label_values = {
        name: {r["label"]: mean_part(r[column]) for r in label_rows}
        for name, column in CSV_METRICS.items()
    }
    macro = {name: mean_sd(list(values.values())) for name, values in label_values.items()}
    return {"all": all_label, "macro": macro, "labels": label_values}


def variant_metrics(variant: str, extractor: str) -> dict:
    paths = list((METRIC_ROOT / variant).glob("*.json"))
    if len(paths) != 1:
        raise ValueError(f"{variant}: expected one JSON, found {paths}")
    payload = json.loads(paths[0].read_text())[extractor]

    labels = sorted(
        key[len("label=") : -len("_irs_adjusted")]
        for key in payload
        if key.startswith("label=") and key.endswith("_irs_adjusted")
    )
    if len(labels) != 19:
        raise ValueError(f"{paths[0]}: expected 19 labels, got {len(labels)}")

    all_label = {
        "coverage": (float(payload["overall_prdc_test"]["coverage"]), None),
        "irs": (float(payload["overall_irs_adjusted"]), None),
        "fid": (float(payload["overall_fid_test"]["fid"]), None),
    }
    label_values = {
        "coverage": {
            label: float(payload[f"label={label}_prdc_test"]["coverage"]) for label in labels
        },
        "irs": {label: float(payload[f"label={label}_irs_adjusted"]) for label in labels},
        "fid": {label: float(payload[f"label={label}_fid_test"]["fid"]) for label in labels},
    }
    macro = {name: mean_sd(list(values.values())) for name, values in label_values.items()}
    return {"all": all_label, "macro": macro, "labels": label_values}


def multi_json_metrics(paths: list[Path], extractor: str) -> dict:
    if len(paths) != 3:
        raise ValueError(f"Expected three generator-seed JSONs, got {paths}")
    seed_rows = []
    for path in sorted(paths):
        payload = json.loads(path.read_text())[extractor]
        labels = sorted(
            key[len("label=") : -len("_irs_adjusted")]
            for key in payload
            if key.startswith("label=") and key.endswith("_irs_adjusted")
        )
        if len(labels) != 19:
            raise ValueError(f"{path}: expected 19 labels, got {len(labels)}")
        seed_rows.append(
            {
                "all": {
                    "coverage": float(payload["overall_prdc_test"]["coverage"]),
                    "irs": float(payload["overall_irs_adjusted"]),
                    "fid": float(payload["overall_fid_test"]["fid"]),
                },
                "labels": {
                    "coverage": {
                        label: float(payload[f"label={label}_prdc_test"]["coverage"])
                        for label in labels
                    },
                    "irs": {
                        label: float(payload[f"label={label}_irs_adjusted"]) for label in labels
                    },
                    "fid": {
                        label: float(payload[f"label={label}_fid_test"]["fid"]) for label in labels
                    },
                },
            }
        )

    all_label = {
        metric: mean_sd([row["all"][metric] for row in seed_rows]) for metric in CSV_METRICS
    }
    label_values = {
        metric: {
            label: sum(row["labels"][metric][label] for row in seed_rows) / len(seed_rows)
            for label in seed_rows[0]["labels"][metric]
        }
        for metric in CSV_METRICS
    }
    macro = {metric: mean_sd(list(values.values())) for metric, values in label_values.items()}
    return {"all": all_label, "macro": macro, "labels": label_values}


def aggregate_random(rows: list[dict]) -> dict:
    all_label = {}
    for metric in CSV_METRICS:
        all_label[metric] = mean_sd([row["all"][metric][0] for row in rows])

    label_values = {}
    for metric in CSV_METRICS:
        labels = rows[0]["labels"][metric]
        label_values[metric] = {
            label: sum(row["labels"][metric][label] for row in rows) / len(rows)
            for label in labels
        }
    macro = {name: mean_sd(list(values.values())) for name, values in label_values.items()}
    return {"all": all_label, "macro": macro, "labels": label_values}


def fmt(metric: str, pair: tuple[float, float | None], *, with_sd: bool = True) -> str:
    value, sd = pair
    digits = 2 if metric == "fid" and value > 1 else 3
    if metric == "fid" and value < 1:
        digits = 5
    if sd is None or not with_sd:
        return f"{value:.{digits}f}"
    sd_digits = 2 if metric == "fid" and value > 1 else digits
    return f"{value:.{digits}f} ± {sd:.{sd_digits}f}"


def write_markdown(results: dict[str, dict[str, dict]], path: Path) -> None:
    order = (
        "paper_grasp_lowdim",
        "ccua_10k",
        "ccua_20k",
        "random_mean",
        "random0",
        "random1",
        "random2",
        "k3",
        "k5",
        "oneperlabel",
        "hmt",
        "hybrid",
    )
    lines = [
        "# Partition-sensitivity metrics, paper-compatible aggregation",
        "",
        "The informed anchor is the published GRASP-LowDim result (three generator "
        "seeds), not a separately computed rebuttal anchor. For that row and both "
        "CCUA rows, all-label `±` is population SD over generator seeds. Every "
        "individual partition variant uses generator seed 0, so its all-label result "
        "is a point estimate. For all "
        "rows, macro `±` is population SD across the 19 label-wise values, exactly as "
        "in the paper. For `Random ×3 mean`, all-label `±` is across partition "
        "assignments; its label-wise values are first averaged across assignments and "
        "then macro-aggregated.",
        "",
    ]
    for extractor, title in (("dinov2", "DINOv2"), ("cxr", "CXR")):
        lines += [
            f"## {title}",
            "",
            "| Partition | K | All Cov ↑ | All IRS ↑ | All FID ↓ | Macro Cov ↑ | Macro IRS ↑ | Macro FID ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key in order:
            row = results[key][extractor]
            all_cells = [fmt(metric, row["all"][metric]) for metric in ("coverage", "irs", "fid")]
            macro_cells = [fmt(metric, row["macro"][metric]) for metric in ("coverage", "irs", "fid")]
            lines.append(
                f"| {DISPLAY[key]} | {EXPERTS[key]} | "
                + " | ".join(all_cells + macro_cells)
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def write_csv(results: dict[str, dict[str, dict]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        fieldnames = ["variant", "display", "experts", "extractor", "scope", "metric", "mean", "sd", "sd_source"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant, by_extractor in results.items():
            for extractor, row in by_extractor.items():
                for scope in ("all", "macro"):
                    for metric, (mean, sd) in row[scope].items():
                        if scope == "macro":
                            source = "19 labels"
                        elif variant in ("paper_grasp_lowdim", "ccua_10k", "ccua_20k"):
                            source = "3 generator seeds"
                        elif variant == "random_mean":
                            source = "3 partition assignments"
                        else:
                            source = "single generator seed (no SD)"
                        writer.writerow(
                            {
                                "variant": variant,
                                "display": DISPLAY[variant],
                                "experts": EXPERTS[variant],
                                "extractor": extractor,
                                "scope": scope,
                                "metric": metric,
                                "mean": f"{mean:.12g}",
                                "sd": "" if sd is None else f"{sd:.12g}",
                                "sd_source": source,
                            }
                        )


def main() -> None:
    results: dict[str, dict[str, dict]] = {
        "paper_grasp_lowdim": {
            extractor: paper_metrics(extractor) for extractor in ("dinov2", "cxr")
        }
    }
    ccua_sources = {
        "ccua_10k": sorted(
            (ROOT / "data" / "grasp_v2" / "metrics" / "no_resample" / "mimic-lt").glob(
                "seed*/ccua/ccua-mimic_k10.json"
            )
        ),
        "ccua_20k": sorted(
            (ROOT / "local_results" / "ccua_long_metrics").glob(
                "seed*/ccua_long-mimic_k10.json"
            )
        ),
    }
    for variant, paths in ccua_sources.items():
        results[variant] = {
            extractor: multi_json_metrics(paths, extractor) for extractor in ("dinov2", "cxr")
        }
    for variant in VARIANTS:
        results[variant] = {
            extractor: variant_metrics(variant, extractor) for extractor in ("dinov2", "cxr")
        }
    results["random_mean"] = {
        extractor: aggregate_random(
            [results[variant][extractor] for variant in ("random0", "random1", "random2")]
        )
        for extractor in ("dinov2", "cxr")
    }

    output_root = ROOT / "local_results" / "rebuttal_partition_metrics"
    write_markdown(results, output_root / "paper_comparable_sensitivity.md")
    write_csv(results, output_root / "paper_comparable_sensitivity.csv")
    print(output_root / "paper_comparable_sensitivity.md")


if __name__ == "__main__":
    main()
