#!/usr/bin/env python3
"""Create deterministic test routing files from audited train partitions.

Each MIMIC-CXR-LT row has one positive condition.  A valid GRASP partition keeps
all rows for a condition in one expert.  We therefore derive the condition map
from each train routing file and apply it to the test CSV.  Before writing any
ablation files, the same operation must reproduce the deployed GRASP test
routing exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


NON_LABEL_COLUMNS = {
    "",
    "subject_id",
    "study_id",
    "dicom_id",
    "path",
    "Finding Labels",
    "impression",
    "Split",
    "FileName",
}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        label_columns = [
            name for name in reader.fieldnames if name not in NON_LABEL_COLUMNS
        ]
        rows = list(reader)
    return label_columns, rows


def condition(row: dict[str, str], label_columns: list[str]) -> str:
    positives = []
    for name in label_columns:
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            continue
        if value == 1.0:
            positives.append(name)
    if len(positives) != 1:
        raise ValueError(
            f"Expected one positive condition, found {positives}; "
            f"dicom_id={row.get('dicom_id')}"
        )
    return positives[0]


def read_labels(path: Path) -> list[int]:
    return [int(value) for value in path.read_text().split()]


def derive_map(
    rows: list[dict[str, str]],
    label_columns: list[str],
    assignments: list[int],
) -> dict[str, int]:
    if len(rows) != len(assignments):
        raise ValueError(
            f"Length mismatch: {len(rows)} CSV rows vs "
            f"{len(assignments)} assignments"
        )
    observed: dict[str, set[int]] = defaultdict(set)
    for row, expert in zip(rows, assignments):
        observed[condition(row, label_columns)].add(expert)
    split = {name: values for name, values in observed.items() if len(values) != 1}
    if split:
        raise ValueError(f"Conditions split across experts: {split}")
    return {name: next(iter(values)) for name, values in observed.items()}


def expand(
    rows: list[dict[str, str]],
    label_columns: list[str],
    mapping: dict[str, int],
) -> list[int]:
    missing = sorted(
        {
            condition(row, label_columns)
            for row in rows
            if condition(row, label_columns) not in mapping
        }
    )
    if missing:
        raise ValueError(f"Test conditions absent from train map: {missing}")
    return [mapping[condition(row, label_columns)] for row in rows]


def digest(values: list[int]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--default-train-labels", type=Path, required=True)
    parser.add_argument("--default-test-labels", type=Path, required=True)
    parser.add_argument("--partition-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Writable destination; defaults to PARTITION_DIR/test.",
    )
    args = parser.parse_args()

    train_columns, train_rows = read_rows(args.train_csv)
    test_columns, test_rows = read_rows(args.test_csv)
    if train_columns != test_columns:
        raise ValueError("Train/test label columns differ")

    default_map = derive_map(
        train_rows, train_columns, read_labels(args.default_train_labels)
    )
    reproduced_default = expand(test_rows, test_columns, default_map)
    retained_default = read_labels(args.default_test_labels)
    if reproduced_default != retained_default:
        mismatches = sum(
            left != right
            for left, right in zip(reproduced_default, retained_default)
        )
        raise ValueError(
            f"Default test-routing reproduction failed: {mismatches} mismatches"
        )

    output_dir = args.output_dir or (args.partition_dir / "test")
    output_dir.mkdir(parents=True, exist_ok=True)
    maps: dict[str, dict[str, int]] = {}
    report: dict[str, dict[str, object]] = {}
    for train_path in sorted(args.partition_dir.glob("labels_*.txt")):
        name = train_path.stem.removeprefix("labels_")
        mapping = derive_map(train_rows, train_columns, read_labels(train_path))
        test_assignments = expand(test_rows, test_columns, mapping)
        output_path = output_dir / train_path.name
        output_path.write_text("".join(f"{value}\n" for value in test_assignments))
        maps[name] = dict(sorted(mapping.items()))
        report[name] = {
            "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "test_sha256": digest(test_assignments),
            "num_experts": len(set(mapping.values())),
            "test_loads": dict(sorted(Counter(test_assignments).items())),
            "test_path": str(output_path),
        }

    (output_dir / "maps_verified.json").write_text(
        json.dumps(maps, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "test_partition_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Verified default routing on {len(test_rows)} test rows; "
        f"wrote {len(report)} ablation test partitions to {output_dir}"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
