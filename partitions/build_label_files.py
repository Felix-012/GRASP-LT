#!/usr/bin/env python3
"""Write per-sample routing files (one expert id per CSV row) from a label->expert map.

Every MIMIC-CXR-LT row has exactly one finding, so a partition is fully specified by
a map from finding name to expert id. `maps.json` stores the exact maps used in the
paper ("submitted" = the GRASP partition; the others are the sensitivity variants).

Examples
  # the GRASP partition for training and generation
  python partitions/build_label_files.py --map submitted \
      --csv $DATA/mimic-lt/mimic-cxr-lt_train.csv --out $DATA/mimic-lt/labels_gt.txt
  python partitions/build_label_files.py --map submitted \
      --csv $DATA/mimic-lt/mimic-cxr-lt_test.csv  --out $DATA/mimic-lt/test/labels_gt.txt

  # recover a map from an existing routing file
  python partitions/build_label_files.py --derive_from labels_gt.txt --csv mimic-cxr-lt_train.csv
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_findings(csv_path, column):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or column not in rows[0]:
        raise SystemExit(f"column '{column}' not found in {csv_path}")
    return [r[column] for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="MIMIC-CXR-LT split CSV (train or test)")
    ap.add_argument("--column", default="Finding Labels")
    ap.add_argument("--maps", default=str(HERE / "maps.json"))
    ap.add_argument("--map", help="name of the map in maps.json (e.g. submitted, random0, k3, hmt)")
    ap.add_argument("--out", help="output routing file")
    ap.add_argument("--derive_from", help="instead of writing, derive the label->expert map from this routing file")
    ap.add_argument("--trailing_newline", action="store_true",
                    help="end the file with a newline (the paper's test-split variant files have one)")
    args = ap.parse_args()

    findings = read_findings(args.csv, args.column)

    if args.derive_from:
        ids = Path(args.derive_from).read_text().split()
        if len(ids) != len(findings):
            raise SystemExit(f"{args.derive_from}: {len(ids)} ids vs {len(findings)} CSV rows")
        seen = collections.defaultdict(set)
        for f, e in zip(findings, ids):
            seen[f].add(int(e))
        split = {f: sorted(v) for f, v in seen.items() if len(v) > 1}
        if split:
            raise SystemExit(f"not a whole-label partition, labels split across experts: {split}")
        print(json.dumps({f: next(iter(v)) for f, v in sorted(seen.items())}, indent=1))
        return

    if not (args.map and args.out):
        ap.error("--map and --out are required unless --derive_from is given")
    maps = json.loads(Path(args.maps).read_text())
    if args.map not in maps:
        raise SystemExit(f"unknown map '{args.map}'; available: {sorted(maps)}")
    m = maps[args.map]
    missing = sorted(set(findings) - set(m))
    if missing:
        raise SystemExit(f"labels without an expert in map '{args.map}': {missing}")
    ids = [str(m[f]) for f in findings]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The original files were written with "\n".join(...) (no trailing newline).
    out.write_text("\n".join(ids) + ("\n" if args.trailing_newline else ""))
    counts = collections.Counter(int(i) for i in ids)
    print(f"wrote {out} ({len(ids)} rows); samples per expert: {dict(sorted(counts.items()))}")
    print(f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
