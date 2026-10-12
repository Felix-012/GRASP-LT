#!/usr/bin/env python3
"""Compare files under $GRASP_DATA with the checksums of the files used for the paper (data/checksums.tsv).

CSV checksums only match if the CSVs were built byte-identically; routing/partition files must match.
"""
import argparse
import hashlib
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.environ.get("GRASP_DATA"))
    ap.add_argument("--table", default=str(HERE / "data" / "checksums.tsv"))
    a = ap.parse_args()
    if not a.data:
        raise SystemExit("set --data or GRASP_DATA")
    bad = 0
    for line in Path(a.table).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel, must = line.split("\t")
        p = Path(a.data) / rel
        if not p.exists():
            print(f"[missing] {rel}")
            bad += must == "required"
            continue
        ok = sha256(p) == digest
        print(f"[{'ok' if ok else 'DIFF'}] {rel}" + ("" if ok or must == "required" else "  (informational)"))
        bad += (not ok) and must == "required"
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
