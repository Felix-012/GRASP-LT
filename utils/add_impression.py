#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path
import pandas as pd


# -------- Path handling --------
def path_to_report_txt(base_path: Path, rel_path: str) -> Path:
    """
    From a 'path' column value, remove everything from the last slash onward
    and replace it with '.txt', then join to base_path.

    Examples:
      'a/b/c/dicom_123.dcm' -> '<base>/a/b/c/.txt'
      'a\\b\\c\\file.png'   -> '<base>/a/b/c/.txt'
    """
    if rel_path is None or str(rel_path).strip() == "" or str(rel_path).lower() == "nan":
        return (base_path / ".txt").resolve()

    p = str(rel_path).replace("\\", "/")
    prefix = p.rsplit("/", 1)[0] if "/" in p else ""
    transformed = (prefix + ".txt").lstrip("/") if prefix else ".txt"
    return (base_path / transformed).resolve()


# -------- Section extraction --------
def _compile_section_start(header: str) -> re.Pattern:
    # e.g., header="IMPRESSION" or "FINDINGS"
    return re.compile(rf"(?mi)^\s*{header}s?\s*:?\s*(.*)$")

_NEXT_HEADING_RE = re.compile(r"(?m)^\s*[A-Z][A-Z0-9 /()\-\.,]{2,}\s*:?\s*$")

def extract_section(text: str, header: str) -> str | None:
    """
    Extract a section starting at `header` (IMPRESSION/FINDINGS), until next heading or EOF.
    Allows inline content on the header line.
    """
    start_re = _compile_section_start(header)
    m = start_re.search(text)
    if not m:
        return None

    start_idx = m.end(0)
    inline_after_colon = m.group(1).strip()

    collected = inline_after_colon + ("\n" if inline_after_colon else "")
    m_next = _NEXT_HEADING_RE.search(text, pos=start_idx)
    end_idx = m_next.start() if m_next else len(text)

    body_after = text[start_idx:end_idx]
    section = (collected + body_after).strip()
    section = re.sub(r"\n{3,}", "\n\n", section)
    return section if section else None


# -------- Main workflow --------
def main():
    ap = argparse.ArgumentParser(
        description="Fill missing 'impression' cells in a CSV by parsing report .txt files; fallback to FINDINGS; finally to 'Finding Labels'."
    )
    ap.add_argument("input_csv", help="Input CSV with columns: impression, path, Finding Labels.")
    ap.add_argument("base_path", help="Base directory to prefix transformed 'path' to locate report .txt.")
    ap.add_argument("-o", "--output-csv", help="Output CSV (default: overwrite input).", default=None)
    ap.add_argument("--encoding", default="utf-8", help="Encoding for CSV and reports (default: utf-8).")
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    base_path = Path(args.base_path)

    if not input_csv.exists():
        print(f"ERROR: input CSV not found: {input_csv}", file=sys.stderr)
        sys.exit(1)
    if not base_path.exists():
        print(f"ERROR: base_path not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_csv, dtype=str, encoding=args.encoding)
    # Case-insensitive column lookup
    cols = {c.lower(): c for c in df.columns}

    required_cols = ["impression", "path", "finding labels"]
    for rc in required_cols:
        if rc not in cols:
            print(f"ERROR: CSV must contain '{rc}' column (case-insensitive).", file=sys.stderr)
            sys.exit(1)

    imp_col = cols["impression"]
    path_col = cols["path"]
    labels_col = cols["finding labels"]

    # Normalize nulls
    df[imp_col] = df[imp_col].astype(str)
    null_mask = df[imp_col].isna() | (df[imp_col].str.strip().eq("")) | (df[imp_col].str.lower().eq("nan"))

    filled_from_impression = 0
    filled_from_findings = 0
    filled_from_labels = 0
    missing_files = 0
    unreadable_files = 0

    for idx in df[null_mask].index:
        rel = df.at[idx, path_col]
        report_path = path_to_report_txt(base_path, rel)

        text = None
        if report_path.exists():
            try:
                text = report_path.read_text(encoding=args.encoding, errors="ignore")
            except Exception as e:
                print(f"WARNING: could not read '{report_path}': {e}", file=sys.stderr)
                unreadable_files += 1
        else:
            missing_files += 1

        value = None
        if text:
            # 1) IMPRESSION
            value = extract_section(text, "IMPRESSION")
            if value:
                filled_from_impression += 1
            else:
                # 2) FINDINGS
                value = extract_section(text, "FINDINGS")
                if value:
                    filled_from_findings += 1

        # 3) Fallback to Finding Labels
        if not value:
            labels_val = df.at[idx, labels_col]
            # Use as-is (even if it's empty, we follow the instruction)
            value = "" if labels_val is None else str(labels_val)
            filled_from_labels += 1

        df.at[idx, imp_col] = value

    out_path = Path(args.output_csv) if args.output_csv else input_csv
    df.to_csv(out_path, index=False, encoding=args.encoding)

    print(
        f"Done. Wrote: {out_path}\n"
        f"  Rows needing fill: {int(null_mask.sum())}\n"
        f"  Filled from IMPRESSION: {filled_from_impression}\n"
        f"  Filled from FINDINGS: {filled_from_findings}\n"
        f"  Filled from Finding Labels: {filled_from_labels}\n"
        f"  Missing report files: {missing_files}\n"
        f"  Unreadable report files: {unreadable_files}"
    )


if __name__ == "__main__":
    main()
