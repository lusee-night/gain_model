#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import pandas as pd

# --- Config ---
CPT_LIST_FILE = os.path.expanduser("~/gain_model/scripts/CPT_directories_withCPT1.txt")
OUTPUT_CSV = os.path.expanduser("~/gain_model/outputs/other/means/gain_column_means_by_CPT.csv")


def extract_cpt_label(path: str) -> str:
    """
    From a path like .../CPT12+Science11/... return 'CPT12'.
    """
    m = re.search(r"/(CPT\d+)[^/]*/", path)
    if not m:
        # try a simpler fallback: segment containing 'CPT' then strip extras
        parts = path.split('/')
        for seg in parts:
            if seg.startswith('CPT'):
                n = re.match(r"(CPT\d+)", seg)
                if n:
                    return n.group(1)
        raise ValueError(f"Could not extract CPT label from path: {path}")
    return m.group(1)


def main():
    # Read the list of CPT directories
    if not os.path.isfile(CPT_LIST_FILE):
        print(f"ERROR: CPT list file not found: {CPT_LIST_FILE}", file=sys.stderr)
        sys.exit(1)

    with open(CPT_LIST_FILE, 'r') as f:
        dirs = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]

    rows = []
    labels = []
    missing = []
    bad = []

    for d in dirs:
        try:
            label = extract_cpt_label(d)
        except Exception as e:
            bad.append((d, str(e)))
            continue

        gain_path = os.path.join(os.path.expanduser(d), "gain.dat")
        if not os.path.isfile(gain_path):
            missing.append(gain_path)
            continue

        try:
            # Load gain.dat; it's whitespace/tab separated and has a header.
            df = pd.read_csv(gain_path, delim_whitespace=True, engine='python')
            # Normalize column names (strip potential BOM/whitespace)
            df.columns = [c.strip() for c in df.columns]
            if 'freq' not in df.columns:
                raise ValueError(f"'freq' column not found in {gain_path}. Columns: {df.columns.tolist()}")

            # Compute means for all columns except freq
            cols_to_avg = [c for c in df.columns if c.lower() != 'freq']
            means = df[cols_to_avg].mean(numeric_only=True)
            rows.append(means)
            labels.append(label)
        except Exception as e:
            bad.append((gain_path, str(e)))

    if not rows:
        print("ERROR: No data rows were produced. Check 'missing' and 'bad' lists printed below.", file=sys.stderr)
        if missing:
            print("\nMissing gain.dat files:", file=sys.stderr)
            for p in missing:
                print(p, file=sys.stderr)
        if bad:
            print("\nFiles with errors:", file=sys.stderr)
            for p, msg in bad:
                print(f"{p} -> {msg}", file=sys.stderr)
        sys.exit(2)

    # Build output DataFrame
    out = pd.DataFrame(rows, index=labels)

    # Sort rows by numeric CPT value
    def cpt_key(lbl):
        m = re.match(r"CPT(\d+)$", lbl)
        return int(m.group(1)) if m else 10**9

    out = out.sort_index(key=lambda idx: [cpt_key(x) for x in idx])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out.index.name = "CPT"
    out.to_csv(OUTPUT_CSV)

    print(f"Wrote {OUTPUT_CSV}")
    if missing:
        print("\nNOTE: These gain.dat files were missing:", file=sys.stderr)
        for p in missing:
            print(p, file=sys.stderr)
    if bad:
        print("\nNOTE: These files/paths had errors:", file=sys.stderr)
        for p, msg in bad:
            print(f"{p} -> {msg}", file=sys.stderr)


if __name__ == "__main__":
    main()
