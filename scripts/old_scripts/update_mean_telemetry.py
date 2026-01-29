#!/usr/bin/env python3
"""
update_mean_telemetry.py

Workflow
────────
1. Read CPT paths from  ~/uncrater/scripts/DCB_directories.txt
2. For each path expect  DCB_telemetry_decoded.csv
3. Compute per-column means
      • drop THERM_FPGA
      • ALWAYS keep THERM_DCB
      • keep any other column whose zero-fraction < 90 %
      • round every mean to 3 decimals
4. Read existing overall file  mean_TFPGA.csv
      • rename 'mean TFPGA' → 'TFPGA'
5. Merge new means
6. Drop telemetry columns whose means are “flat” across CPTs
      (max-min) / |overall mean| ≤ 0.001  (0.1 %)  or abs(max-min) ≤ 1e-6
7. Write augmented, reordered table to  means.csv  with columns
      CPT, TFPGA, <telemetry>, L0 … H3
"""

from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

# ─── paths & parameters ──────────────────────────────────────────────────
LIST_FILE   = Path("~/uncrater/scripts/DCB_directories.txt").expanduser()

INPUT_FILE  = Path("~/uncrater/data/plots/overall/mean_TFPGA.csv").expanduser()
OUTPUT_FILE = Path("~/uncrater/data/plots/overall/means.csv").expanduser()

DCB_NAME    = "DCB_telemetry_decoded.csv"
ZERO_THRESH = 0.90                       # ≥ 90 % zeros → drop (except ALWAYS_KEEP)
ALWAYS_KEEP = {"THERM_DCB"}              # never dropped by zero-fraction test
ROUND_DEC   = 3                          # decimals to round new means

FLAT_REL_TOL = 0.001                     # 0.1 %
FLAT_ABS_TOL = 1e-6                      # fallback if mean ≈ 0
# ─────────────────────────────────────────────────────────────────────────

GAIN_COLS = [
    "L0","M0","H0","L1","M1","H1",
    "L2","M2","H2","L3","M3","H3"
]


def gather_means() -> pd.DataFrame:
    """Return DataFrame: one row per CPT, rounded means of kept columns."""
    rows: list[dict[str, float]] = []

    if not LIST_FILE.is_file():
        sys.exit(f"Directory list not found: {LIST_FILE}")

    with LIST_FILE.open() as f:
        for line in f:
            dcb_dir = Path(os.path.expanduser(line.strip()))
            if not dcb_dir:
                continue
            csv_path = dcb_dir / DCB_NAME
            if not csv_path.is_file():
                print(f"[WARN] Missing {csv_path} – skipping", file=sys.stderr)
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"[WARN] Could not read {csv_path}: {exc}", file=sys.stderr)
                continue

            # Drop THERM_FPGA
            df = df.drop(columns=["THERM_FPGA"], errors="ignore")
            if "THERM_FPGA" not in df.columns and df.shape[1] > 0:
                df = df.iloc[:, 1:]     # positional fallback

            # Decide which columns to keep
            keep = []
            for col in df.columns:
                if col in ALWAYS_KEEP:
                    keep.append(col)
                elif (df[col] == 0).mean() < ZERO_THRESH:
                    keep.append(col)

            if not keep:
                continue

            means = df[keep].mean(numeric_only=True).round(ROUND_DEC)
            rows.append({"CPT": dcb_dir.name, **means.to_dict()})

    return pd.DataFrame(rows)


def drop_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove telemetry columns that are almost constant across CPT rows."""
    keep_cols = []
    for col in df.columns:
        vals = df[col].dropna()
        if vals.empty or col in ("CPT", "TFPGA") or col in GAIN_COLS:
            keep_cols.append(col)
            continue

        vmax, vmin = vals.max(), vals.min()
        spread = vmax - vmin
        mean_abs = vals.mean()
        if (mean_abs != 0 and spread / abs(mean_abs) > FLAT_REL_TOL) or spread > FLAT_ABS_TOL:
            keep_cols.append(col)

    return df[keep_cols]


def main() -> None:
    if not INPUT_FILE.is_file():
        sys.exit(f"Overall mean file not found: {INPUT_FILE}")

    overall = pd.read_csv(INPUT_FILE)

    # rename 'mean TFPGA' → 'TFPGA'
    for col in overall.columns:
        if col.strip().lower().replace(" ", "") == "meantfpga":
            overall = overall.rename(columns={col: "TFPGA"})
            break

    new_means = gather_means()
    if new_means.empty:
        print("No new means to merge – nothing written.")
        return

    merged = overall.merge(new_means, on="CPT", how="left", suffixes=("", "_new"))

    # integrate “…_new” columns
    for col in merged.filter(like="_new").columns:
        base = col[:-4]
        if base in merged.columns:
            merged[base] = merged[base].fillna(merged[col])
        else:
            merged.rename(columns={col: base}, inplace=True)
    merged.drop(columns=merged.filter(like="_new").columns, inplace=True)

    # Drop columns that are flat across CPTs
    pruned = drop_flat_columns(merged)

    # ─── reorder columns ────────────────────────────────────────────────
    static     = ["CPT", "TFPGA"]
    gains      = [c for c in GAIN_COLS if c in pruned.columns]
    telemetry  = [c for c in pruned.columns if c not in static + gains]
    pruned     = pruned[static + telemetry + gains]
    # ────────────────────────────────────────────────────────────────────

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pruned.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved updated means → {OUTPUT_FILE.relative_to(Path.home())}")


if __name__ == "__main__":
    main()
