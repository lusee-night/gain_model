#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script aggregates decoded DCB telemetry by analog gain setting (L/M/H × channel 0-3)
across multiple CPT runs.

For each CPT:
  • The script parses commander.log to identify time windows where TONE measurements
    are active for a given gain setting and channel.
  • Only the first start time and last end time of each (gain, channel) measurement
    window are retained per CPT.
  • These wall-clock times (seconds) are mapped to telemetry packet indices assuming
    a fixed packet cadence (default: 30 s).

For each gain setting (L0-H3):
  • The corresponding packet range is sliced from the decoded DCB telemetry CSV.
  • Mean values are computed for all numeric telemetry fields over that interval.
  • One row per CPT is written, preserving CPT order as listed in DCB_directories.txt.

Outputs:
  • Twelve CSV files (one per gain setting) containing per-CPT mean telemetry values:
        telemetry_per_gainsetting/{L0,M0,H0,...,H3}.csv
  • A second pass is executed including CPT1, producing:
        telemetry_per_gainsetting_withCPT1/{setting}_withCPT1.csv

Key assumptions:
  • Telemetry packets arrive at a constant cadence.
  • commander.log timestamps are authoritative for measurement timing.

This output is intended for downstream gain modeling and regression analysis,
where telemetry statistics must be aligned precisely with gain-setting intervals.
"""


import os
import re
import csv
import numpy as np
import pandas as pd

# -------- Config --------
CPT_LIST_PATH = os.path.expanduser("~/gain_model/scripts/CPT_directories.txt")
DCB_LIST_PATH = os.path.expanduser("~/gain_model/scripts/DCB_directories.txt")
PACKET_PERIOD_SEC = 30.0
LOG_FILENAME = "commander.log"
TELEM_FILENAME = "DCB_telemetry_decoded.csv"

OUTPUT_ROOT = os.path.expanduser(
    "~/gain_model/outputs/telemetry_per_gainsetting"
)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

ALL_SETTINGS = [f"{g}{ch}" for g in ("L", "M", "H") for ch in range(4)]

# Gain decode (from ANA_SET argument → label)
GAIN_DECODE = {0: "L", 85: "M", 170: "H"}

# -------- Regex --------
time_pattern = re.compile(r"\[\s*(\d+)s\s*\]")
# TONE <channel:int> <frequency_MHz:float> <measure_flag:float>
float_re = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
tone_pattern = re.compile(
    rf"(?:Received\s+)?AWG command:\s*TONE\s+(\d+)\s+({float_re})\s+({float_re})"
)
gain_set_pattern = re.compile(r"RFS_SET_GAIN_ANA_SET with argument (\d+)")

# -------- Helpers --------
def round_to_packet_index(t_sec: int | float, period=PACKET_PERIOD_SEC) -> int:
    """Map wall-clock seconds to packet index by rounding to nearest."""
    return int(round(float(t_sec) / float(period)))

def parse_first_last_ranges(log_path: str) -> dict:
    """
    commander.log → { 'L0': {'first_start': s, 'last_end': e}, ... }.
    Starts a window when measure_flag != 0 for TONE ch under current gain.
    Ends window at next measurement start or gain change; aggregates first/last.
    """
    current_gain = None          # 'L'/'M'/'H'
    active = None                # {'setting': 'L0'..'H3', 'start': int}
    last_ts = None
    ranges = {}                  # setting -> {'first_start': int, 'last_end': int}

    def close_active(end_ts: int):
        nonlocal active, ranges
        if active is None or end_ts is None:
            return
        setting = active['setting']
        start = active['start']
        if start is None:
            active = None
            return
        if setting not in ranges:
            ranges[setting] = {'first_start': start, 'last_end': end_ts}
        else:
            if start < ranges[setting]['first_start']:
                ranges[setting]['first_start'] = start
            if end_ts > ranges[setting]['last_end']:
                ranges[setting]['last_end'] = end_ts
        active = None

    with open(log_path, 'r') as f:
        for line in f:
            m = time_pattern.search(line)
            if m:
                last_ts = int(m.group(1))

            g = gain_set_pattern.search(line)
            if g:
                arg = int(g.group(1))
                new_gain = GAIN_DECODE.get(arg, f"UNKNOWN_{arg}")
                if active is not None:
                    close_active(last_ts)
                current_gain = new_gain
                continue

            t = tone_pattern.search(line)
            if t:
                ch_str, _freq_mhz_str, flag_str = t.groups()
                ch = int(ch_str)
                measure_flag = float(flag_str)
                if current_gain is None or last_ts is None:
                    continue
                if measure_flag != 0.0:
                    if active is not None:
                        close_active(last_ts)
                    active = {'setting': f"{current_gain}{ch}", 'start': last_ts}

    if active is not None:
        close_active(last_ts)

    return ranges

def clamp_slice(df: pd.DataFrame, k_start: int, k_end: int) -> pd.DataFrame:
    """Inclusive slice [k_start, k_end] with clamping to DF bounds; empty → empty DF."""
    n = len(df)
    if n == 0:
        return df.iloc[0:0]
    k_start = max(0, min(int(k_start), n - 1))
    k_end   = max(0, min(int(k_end),   n - 1))
    if k_end < k_start:
        return df.iloc[0:0]
    return df.iloc[k_start : k_end + 1]

# -------- Main --------
def main():
    # Read directory lists
    with open(CPT_LIST_PATH, 'r') as f:
        cpt_dirs = [os.path.expanduser(s.strip()) for s in f if s.strip()]
    with open(DCB_LIST_PATH, 'r') as f:
        dcb_dirs = [os.path.expanduser(s.strip()) for s in f if s.strip()]

    if len(cpt_dirs) != len(dcb_dirs):
        raise ValueError("Mismatch: CPT_directories.txt and DCB_directories.txt lengths differ")

    # CPT labels exactly from DCB_directories.txt (order preserved)
    CPT_LABELS = [os.path.basename(os.path.normpath(p)) for p in dcb_dirs]
    CPT_INDEX = {lbl: i for i, lbl in enumerate(CPT_LABELS)}

    # Prepare per-setting aggregation: mapping setting -> rows (each row = [CPT, telemetry means...])
    per_setting_rows: dict[str, list[list]] = {s: [] for s in ALL_SETTINGS}
    telemetry_cols_global = None  # lock to numeric columns found in the first CSV

    for cpt_dir, dcb_dir in zip(cpt_dirs, dcb_dirs):
        log_path = os.path.join(cpt_dir, LOG_FILENAME)
        telem_path = os.path.join(dcb_dir, TELEM_FILENAME)
        cpt_label = os.path.basename(os.path.normpath(dcb_dir))  # strict from DCB list

        if not os.path.exists(log_path):
            print(f"[WARN] Missing log: {log_path}")
            continue
        if not os.path.exists(telem_path):
            print(f"[WARN] Missing telemetry: {telem_path}")
            continue

        # Parse first/last ranges for this CPT
        ranges = parse_first_last_ranges(log_path)
        if not ranges:
            print(f"[WARN] No measurement ranges found in {log_path}")
            continue

        # Load telemetry for this CPT
        try:
            df = pd.read_csv(telem_path)
        except Exception as e:
            print(f"[WARN] Failed to read telemetry CSV {telem_path}: {e}")
            continue

        # Lock telemetry column order to first file's numeric columns,
        # and on subsequent files keep their intersection (order from first).
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if telemetry_cols_global is None:
            telemetry_cols_global = numeric_cols
        else:
            telemetry_cols_global = [c for c in telemetry_cols_global if c in numeric_cols]

        # For each setting present in this CPT, compute means over the packet range and store row
        for setting, rr in ranges.items():
            if setting not in per_setting_rows:
                continue
            first_start = rr['first_start']
            last_end    = rr['last_end']

            k_start = round_to_packet_index(first_start)
            k_end   = round_to_packet_index(last_end)

            df_slice = clamp_slice(df, k_start, k_end)
            if len(df_slice) == 0 or telemetry_cols_global is None or len(telemetry_cols_global) == 0:
                continue

            means = df_slice[telemetry_cols_global].mean()
            means_row = [means.get(col, np.nan) for col in telemetry_cols_global]
            per_setting_rows[setting].append([cpt_label] + means_row)

    # Write 12 CSVs (one per setting) with header: CPT + telemetry columns
    if telemetry_cols_global is None or len(telemetry_cols_global) == 0:
        print("[WARN] No numeric telemetry columns discovered; nothing to write.")
        return

    header = ["CPT"] + telemetry_cols_global
    for setting in ALL_SETTINGS:
        rows = per_setting_rows.get(setting, [])
        # sort rows by DCB list order
        rows.sort(key=lambda r: CPT_INDEX.get(r[0], 10**9))  # r[0] is CPT label

        out_path = os.path.join(OUTPUT_ROOT, f"{setting}.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            if rows:
                w.writerows(rows)
        print(f"[OK] {setting}: wrote {len(rows)} CPT rows → {out_path}")

# -------- Second run: include CPT1 --------
def main_withCPT1():
    CPT_LIST_PATH_2 = os.path.expanduser("~/gain_model/scripts/CPT_directories_withCPT1.txt")
    DCB_LIST_PATH_2 = os.path.expanduser("~/gain_model/scripts/DCB_directories_withCPT1.txt")
    OUTPUT_ROOT_2 = os.path.expanduser(
        "~/gain_model/outputs/telemetry_per_gainsetting_withCPT1"
    )
    os.makedirs(OUTPUT_ROOT_2, exist_ok=True)

    # Reuse everything else by temporarily overriding globals
    global CPT_LIST_PATH, DCB_LIST_PATH, OUTPUT_ROOT
    old_CPT, old_DCB, old_OUT = CPT_LIST_PATH, DCB_LIST_PATH, OUTPUT_ROOT
    CPT_LIST_PATH, DCB_LIST_PATH, OUTPUT_ROOT = CPT_LIST_PATH_2, DCB_LIST_PATH_2, OUTPUT_ROOT_2

    # Run main()
    main()

    # Rename each exported file to have "_withCPT1"
    for setting in ALL_SETTINGS:
        src = os.path.join(OUTPUT_ROOT_2, f"{setting}.csv")
        dst = os.path.join(OUTPUT_ROOT_2, f"{setting}_withCPT1.csv")
        if os.path.exists(src):
            os.rename(src, dst)

    # Restore globals
    CPT_LIST_PATH, DCB_LIST_PATH, OUTPUT_ROOT = old_CPT, old_DCB, old_OUT


if __name__ == "__main__":
    main()          # Original run (without CPT1)
    main_withCPT1() # Second run (with CPT1)
