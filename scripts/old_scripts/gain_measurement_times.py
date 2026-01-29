#!/usr/bin/env python3

"""
Extract gain measurement timestamps from commander logs.

What it does:
  - Parses commander.log files for each CPT.
  - Tracks gain-setting changes and AWG tone commands.
  - Records the time, gain setting (L/M/H), and tone frequency
    whenever a nonzero-frequency tone is issued.
  - Writes a detailed CSV per CPT with gain measurement timing.

Inputs:
  - CPT directory list:
      ~/uncrater/scripts/CPT_directories.txt
  - Output directory list (paired with CPTs):
      ~/uncrater/scripts/output_directories.txt
  - For each CPT directory:
      commander.log

Outputs:
  - For each output directory:
      gain_measurement_times_detailed.csv

How to run:
  python3 scripts/gain_measurement_times.py

Notes:
  - This script assumes CPT_directories.txt and output_directories.txt
    have matching order and length.
  - Output directories can be updated manually to point under
    ~/uncrater/outputs/... if desired.
"""

import os
import re
import csv


# ---------------------------
# Path configuration
# ---------------------------

cpt_list_path = os.path.expanduser("~/uncrater/scripts/CPT_directories.txt")
out_list_path = os.path.expanduser("~/uncrater/scripts/output_directories.txt")


# ---------------------------
# Load CPT / output directories
# ---------------------------

with open(cpt_list_path, 'r') as f:
    cpt_dirs = [os.path.expanduser(line.strip()) for line in f if line.strip()]

with open(out_list_path, 'r') as f:
    out_dirs = [os.path.expanduser(line.strip()) for line in f if line.strip()]

if len(cpt_dirs) != len(out_dirs):
    raise ValueError("Mismatch between CPT and output directory counts")


# ---------------------------
# Gain decoding configuration
# ---------------------------

# Maps analog gain-setting arguments to labels
gain_decode = {
    0: 'L',
    85: 'M',
    170: 'H'
}


# ---------------------------
# Regex patterns for log parsing
# ---------------------------

# Timestamp pattern: "[ 123s ]"
time_pattern = re.compile(r"\[\s*(\d+)s\s*\]")

# AWG tone command pattern
tone_pattern = re.compile(r"AWG command: TONE\s+(\d)\s+([\d.]+)\s+([\d.]+)")

# Gain-setting command pattern
gain_set_pattern = re.compile(r"RFS_SET_GAIN_ANA_SET with argument (\d+)")


# ---------------------------
# Main parsing loop
# ---------------------------

for cpt_dir, out_dir in zip(cpt_dirs, out_dirs):
    log_path = os.path.join(cpt_dir, "commander.log")
    output_csv = os.path.join(out_dir, "gain_measurement_times_detailed.csv")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(log_path):
        print(f"[WARNING] Log file not found: {log_path}")
        continue

    current_gain = None
    current_freq = None
    records = []

    with open(log_path, 'r') as f:
        for line in f:
            # Extract timestamp (seconds since start)
            time_match = time_pattern.search(line)
            timestamp = int(time_match.group(1)) if time_match else None

            # Update current gain setting when command is seen
            gain_match = gain_set_pattern.search(line)
            if gain_match:
                arg = int(gain_match.group(1))
                current_gain = gain_decode.get(arg, f"UNKNOWN_{arg}")

            # Capture AWG tone commands
            tone_match = tone_pattern.search(line)
            if tone_match:
                ch, freq, amp = tone_match.groups()
                freq = float(freq)

                # Record only nonzero-frequency tones with valid context
                if freq != 0 and timestamp is not None and current_gain is not None:
                    records.append((timestamp, current_gain, freq))

    # ---------------------------
    # Write CSV output
    # ---------------------------

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time_sec', 'gain_setting', 'frequency_Hz'])
        for row in records:
            writer.writerow(row)

    print(f"[SUCCESS] Wrote {len(records)} detailed gain times to {output_csv}")
