#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
TELEMETRY_VARS = [
    "TFPGA", "SPE_ADC0_T", "SPE_ADC1_T",
    "PFPS_PA3_T", "PFPS_PA2_T", "PFPS_PA1_T", "PFPS_PA0_T"
]
GAIN_SETTINGS = [
    "L0", "M0", "H0", "L1", "M1", "H1",
    "L2", "M2", "H2", "L3", "M3", "H3"
]
COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink"
]

# --- File paths ---
MEANS_CSV = os.path.expanduser("~/uncrater/data/plots/overall/means.csv")
CPT_LIST_FILE = os.path.expanduser("~/uncrater/scripts/CPT_directories.txt")
OUTPUT_DIR = os.path.expanduser("~/uncrater/data/plots/overall/gain_vs_temp")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Clean output directory ---
for f in os.listdir(OUTPUT_DIR):
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(path):
        os.remove(path)

# --- Load data ---
means_df = pd.read_csv(MEANS_CSV)
with open(CPT_LIST_FILE) as f:
    cpt_dirs = [os.path.expanduser(line.strip()) for line in f if line.strip()]

if len(cpt_dirs) != len(means_df):
    raise ValueError("Mismatch: Number of CPT directories does not match rows in means.csv")

# --- Data container ---
gain_data = {}  # (gain_setting, freq): {"gain": [], var1: [], ...}
fit_coeffs = {gain: [] for gain in GAIN_SETTINGS}  # gain setting → list of fit rows

# --- Aggregate gain + telemetry data ---
for row_idx, cpt_dir in enumerate(cpt_dirs):
    gain_path = os.path.join(cpt_dir, "gain.dat")
    if not os.path.exists(gain_path):
        continue

    gain_df = pd.read_csv(gain_path, delim_whitespace=True)
    telemetry_row = means_df.iloc[row_idx]

    for gain in GAIN_SETTINGS:
        if gain not in gain_df.columns:
            continue
        for i, freq in enumerate(gain_df["freq"]):
            key = (gain, freq)
            if key not in gain_data:
                gain_data[key] = {var: [] for var in TELEMETRY_VARS}
                gain_data[key]["gain"] = []
            gain_val = gain_df[gain].iloc[i]
            gain_data[key]["gain"].append(gain_val)
            for var in TELEMETRY_VARS:
                gain_data[key][var].append(telemetry_row[var])

# --- Plotting with quadratic fits ---
for (gain_setting, freq), values in gain_data.items():
    y_vals = values["gain"]
    if not y_vals:
        continue

    plt.figure()
    for var, color in zip(TELEMETRY_VARS, COLORS):
        x_vals = values[var]
        if not x_vals or len(x_vals) < 3:
            continue

        # Scatter points
        plt.scatter(x_vals, y_vals, label=var, color=color, alpha=0.6)

        # Quadratic fit
        try:
            coeffs = np.polyfit(x_vals, y_vals, deg=2)  # a x² + b x + c
            poly = np.poly1d(coeffs)
            x_range = np.linspace(min(x_vals), max(x_vals), 300)
            plt.plot(x_range, poly(x_range), color=color, linestyle='--', alpha=0.9)

            # Store coefficients as c, b, a (for consistency with CSV naming)
            fit_coeffs[gain_setting].append({
                "Frequency_MHz": round(freq, 6),
                "Telemetry_Var": var,
                "Coeff_0": coeffs[2],  # constant
                "Coeff_1": coeffs[1],  # linear
                "Coeff_2": coeffs[0],  # quadratic
            })
        except Exception as e:
            print(f"Fit failed for {gain_setting}, {var} @ {freq:.4f} MHz: {e}")

    plt.xlabel("Telemetry Value (°C)")
    plt.ylabel("Gain Value")
    plt.title(f"{gain_setting} Gain vs Telemetry @ {freq:.4f} MHz")
    plt.legend()
    plt.tight_layout()

    filename = f"{gain_setting}_ALL_{freq:.4f}MHz.png".replace(" ", "_").replace("/", "_")
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()

    print(f"Saved plot: {filename}")

# --- Save coefficient CSVs ---
for gain in GAIN_SETTINGS:
    rows = fit_coeffs[gain]
    if not rows:
        continue
    df = pd.DataFrame(rows)
    csv_name = f"{gain}_fit_coeffs.csv"
    df.to_csv(os.path.join(OUTPUT_DIR, csv_name), index=False)
    print(f"Saved coefficients: {csv_name}")
