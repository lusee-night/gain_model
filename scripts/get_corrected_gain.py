#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict corrected gain spectrum from exported PCA model products.

What this script does
---------------------
Given:
    - gain level: L, M, or H
    - channel: 0, 1, 2, or 3
    - telemetry values used by the regression model

it:
    1. builds the gain key (e.g. L0, M2, H3)
    2. loads the exported PCA model products for that gain:
           - alpha_refit.csv
           - {gain}_mean.npy
           - {gain}_eigvecs.npy
           - {gain}_freqs.npy
    3. predicts PC1 and PC2 using the saved refit coefficients
    4. reconstructs the corrected gain spectrum using the first two PCs
    5. prints the predicted gain values vs frequency
    6. optionally writes a CSV
    7. always displays a plot of predicted gain vs frequency
    8. optionally saves the plot if --output_png is provided

Important
---------
This predicts the "corrected" gain spectrum, i.e. the same quantity modeled in:
    ~/gain_model/outputs/gain_pca_model/corrected/...

So it matches the preamp-removed model output, not the raw gain.dat spectrum.

Usage examples
--------------
For channel 0 or 1:
python get_corrected_gain.py \
    --level L \
    --channel 0 \
    --THERM_FPGA 27.3 \
    --SPE_ADC0_T 28.1 \
    --SPE_1VAD8_V 1.803 \
    --VMON_1V2D 1.198 \
    --SPE_1VAD8_C 0.044

For channel 2 or 3:
python get_corrected_gain.py \
    --level H \
    --channel 3 \
    --THERM_FPGA 30.4 \
    --SPE_ADC1_T 29.8 \
    --SPE_1VAD8_V 1.799 \
    --VMON_1V2D 1.201 \
    --SPE_1VAD8_C 0.045

Optional saving:
python get_corrected_gain.py \
    --level M \
    --channel 2 \
    --THERM_FPGA 29.1 \
    --SPE_ADC1_T 28.4 \
    --SPE_1VAD8_V 1.801 \
    --VMON_1V2D 1.199 \
    --SPE_1VAD8_C 0.043 \
    --output_csv predicted_M2.csv \
    --output_png predicted_M2.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

ROOT = os.path.expanduser("~/gain_model")
OUT_ROOT = os.path.join(ROOT, "outputs", "gain_pca_model")

OUT_CORR = os.path.join(OUT_ROOT, "corrected")
OUT_CORR_PHASE1 = os.path.join(OUT_CORR, "phase1")
OUT_CORR_PHASE2 = os.path.join(OUT_CORR, "phase2")

PREAMP_DIR = os.path.join(OUT_CORR_PHASE1, "pca")
ALPHAS_DIR = os.path.join(OUT_CORR_PHASE2, "alphas")


# -------------------------------------------------------------------
# Model helpers
# -------------------------------------------------------------------

PC_COMPONENTS = ["PC1", "PC2"]


def make_gain_name(level: str, channel: int) -> str:
    """
    Build gain name like L0, M2, H3 from level and channel.
    """
    level = str(level).strip().upper()
    if level not in {"L", "M", "H"}:
        raise ValueError("level must be one of: L, M, H")

    if channel not in {0, 1, 2, 3}:
        raise ValueError("channel must be one of: 0, 1, 2, 3")

    return f"{level}{channel}"


def get_telemetry_cols_for_gain(gain: str):
    """
    Must exactly match the original modeling script.
    ch 0,1 -> SPE_ADC0_T
    ch 2,3 -> SPE_ADC1_T
    """
    ch = int(gain[-1])
    adc_col = "SPE_ADC0_T" if ch in (0, 1) else "SPE_ADC1_T"
    return ["THERM_FPGA", adc_col, "SPE_1VAD8_V", "VMON_1V2D", "SPE_1VAD8_C"]


def build_feature_matrix(X, tele_cols, order=2):
    """
    Exact same feature construction as the original training script.

    Features included:
      intercept
      linear telemetry terms
      quadratic terms only for:
          THERM_FPGA^2
          ADC^2
          THERM_FPGA * ADC
    """
    n_samples, n_features = X.shape
    Z = np.ones((n_samples, 1), dtype=float)
    feature_labels = ["1"]

    if order >= 1 and n_features > 0:
        Z = np.hstack([Z, X.astype(float)])
        feature_labels += list(tele_cols)

    if order == 2:
        quad_blocks = []
        quad_labels = []

        idx_th = tele_cols.index("THERM_FPGA") if "THERM_FPGA" in tele_cols else None
        adc_idx, adc_name = None, None
        for cand in ("SPE_ADC0_T", "SPE_ADC1_T"):
            if cand in tele_cols:
                adc_idx = tele_cols.index(cand)
                adc_name = cand
                break

        if idx_th is not None:
            th = X[:, idx_th]
            quad_blocks.append((th * th).reshape(-1, 1))
            quad_labels.append("THERM_FPGA*THERM_FPGA")

        if adc_idx is not None:
            adc = X[:, adc_idx]
            quad_blocks.append((adc * adc).reshape(-1, 1))
            quad_labels.append(f"{adc_name}*{adc_name}")

        if idx_th is not None and adc_idx is not None:
            th = X[:, idx_th]
            adc = X[:, adc_idx]
            quad_blocks.append((th * adc).reshape(-1, 1))
            quad_labels.append(f"THERM_FPGA*{adc_name}")

        if quad_blocks:
            Z = np.hstack([Z] + quad_blocks)
            feature_labels += quad_labels

    return Z, feature_labels


def load_model_products(gain: str):
    """
    Load exported model products for one gain.
    """
    alpha_path = os.path.join(ALPHAS_DIR, "alpha_refit.csv")
    mean_path = os.path.join(PREAMP_DIR, f"{gain}_mean.npy")
    eig_path = os.path.join(PREAMP_DIR, f"{gain}_eigvecs.npy")
    freq_path = os.path.join(PREAMP_DIR, f"{gain}_freqs.npy")

    missing = [p for p in [alpha_path, mean_path, eig_path, freq_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required model files:\n" + "\n".join(missing)
        )

    alphas_refit = pd.read_csv(alpha_path)
    mean_vec = np.load(mean_path)
    eigvecs = np.load(eig_path)
    freqs = np.load(freq_path)

    return alphas_refit, mean_vec, eigvecs, freqs


def build_single_feature_row(gain: str, telemetry_values: dict):
    """
    Create the 1-row design matrix for this gain, with labels.
    """
    tele_cols = get_telemetry_cols_for_gain(gain)

    missing = [c for c in tele_cols if c not in telemetry_values or telemetry_values[c] is None]
    if missing:
        raise ValueError(f"Missing telemetry inputs for {gain}: {missing}")

    X = np.array([[float(telemetry_values[c]) for c in tele_cols]], dtype=float)
    Z, labels = build_feature_matrix(X, tele_cols, order=2)

    return Z[0], labels, tele_cols


def predict_pc_from_refit(alphas_refit: pd.DataFrame, gain: str, pc: str, feature_row, labels):
    """
    Predict a single PC using the exported refit coefficients.
    """
    sub = alphas_refit[
        (alphas_refit["gain_setting"] == gain) &
        (alphas_refit["component"] == pc) &
        (alphas_refit["model"] == "quadratic")
    ]

    if sub.empty:
        return 0.0

    label_to_value = {lbl: feature_row[i] for i, lbl in enumerate(labels)}

    yhat = 0.0
    for _, row in sub.iterrows():
        term = str(row["term"])
        alpha = float(row["alpha_refit"])
        if term in label_to_value:
            yhat += alpha * label_to_value[term]

    return float(yhat)


def reconstruct_gain_from_pcs(mean_vec, eigvecs, pc1, pc2):
    """
    Reconstruct corrected gain spectrum from PC1 and PC2.
    """
    v1 = eigvecs[:, 0]
    v2 = eigvecs[:, 1]
    pred = mean_vec + pc1 * v1 + pc2 * v2
    return pred


def predict_gain_spectrum(level, channel, telemetry_values):
    """
    Main callable function.

    Returns
    -------
    gain : str
        Gain setting like L0, M2, H3
    freqs : ndarray
        Anchor frequencies in MHz
    pred_gain : ndarray
        Predicted corrected gain values at each anchor frequency
    info : dict
        Predicted PC values and metadata
    """
    gain = make_gain_name(level, channel)
    alphas_refit, mean_vec, eigvecs, freqs = load_model_products(gain)

    feature_row, labels, tele_cols = build_single_feature_row(gain, telemetry_values)

    pc1_hat = predict_pc_from_refit(alphas_refit, gain, "PC1", feature_row, labels)
    pc2_hat = predict_pc_from_refit(alphas_refit, gain, "PC2", feature_row, labels)

    pred_gain = reconstruct_gain_from_pcs(mean_vec, eigvecs, pc1_hat, pc2_hat)

    info = {
        "PC1": pc1_hat,
        "PC2": pc2_hat,
        "telemetry_cols": tele_cols,
    }

    return gain, freqs, pred_gain, info


def show_gain_plot(freqs, pred_gain, gain, info, output_png=None):
    """
    Always display predicted gain vs frequency.
    Optionally save to output_png if provided.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, pred_gain, marker='o', linewidth=1.8)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Predicted corrected gain")
    plt.title(
        f"Predicted corrected gain vs frequency for {gain}\n"
        f"PC1 = {info['PC1']:.4f}, PC2 = {info['PC2']:.4f}"
    )
    plt.grid(True)
    plt.tight_layout()

    if output_png is not None:
        plt.savefig(output_png, dpi=200)
        print(f"Wrote PNG: {output_png}")

    plt.show()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict corrected gain spectrum from exported PCA + telemetry regression model."
    )

    parser.add_argument(
        "--level",
        required=True,
        choices=["L", "M", "H", "l", "m", "h"],
        help="Gain level: L, M, or H"
    )
    parser.add_argument(
        "--channel",
        required=True,
        type=int,
        choices=[0, 1, 2, 3],
        help="Channel: 0, 1, 2, or 3"
    )

    parser.add_argument(
        "--THERM_FPGA",
        type=float,
        required=True,
        help="THERM_FPGA telemetry value"
    )

    parser.add_argument(
        "--SPE_ADC0_T",
        type=float,
        default=None,
        help="SPE_ADC0_T telemetry value (used for ch0/ch1)"
    )
    parser.add_argument(
        "--SPE_ADC1_T",
        type=float,
        default=None,
        help="SPE_ADC1_T telemetry value (used for ch2/ch3)"
    )

    parser.add_argument(
        "--SPE_1VAD8_V",
        type=float,
        required=True,
        help="SPE_1VAD8_V telemetry value"
    )
    parser.add_argument(
        "--VMON_1V2D",
        type=float,
        required=True,
        help="VMON_1V2D telemetry value"
    )
    parser.add_argument(
        "--SPE_1VAD8_C",
        type=float,
        required=True,
        help="SPE_1VAD8_C telemetry value"
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save predicted spectrum CSV"
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=None,
        help="Optional path to save predicted gain-vs-frequency PNG"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    telemetry_values = {
        "THERM_FPGA": args.THERM_FPGA,
        "SPE_ADC0_T": args.SPE_ADC0_T,
        "SPE_ADC1_T": args.SPE_ADC1_T,
        "SPE_1VAD8_V": args.SPE_1VAD8_V,
        "VMON_1V2D": args.VMON_1V2D,
        "SPE_1VAD8_C": args.SPE_1VAD8_C,
    }

    gain, freqs, pred_gain, info = predict_gain_spectrum(
        level=args.level,
        channel=args.channel,
        telemetry_values=telemetry_values
    )

    print(f"\nPredicted corrected gain spectrum for {gain}")
    print(f"Predicted PCs: PC1 = {info['PC1']:.6f}, PC2 = {info['PC2']:.6f}\n")
    print(f"{'freq_MHz':>12s}    {'predicted_gain':>18s}")
    print("-" * 34)
    for f, g in zip(freqs, pred_gain):
        print(f"{f:12.6f}    {g:18.10e}")

    if args.output_csv is not None:
        out_df = pd.DataFrame({
            "freq_MHz": freqs,
            "predicted_gain_corrected": pred_gain
        })
        out_df.to_csv(args.output_csv, index=False)
        print(f"\nWrote CSV: {args.output_csv}")

    show_gain_plot(freqs, pred_gain, gain, info, args.output_png)


if __name__ == "__main__":
    main()