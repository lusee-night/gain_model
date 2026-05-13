#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the LuSEE-Night telemetry-regressed noise model at one frequency,
print the full predicted spectrum as CSV-style text, and optionally display
the predicted noise spectrum interactively.

No CSV or PNG files are exported.

For auto-correlations only, bins outside the PCA-supported frequency region
are excluded from the printed spectrum and plot:
    - keep 0.100 <= f <= 0.275 MHz
    - keep 0.500 <= f <= 50.0 MHz

Products 0-3:
    Uses SVD stage 3 products:
        gain_<GAIN>/svd_3/

Products 4-15:
    Uses SVD stage 1 products:
        gain_<GAINPAIR>/
        gain_<GAINPAIR>/svd_1/regression/

Reconstruction conventions:
    Autos (0-3):
        fractional residuals
        Xhat = mu * (1 + PC1*v1 + PC2*v2)

    Crosses (4-15):
        absolute residuals
        Xhat = mu + (PC1*v1 + PC2*v2)

Examples
--------
Auto-correlation example:
    python get_corrected_noise.py \
        --noise-root ~/gain_model/outputs/noise_svd_nmod4 \
        --product 0 \
        --gain H \
        --nmod4 1 \
        --freq-mhz 10.0 \
        --THERM_FPGA 33.58 \
        --SPE_ADC0_T 41.25 \
        --SPE_ADC1_T 39.12 \
        --SPE_1VAD8_V 1.906 \
        --VMON_1V2D 1.196 \
        --SPE_1VAD8_C 0.198

Cross-correlation example:
    python get_corrected_noise.py \
        --noise-root ~/gain_model/outputs/noise_svd_nmod4 \
        --product 10 \
        --gain HH \
        --nmod4 2 \
        --freq-mhz 15.0 \
        --THERM_FPGA 33.58 \
        --SPE_ADC0_T 41.25 \
        --SPE_ADC1_T 39.12 \
        --SPE_1VAD8_V 1.906 \
        --VMON_1V2D 1.196 \
        --SPE_1VAD8_C 0.198

Display plot example:
    python get_corrected_noise.py \
        --noise-root ~/gain_model/outputs/noise_svd_nmod4 \
        --product 0 \
        --gain H \
        --nmod4 1 \
        --freq-mhz 10.0 \
        --THERM_FPGA 33.58 \
        --SPE_ADC0_T 41.25 \
        --SPE_ADC1_T 39.12 \
        --SPE_1VAD8_V 1.906 \
        --VMON_1V2D 1.196 \
        --SPE_1VAD8_C 0.198 \
        --plot
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CHANNEL_BIN_MHZ = 0.025

PRODUCT_NAMES = {
    0: "Ch0 Auto",
    1: "Ch1 Auto",
    2: "Ch2 Auto",
    3: "Ch3 Auto",
    4: "Ch0×Ch1 Re",
    5: "Ch0×Ch1 Im",
    6: "Ch0×Ch2 Re",
    7: "Ch0×Ch2 Im",
    8: "Ch0×Ch3 Re",
    9: "Ch0×Ch3 Im",
    10: "Ch1×Ch2 Re",
    11: "Ch1×Ch2 Im",
    12: "Ch1×Ch3 Re",
    13: "Ch1×Ch3 Im",
    14: "Ch2×Ch3 Re",
    15: "Ch2×Ch3 Im",
}

CROSS_PRODUCT_CHANNELS = {
    4: (0, 1),
    5: (0, 1),
    6: (0, 2),
    7: (0, 2),
    8: (0, 3),
    9: (0, 3),
    10: (1, 2),
    11: (1, 2),
    12: (1, 3),
    13: (1, 3),
    14: (2, 3),
    15: (2, 3),
}


def product_dir_name(product: int) -> str:
    raw = PRODUCT_NAMES.get(product, f"prod{product}")
    safe = (
        raw.replace("×", "x")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )
    return f"prod_{product:03d}_{safe}"


def normalize_gain(product: int, gain: str) -> str:
    gain = str(gain).strip().upper()

    if product <= 3:
        if gain not in {"L", "M", "H"}:
            raise ValueError("For auto products 0-3, --gain must be one of L, M, H")
        return gain

    if len(gain) == 1:
        if gain not in {"L", "M", "H"}:
            raise ValueError("For cross products, single-letter --gain must be L, M, or H")
        return gain + gain

    if len(gain) == 2 and all(ch in {"L", "M", "H"} for ch in gain):
        return gain

    raise ValueError("For cross products 4-15, --gain must be a pair like LL, LM, MH, HH, etc.")


def find_model_dirs(noise_root: str, product: int, gain: str, nmod4: int) -> Tuple[str, str, str]:
    noise_root = os.path.expanduser(noise_root)
    group_out = os.path.join(noise_root, f"nmod4_{int(nmod4)}")
    prod_out = os.path.join(group_out, product_dir_name(product))
    gain_out = os.path.join(prod_out, f"gain_{gain}")

    if not os.path.isdir(gain_out):
        matches = glob.glob(os.path.join(group_out, f"prod_{product:03d}_*", f"gain_{gain}"))
        if len(matches) == 1:
            gain_out = matches[0]
        elif len(matches) > 1:
            raise FileNotFoundError("Multiple matching model directories found:\n" + "\n".join(matches))
        else:
            raise FileNotFoundError(f"Could not find model directory: {gain_out}")

    if product <= 3:
        stage_name = "svd_3"
        pca_dir = os.path.join(gain_out, "svd_3")
        regression_dir = os.path.join(pca_dir, "regression")
    else:
        stage_name = "svd_1"
        pca_dir = gain_out
        regression_dir = os.path.join(gain_out, "svd_1", "regression")

    return stage_name, pca_dir, regression_dir


def load_mean_and_eigvecs(pca_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    mean_path = os.path.join(pca_dir, "mean.npy")
    eig_path = os.path.join(pca_dir, "eigvecs.npy")

    missing = [p for p in (mean_path, eig_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing PCA/SVD export(s):\n" + "\n".join(missing))

    mean_vec = np.asarray(np.load(mean_path), dtype=float).reshape(-1)
    eigvecs = np.asarray(np.load(eig_path), dtype=float)

    if eigvecs.ndim != 2:
        raise ValueError(f"eigvecs.npy must be 2D, got shape {eigvecs.shape}")
    if eigvecs.shape[0] != mean_vec.size:
        raise ValueError(f"mean/eigvec length mismatch: mean={mean_vec.size}, eigvecs={eigvecs.shape}")
    if eigvecs.shape[1] < 2:
        raise ValueError(f"Need at least two eigenvectors, got {eigvecs.shape[1]}")

    return mean_vec, eigvecs


def load_alphas(regression_dir: str) -> pd.DataFrame:
    path = os.path.join(regression_dir, "pc_regression_alphas.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing regression coefficient file: {path}")

    df = pd.read_csv(path)
    required = {"pc", "model", "term", "alpha"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    return df


def telemetry_row_for_product(product: int, tele: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
    if product <= 3:
        tadc = tele["SPE_ADC0_T"] if product in (0, 1) else tele["SPE_ADC1_T"]
    else:
        ch_a, ch_b = CROSS_PRODUCT_CHANNELS[product]

        if ch_a in (0, 1) and ch_b in (0, 1):
            tadc = tele["SPE_ADC0_T"]
        elif ch_a in (2, 3) and ch_b in (2, 3):
            tadc = tele["SPE_ADC1_T"]
        else:
            tadc = 0.5 * (tele["SPE_ADC0_T"] + tele["SPE_ADC1_T"])

    labels = [
        "1",
        "THERM_FPGA",
        "TADC",
        "SPE_1VAD8_V",
        "VMON_1V2D",
        "SPE_1VAD8_C",
        "THERM_FPGA*THERM_FPGA",
        "TADC*TADC",
        "THERM_FPGA*TADC",
    ]

    th = tele["THERM_FPGA"]

    row = np.array(
        [
            1.0,
            th,
            tadc,
            tele["SPE_1VAD8_V"],
            tele["VMON_1V2D"],
            tele["SPE_1VAD8_C"],
            th * th,
            tadc * tadc,
            th * tadc,
        ],
        dtype=float,
    )

    return row, labels


def predict_one_pc(
    alphas: pd.DataFrame,
    pc: str,
    model: str,
    feature_row: np.ndarray,
    labels: List[str],
) -> float:
    sub = alphas[
        (alphas["pc"].astype(str) == pc)
        & (alphas["model"].astype(str) == model)
    ].copy()

    if sub.empty:
        return 0.0

    if "kept" in sub.columns:
        sub = sub[sub["kept"].fillna(1).astype(int) == 1]

    if sub.empty:
        return 0.0

    label_to_value = {lbl: float(feature_row[i]) for i, lbl in enumerate(labels)}
    yhat = 0.0
    ignored_terms = []

    for _, row in sub.iterrows():
        term = str(row["term"])

        if term in label_to_value:
            yhat += float(row["alpha"]) * label_to_value[term]
        else:
            ignored_terms.append(term)

    if ignored_terms:
        print(
            f"[WARN] Ignored terms not in feature row for {pc}/{model}: {sorted(set(ignored_terms))}",
            file=sys.stderr,
        )

    return float(yhat)


def reconstruct_spectrum(
    mean_vec: np.ndarray,
    eigvecs: np.ndarray,
    pc1: float,
    pc2: float,
    residual_mode: str,
) -> np.ndarray:
    resid_hat = pc1 * eigvecs[:, 0] + pc2 * eigvecs[:, 1]

    if residual_mode == "fractional":
        return mean_vec * (1.0 + resid_hat)

    if residual_mode == "absolute":
        return mean_vec + resid_hat

    raise ValueError(f"Unsupported residual_mode: {residual_mode}")


def filter_auto_bins(
    product: int,
    freq_grid_mhz: np.ndarray,
    spectrum: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    if product > 3:
        return freq_grid_mhz, spectrum

    keep = (
        ((freq_grid_mhz >= 0.100) & (freq_grid_mhz <= 0.275))
        |
        ((freq_grid_mhz >= 0.500) & (freq_grid_mhz <= 50.0))
    )

    return freq_grid_mhz[keep], spectrum[keep]

def interpolate_at_frequency(
    freq_grid_mhz: np.ndarray,
    spectrum: np.ndarray,
    freq_mhz: float,
) -> float:
    freq_mhz = float(freq_mhz)

    if freq_mhz < float(freq_grid_mhz[0]) or freq_mhz > float(freq_grid_mhz[-1]):
        raise ValueError(
            f"Requested frequency {freq_mhz} MHz is outside included model grid "
            f"[{freq_grid_mhz[0]}, {freq_grid_mhz[-1]}] MHz"
        )

    return float(np.interp(freq_mhz, freq_grid_mhz, spectrum))


def show_spectrum_plot(
    freq_grid_mhz: np.ndarray,
    spectrum: np.ndarray,
    title: str,
    query_freq_mhz: float,
    query_value: float,
) -> None:
    y = np.asarray(spectrum, dtype=float).copy()
    y[~np.isfinite(y)] = np.nan

    plt.figure(figsize=(9, 5.5))
    plt.plot(freq_grid_mhz, y, label="Predicted spectrum")

    plt.scatter(
        [query_freq_mhz],
        [query_value],
        marker="x",
        s=60,
        label=f"Query point: {query_freq_mhz:.3f} MHz",
        zorder=5,
        color='red',
    )

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Predicted noise (nV/√Hz)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if np.nanmin(y) >= 0:
        plt.ylim(bottom=0)

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate telemetry-regressed LuSEE-Night noise model."
    )

    ap.add_argument(
        "--noise-root",
        default=os.path.expanduser("~/gain_model/outputs/noise_svd_nmod4"),
    )
    ap.add_argument("--product", type=int, required=True, choices=range(16), metavar="{0..15}")
    ap.add_argument(
        "--gain",
        required=True,
        help="Autos: L/M/H. Crosses: LL/LM/.../HH, or a single letter to use LL/MM/HH.",
    )
    ap.add_argument("--nmod4", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--freq-mhz", type=float, required=True)
    ap.add_argument("--model", choices=("linear", "quadratic"), default="quadratic")

    ap.add_argument("--THERM_FPGA", type=float, required=True)
    ap.add_argument("--SPE_ADC0_T", type=float, required=True)
    ap.add_argument("--SPE_ADC1_T", type=float, required=True)
    ap.add_argument("--SPE_1VAD8_V", type=float, required=True)
    ap.add_argument("--VMON_1V2D", type=float, required=True)
    ap.add_argument("--SPE_1VAD8_C", type=float, required=True)

    ap.add_argument(
        "--plot",
        action="store_true",
        help="Display the predicted spectrum interactively. Does not save a PNG.",
    )

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    product = int(args.product)
    gain = normalize_gain(product, args.gain)

    stage_name, pca_dir, regression_dir = find_model_dirs(
        args.noise_root,
        product,
        gain,
        args.nmod4,
    )

    mean_vec, eigvecs = load_mean_and_eigvecs(pca_dir)
    alphas = load_alphas(regression_dir)

    if "used_jackknife" in alphas.columns and np.any(
        alphas["used_jackknife"].fillna(0).astype(int) == 1
    ):
        print(
            "[WARN] This coefficient file contains jackknife-filtered fits. "
            "noise_model.py exports the full-fit alpha values plus kept flags, not the post-selection refit coefficients. "
            "This script uses kept exported alpha values.",
            file=sys.stderr,
        )

    tele = {
        "THERM_FPGA": float(args.THERM_FPGA),
        "SPE_ADC0_T": float(args.SPE_ADC0_T),
        "SPE_ADC1_T": float(args.SPE_ADC1_T),
        "SPE_1VAD8_V": float(args.SPE_1VAD8_V),
        "VMON_1V2D": float(args.VMON_1V2D),
        "SPE_1VAD8_C": float(args.SPE_1VAD8_C),
    }

    feature_row, labels = telemetry_row_for_product(product, tele)

    pc1 = predict_one_pc(alphas, "PC1", args.model, feature_row, labels)
    pc2 = predict_one_pc(alphas, "PC2", args.model, feature_row, labels)

    residual_mode = "fractional" if product <= 3 else "absolute"

    spectrum = reconstruct_spectrum(
        mean_vec,
        eigvecs,
        pc1,
        pc2,
        residual_mode=residual_mode,
    )

    freq_grid_mhz = np.arange(mean_vec.size, dtype=float) * CHANNEL_BIN_MHZ

    freq_grid_mhz_filtered, spectrum_filtered = filter_auto_bins(
        product,
        freq_grid_mhz,
        spectrum,
    )

    value = interpolate_at_frequency(
        freq_grid_mhz_filtered,
        spectrum_filtered,
        args.freq_mhz,
    )

    print(f"product={product} ({PRODUCT_NAMES.get(product, 'unknown')})")
    print(f"gain={gain}")
    print(f"nmod4={args.nmod4}")
    print(f"stage={stage_name}")
    print(f"model={args.model}")
    print(f"residual_mode={residual_mode}")
    print(f"PC1_pred={pc1:.12g}")
    print(f"PC2_pred={pc2:.12g}")
    print(f"frequency_mhz={args.freq_mhz:.12g}")
    print(f"predicted_noise_nV_per_sqrtHz={value:.12g}")

    print("\n# ---------------- FULL PREDICTED SPECTRUM ----------------")
    print("frequency_mhz,predicted_noise_nV_per_sqrtHz")

    for f, s in zip(freq_grid_mhz_filtered, spectrum_filtered):
        print(f"{f:.6f},{s:.12g}")

    print("# --------------------------------------------------------")

    if args.plot:
        title = f"P{product} {gain} nmod4={args.nmod4} {stage_name} {args.model}"
        show_spectrum_plot(
            freq_grid_mhz_filtered,
            spectrum_filtered,
            title,
            args.freq_mhz,
            value,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())