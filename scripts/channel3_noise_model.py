#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Noise PCA + PC Regression (LuSEE / channel 3 noise model)
===============================================================================

What this script does (high-level)
----------------------------------
This script builds a *noise model* for LuSEE channel 3 (gains L3/M3/H3) in two stages:

(1) PCA stage (always useful, even without regression)
    - For each gain (L3, M3, H3) and each CPT directory:
        * Load the corresponding power_zero_<L/M/H>_3.dat file
        * Compute the column-wise mean spectrum (mean over rows)
        * Apply bin-masking / cleaning:
            - zero out bins in a precomputed "bad bins" list
            - zero out bins whose index % 8 == 0 (including bin 0)
            - zero out several additional hard-coded bin ranges
    - Stack all CPT mean spectra into a matrix X (n_CPT x n_bins).
    - Convert X into *fractional residuals*:
          mu[j] = mean over CPTs of X[:, j]
          R[:, j] = (X[:, j] - mu[j]) / mu[j]    if mu[j] != 0
                 = 0                             if mu[j] == 0  (masked bins)
    - Perform SVD on R to obtain:
          R = U S V^T
          PCs = U*S
          Eigenvectors = V
          Eigenvalues = S^2/(n-1)
    - Save standard PCA artifacts: PC tables, eigenvalues, eigenvectors/means, plots.

(2) Regression stage (predict PC1/PC2/PC3 from telemetry)
    - Load telemetry features for each gain from telemetry_per_gainsetting/<gain>.csv
    - Fit models for each PC (PC1..PC3):
        * "mean-only" baseline (constant predictor)
        * linear regression on telemetry features
        * quadratic regression using temperature-only quadratic terms
    - Use jackknife (leave-one-out) to estimate coefficient uncertainty:
          ratio = |alpha / stderr|
      and drop terms with ratio <= THRESHOLD_RATIO.
    - Refit the filtered model and save refit coefficients + diagnostic plots.

(3) Predicted spectra reconstruction + evaluation
    - Using predicted PC1..PC3, reconstruct predicted fractional residuals R_hat and then:
          X_hat = mu * (1 + R_hat)
      (masked bins forced to 0)
    - Compare predicted spectra vs:
        * measured cleaned spectra (X)
        * "Actual PC123" reconstruction using true PCs (best possible with 3 PCs)
    - Compute nRMS **only over valid bins** (mu != 0):
          nRMS = RMS(meas - pred) / mean(meas)   restricted to valid bins
    - Save per-CPT plots and a table of masked-bin nRMS values.

Directory organization
------------------------------------------------------
Root output:
  ~/gain_model/outputs/noise_pca_ch3_pc123/

1) PCA artifacts:
  pca/
    pcs/                 noise_pca_L3.csv, ...
    pcs_first5/          noise_pca_L3_first5.csv, ...
    eigenvalues/         noise_eigenvalues_L3.csv, ...
    eigenspectra/        noise_eigenspectrum_L3_log.png, ...
    bases/               L3_noise_mean.npy, L3_noise_eigvecs.npy, ...
    mean_spectra/        mean_power_L3.png, ...
    fractional_residuals/fractional_residual_spectra_L3.png, ...

2) PC regression artifacts:
  regression/
    jackknife/
      jackknife_alphas_with_errors.csv
      unreliable_alphas.txt
      alpha_refit.csv
      pc_rms_errors.csv
    pc_scatter/
    predicted_noise/
      spectra_csv/
        L3_linear_PC123_predicted.csv
        L3_actual_PC123_reconstruction.csv
        ...
      plots_per_cpt/
    nrms_errors_maskedbins.csv

Notes / assumptions
-------------------
- CPT names are assumed to match CPT_directories.txt ordering.
===============================================================================
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# CPT names aligned with CPT_directories.txt
CPT_NAMES = [
    "CPT2", "CPT3", "CPT4", "CPT5", "CPT6", "CPT7", "CPT8",
    "CPT9", "CPT10", "CPT11", "CPT12", "CPT13", "CPT15", "CPT16",
]

# Channel 3 only
GAINS_TO_RUN = ["L3", "M3", "H3"]

# PCs to model / reconstruct
PC_COMPONENTS = ["PC1", "PC2", "PC3"]
K_RECON = 3

# Input list of CPT directories
CPT_DIRS_TXT = os.path.expanduser("~/gain_model/scripts/CPT_directories.txt")

# Where per-gain bad-bin CSVs live (0-based file column numbers)
BAD_BINS_DIR_002 = os.path.expanduser(
    "~/gain_model/outputs/noise_ratio_cpt2_over_cpt1/bad_bins_lists_002"
)

# Telemetry per gain (files like L3.csv, M3.csv, H3.csv)
TELEM_PER_GAIN_DIR = os.path.expanduser(
    "~/gain_model/outputs/telemetry_per_gainsetting"
)

# Jackknife filter threshold (same spirit as gain model)
THRESHOLD_RATIO = 2.0

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

# Root output base
OUT_ROOT = os.path.expanduser("~/gain_model/outputs/noise_pca_ch3_pc123")
os.makedirs(OUT_ROOT, exist_ok=True)

# ---- 1) PCA artifacts ----
PCA_DIR = os.path.join(OUT_ROOT, "pca")
PCS_DIR = os.path.join(PCA_DIR, "pcs")
PCS5_DIR = os.path.join(PCA_DIR, "pcs_first5")
EIGVAL_DIR = os.path.join(PCA_DIR, "eigenvalues")
EIGPLOT_DIR = os.path.join(PCA_DIR, "eigenspectra")
BASES_DIR = os.path.join(PCA_DIR, "bases")
MEAN_SPEC_DIR = os.path.join(PCA_DIR, "mean_spectra")
FRAC_RES_DIR = os.path.join(PCA_DIR, "fractional_residuals")

for d in [PCA_DIR, PCS_DIR, PCS5_DIR, EIGVAL_DIR, EIGPLOT_DIR, BASES_DIR, MEAN_SPEC_DIR, FRAC_RES_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- 2) Regression artifacts ----
REG_DIR = os.path.join(OUT_ROOT, "regression")
REG_JACK_DIR = os.path.join(REG_DIR, "jackknife")
REG_PC_SCATTER_DIR = os.path.join(REG_DIR, "pc_scatter")

REG_PRED_DIR = os.path.join(REG_DIR, "predicted_noise")
REG_SPECTRA_CSV_DIR = os.path.join(REG_PRED_DIR, "spectra_csv")
REG_PLOTS_PER_CPT_DIR = os.path.join(REG_PRED_DIR, "plots_per_cpt")

for d in [REG_DIR, REG_JACK_DIR, REG_PC_SCATTER_DIR, REG_PRED_DIR, REG_SPECTRA_CSV_DIR, REG_PLOTS_PER_CPT_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# HELPERS: DATA PREP + PCA
# =============================================================================

def fractional_residual_matrix(matrix: np.ndarray, eps: float = 0.0):
    """
    Compute fractional residuals per bin across CPTs.

      mu = mean(matrix, axis=0)
      R[:,j] = (matrix[:,j] - mu[j]) / mu[j]   if mu[j] != 0
             = 0                              if mu[j] == 0  (masked/constant bins)

    Returns:
      R (n_cpt, n_bins), mu (n_bins,)
    """
    mu = np.mean(matrix, axis=0).astype(float)
    R = matrix.astype(float) - mu

    denom = mu.copy()
    if eps > 0:
        zero_like = np.abs(denom) < eps
    else:
        zero_like = (denom == 0.0)

    denom_safe = denom.copy()
    denom_safe[zero_like] = 1.0

    R = R / denom_safe
    R[:, zero_like] = 0.0
    return R, mu


def compute_pca_svd_from_matrix(X: np.ndarray):
    """
    PCA via SVD on matrix X.

      X = U S V^T
      pcs = U*S
      eigvecs = V = Vt.T
      eigenvals = S^2/(n_samples-1)
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = U * S
    eigvecs = Vt.T
    n = X.shape[0]
    eigenvals = (S ** 2) / (n - 1) if n > 1 else (S ** 2)
    return pcs, eigvecs, eigenvals


def gain_to_power_filename(gain: str) -> str:
    """
    'L3' -> power_zero_L_3.dat
    """
    if len(gain) != 2:
        raise ValueError(f"Unexpected gain label '{gain}'")
    letter = gain[0]
    ch = gain[1]
    return f"power_zero_{letter}_{ch}.dat"


def read_bad_bins_for_gain_002(gain: str) -> np.ndarray:
    """
    Reads:
      bad_bins_filecol0based_<gain>_002.csv
    and returns sorted unique array of ints (0-based file columns).
    """
    path = os.path.join(BAD_BINS_DIR_002, f"bad_bins_filecol0based_{gain}_002.csv")
    if not os.path.exists(path):
        print(f"[WARN] No bad-bin CSV found for {gain}: {path} (treating as none)")
        return np.array([], dtype=int)

    vals = []
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)  # header
        for row in r:
            if not row:
                continue
            vals.append(int(row[0]))

    if not vals:
        return np.array([], dtype=int)

    return np.array(sorted(set(vals)), dtype=int)


def mask_manual_bin_ranges(vec: np.ndarray) -> None:
    """
    Zero out additional hard-coded bin ranges IN PLACE (inclusive, 0-based).
    """
    ranges = [
        (375, 690),
        (765, 950),
        (1050, 1515),
    ]
    n = vec.size
    for lo, hi in ranges:
        lo_eff = max(lo, 0)
        hi_eff = min(hi, n - 1)
        if lo_eff <= hi_eff:
            vec[lo_eff:hi_eff + 1] = 0.0


def load_mean_vector_and_zero_bins(cpt_dir: str, gain: str, bins_to_zero: np.ndarray) -> np.ndarray:
    """
    Load power_zero file (2D), take column-wise mean over rows -> col_means,
    then set specified bins (0-based file column indices) to 0.
    """
    path = os.path.join(os.path.expanduser(cpt_dir), gain_to_power_filename(gain))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file for {gain}: {path}")

    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 1:
        raise ValueError(f"Expected at least 1 column in {path}, got {data.shape[1]}")

    col_means = data.mean(axis=0).astype(float)

    if bins_to_zero.size > 0:
        valid = (bins_to_zero >= 0) & (bins_to_zero < col_means.size)
        col_means[bins_to_zero[valid]] = 0.0

    return col_means


def write_pc_csv(path: str, cpt_names, pcs: np.ndarray):
    """
    Save PC scores (all components) to CSV.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["CPT"] + [f"PC{i+1}" for i in range(pcs.shape[1])])
        for name, row in zip(cpt_names, pcs):
            w.writerow([name] + row.tolist())


def write_first_k_pcs(path: str, cpt_names, pcs: np.ndarray, k: int = 5):
    """
    Save first k PC scores to CSV.
    """
    k_use = min(k, pcs.shape[1])
    pcs_k = pcs[:, :k_use]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["CPT"] + [f"PC{i+1}" for i in range(k_use)])
        for name, row in zip(cpt_names, pcs_k):
            w.writerow([name] + row.tolist())


def write_eigenvalues_csv(path: str, eigenvals: np.ndarray):
    """
    Save eigenvalues + explained variance ratio to CSV.
    """
    total = float(eigenvals.sum()) if eigenvals.size else 0.0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "eigenvalue", "explained_variance_ratio"])
        for i, lam in enumerate(eigenvals, start=1):
            ratio = float(lam / total) if total > 0 else 0.0
            w.writerow([i, float(lam), ratio])


def plot_eigen_spectrum_png(path: str, eigenvals: np.ndarray, gain: str):
    """
    Log-scale scree plot including all eigenvalues.
    """
    if eigenvals.size == 0:
        return
    idx = np.arange(1, len(eigenvals) + 1)
    eps = 1e-16
    eigenvals_safe = np.where(eigenvals > 0, eigenvals, eps)

    plt.figure(figsize=(6, 4))
    plt.plot(idx, eigenvals_safe, marker="o")
    plt.yscale("log")
    plt.xlabel("Principal component index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title(f"Eigenvalue spectrum for {gain} (log scale)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_eigen_spectrum_exclude_last_png(path: str, eigenvals: np.ndarray, gain: str):
    """
    Log-scale scree plot excluding the smallest (last) eigenvalue.
    """
    if eigenvals.size <= 1:
        return
    eigenvals_trim = eigenvals[:-1]
    idx = np.arange(1, len(eigenvals_trim) + 1)

    eps = 1e-16
    eigenvals_safe = np.where(eigenvals_trim > 0, eigenvals_trim, eps)

    plt.figure(figsize=(6, 4))
    plt.plot(idx, eigenvals_safe, marker="o")
    plt.yscale("log")
    plt.xlabel("Principal component index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title(f"Eigenvalue spectrum for {gain} (log scale, smallest removed)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _get_cpt_colors():
    """
    Fixed color palette for consistent CPT lines across plots.
    """
    return [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
        "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
    ]


def zero_to_nan(y: np.ndarray) -> np.ndarray:
    """
    Convert zeros to NaN for plotting gaps (masked bins disappear).
    """
    y_plot = y.astype(float).copy()
    y_plot[y_plot == 0.0] = np.nan
    return y_plot


def plot_mean_spectra_all_bins(matrix: np.ndarray, gain: str):
    """
    Plot mean spectra per CPT for a gain (masked bins hidden by NaN).
    x-axis: 0..N-1 (file column indices)
    """
    n_cpt, n_bins = matrix.shape
    x = np.arange(0, n_bins)
    colors = _get_cpt_colors()

    plt.figure(figsize=(9, 6))
    for row, label, color in zip(matrix, CPT_NAMES, colors):
        y_plot = zero_to_nan(row)
        plt.plot(x, y_plot, label=label, color=color)

    plt.xlabel("Bin index (file column, 0-based)")
    plt.ylabel("Mean power")
    plt.title(f"Mean power_zero spectrum per CPT for {gain} (masked bins hidden)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out_path = os.path.join(MEAN_SPEC_DIR, f"mean_power_{gain}.png")
    plt.savefig(out_path)
    plt.close()


def load_noise_matrix_for_gain_cleaned(cpt_dirs, gain: str) -> np.ndarray:
    """
    Build cleaned noise matrix for a gain across CPTs:
      - mean over rows (per CPT)
      - zero out bins listed in bad-bin CSV
      - zero out bins with idx % 8 == 0 (includes bin 0)
      - zero manual bin ranges
    Returns X: shape (n_CPT, n_bins)
    """
    bad_bins = read_bad_bins_for_gain_002(gain)
    rows = []

    for d, _cpt_name in zip(cpt_dirs, CPT_NAMES):
        vec = load_mean_vector_and_zero_bins(d, gain, bins_to_zero=bad_bins)

        idx = np.arange(vec.size)
        vec[(idx % 8) == 0] = 0.0

        mask_manual_bin_ranges(vec)
        rows.append(vec)

    return np.vstack(rows)


def plot_fractional_residual_spectra(frac_mat: np.ndarray, gain: str, out_dir: str, mhz_per_bin: float = 0.025):
    """
    Plot fractional residual spectra per CPT: y = (x - mu)/mu.
    Masked bins are 0 in frac_mat; convert zeros to NaN so gaps show.
    """
    n_cpt, n_bins = frac_mat.shape
    x = np.arange(n_bins)
    colors = _get_cpt_colors()

    plt.figure(figsize=(9, 6))
    for row, label, color in zip(frac_mat, CPT_NAMES, colors):
        y = row.astype(float).copy()
        y[y == 0.0] = np.nan
        plt.plot(x, y, label=label, color=color)

    max_mhz = (n_bins - 1) * mhz_per_bin
    plt.xlabel(f"Bin index (0-based); max freq ≈ {max_mhz:.3f} MHz")
    plt.ylabel("Fractional residual (x - mean)/mean")
    plt.title(f"Fractional residual spectra per CPT for {gain} (masked bins hidden)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"fractional_residual_spectra_{gain}.png")
    plt.savefig(out_path)
    plt.close()

# =============================================================================
# HELPERS: REGRESSION + METRICS + RECONSTRUCTION
# =============================================================================

def get_telemetry_cols_for_noise_gain(gain: str):
    """
    Telemetry features used to regress PCs (channel 3 gains).
    """
    return ["THERM_FPGA", "SPE_ADC1_T", "SPE_1VAD8_V", "VMON_1V2D", "SPE_1VAD8_C", "PFPS_PA3_T"]


def build_feature_matrix(X, tele_cols, order=2):
    """
    Feature builder:
      - intercept
      - linear terms for ALL tele_cols
      - quadratic block (order==2): temperature-only quadratic terms (expanded):
            THERM_FPGA^2
            (SPE_ADCx_T)^2
            (PFPS_PA3_T)^2
            THERM_FPGA*SPE_ADCx_T
            THERM_FPGA*PFPS_PA3_T
            SPE_ADCx_T*PFPS_PA3_T

    NOTE: As currently implemented, we *only* add:
          - squares of THERM_FPGA, SPE_ADC(0/1)_T, PFPS_PA3_T when present
          - cross-term THERM_FPGA * SPE_ADCx_T (when both exist)
    """
    n_samples, n_features = X.shape
    Z = np.ones((n_samples, 1), dtype=float)
    feature_labels = ["1"]

    # Linear terms
    if order >= 1 and n_features > 0:
        Z = np.hstack([Z, X.astype(float)])
        feature_labels += list(tele_cols)

    # Quadratic temperature terms
    if order == 2:
        quad_blocks = []
        quad_labels = []

        idx_th = tele_cols.index("THERM_FPGA") if "THERM_FPGA" in tele_cols else None

        adc_idx = None
        adc_name = None
        for cand in ("SPE_ADC0_T", "SPE_ADC1_T"):
            if cand in tele_cols:
                adc_idx = tele_cols.index(cand)
                adc_name = cand
                break

        idx_pfps = tele_cols.index("PFPS_PA3_T") if "PFPS_PA3_T" in tele_cols else None

        th = X[:, idx_th] if idx_th is not None else None
        adc = X[:, adc_idx] if adc_idx is not None else None
        pfps = X[:, idx_pfps] if idx_pfps is not None else None

        # Squares
        if th is not None:
            quad_blocks.append((th * th).reshape(-1, 1))
            quad_labels.append("THERM_FPGA*THERM_FPGA")

        if adc is not None:
            quad_blocks.append((adc * adc).reshape(-1, 1))
            quad_labels.append(f"{adc_name}*{adc_name}")

        if pfps is not None:
            quad_blocks.append((pfps * pfps).reshape(-1, 1))
            quad_labels.append("PFPS_PA3_T*PFPS_PA3_T")

        # Cross-terms (currently only THERM_FPGA*ADC)
        if th is not None and adc is not None:
            quad_blocks.append((th * adc).reshape(-1, 1))
            quad_labels.append(f"THERM_FPGA*{adc_name}")

        if quad_blocks:
            Z = np.hstack([Z] + quad_blocks)
            feature_labels += quad_labels

    return Z, feature_labels


def nrms_masked(meas: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> float:
    """
    nRMS over valid bins only:
      nRMS = RMS(meas-pred) / mean(meas), restricted to valid==True bins.
    """
    m = meas[valid].astype(float)
    p = pred[valid].astype(float)
    if m.size == 0:
        return float("nan")
    rms = float(np.sqrt(np.mean((m - p) ** 2)))
    denom = float(np.mean(m))
    return float(rms / denom) if denom != 0.0 else float("nan")


def global_nrms_masked(measured_mat: np.ndarray, pred_mat: np.ndarray, valid: np.ndarray) -> float:
    """
    Global nRMS over all CPT×valid_bins:
      RMS(errors)/mean(measured), restricted to valid bins.
    """
    M = measured_mat[:, valid].astype(float)
    P = pred_mat[:, valid].astype(float)
    if M.size == 0:
        return float("nan")
    rms = float(np.sqrt(np.mean((M - P) ** 2)))
    denom = float(np.mean(M))
    return float(rms / denom) if denom != 0.0 else float("nan")


def reconstruct_noise_from_pcs(pcs_k: np.ndarray, mean_vec: np.ndarray, eigvecs: np.ndarray, k: int):
    """
    Noise PCA is performed on fractional residuals R.
    Reconstruction:
      R_hat = pcs_k @ V_k^T
      X_hat = mean_vec * (1 + R_hat)
    Masked bins: mean_vec == 0 -> force X_hat = 0 there.
    """
    V_k = eigvecs[:, :k]      # (n_bins, k)
    R_hat = pcs_k @ V_k.T     # (n_cpt, n_bins)
    X_hat = mean_vec.reshape(1, -1) * (1.0 + R_hat)

    masked = (mean_vec == 0.0)
    if np.any(masked):
        X_hat[:, masked] = 0.0
    return X_hat


def save_spectrum_csv(path: str, cpt_names, mat: np.ndarray):
    """
    Save spectra matrix with CPT labels. Columns labeled by bin index.
    """
    n_bins = mat.shape[1]
    cols = [str(i) for i in range(n_bins)]
    df = pd.DataFrame(mat, columns=cols)
    df.insert(0, "CPT", list(cpt_names))
    df.to_csv(path, index=False)


def plot_cpt_noise_comparison(
    out_png: str,
    gain: str,
    cpt: str,
    measured: np.ndarray,
    predicted: np.ndarray,
    actual_pc123: np.ndarray,
    mean_vec: np.ndarray,
    model_label: str,
    nrms_pred: float,
    nrms_actual: float,
):
    """
    Plot measured vs predicted vs actual PC123 for one CPT and gain.
    Masked bins (mean_vec==0) hidden by NaN in plotting.
    """
    valid = (mean_vec != 0.0)
    x = np.arange(measured.size)

    y_meas = measured.astype(float).copy()
    y_pred = predicted.astype(float).copy()
    y_act = actual_pc123.astype(float).copy()

    y_meas[~valid] = np.nan
    y_pred[~valid] = np.nan
    y_act[~valid] = np.nan

    plt.figure(figsize=(9, 5.5))
    plt.plot(x, y_meas, label="Measured (cleaned)")
    plt.plot(x, y_pred, linestyle="--", label=f"Predicted ({model_label}, PC1-3)")
    plt.plot(x, y_act, linestyle=":", label="Actual PC1-3")

    plt.xlabel("Bin index (0-based)")
    plt.ylabel("Mean power")
    plt.title(
        f"{gain} {model_label} PC1-3 — {cpt}\n"
        f"nRMS_pred(masked)={nrms_pred:.4f}   "
        f"nRMS_actualPC123(masked)={nrms_actual:.4f}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_pc_scatter(out_png: str, gain: str, pc: str, model_label: str, y_true: np.ndarray, y_pred: np.ndarray, rms_val: float):
    """
    Scatter of predicted vs actual PC values for one PC.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.85)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel(f"Actual {pc}")
    plt.ylabel(f"Predicted {pc}")
    plt.title(f"{gain}: {model_label} ({pc})\nPC-RMS={rms_val:.4f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Orchestrates:
      - loading CPT directories
      - PCA per gain (L3/M3/H3)
      - regression per gain and per PC (PC1..PC3)
      - predicted spectra reconstruction + evaluation
      - writing summary CSV outputs
    """
    # ---- Load CPT directory list ----
    with open(CPT_DIRS_TXT, "r") as f:
        cpt_dirs = [line.strip() for line in f if line.strip()]

    if len(cpt_dirs) != len(CPT_NAMES):
        raise RuntimeError(
            f"Mismatch: CPT_directories.txt has {len(cpt_dirs)} dirs, "
            f"but CPT_NAMES has {len(CPT_NAMES)} entries."
        )

    # ---- Accumulators for regression outputs across gains ----
    all_alpha_records = []
    unreliable_alpha_records = []
    final_alpha_records = []
    pc_rms_records = []
    nrms_records = []  # masked-bin nRMS for spectra comparisons

    # ---- Loop over channel-3 gains ----
    for gain in GAINS_TO_RUN:
        print(f"[Noise SVD-PCA] Processing {gain} (channel 3 cleaned)...")

        # ---------------------------------------------------------------------
        # (A) Build cleaned noise matrix X (n_CPT x n_bins)
        # ---------------------------------------------------------------------
        matrix = load_noise_matrix_for_gain_cleaned(cpt_dirs, gain)

        # ---------------------------------------------------------------------
        # (B) Fractional residuals R and mean spectrum mu (mean_vec)
        # ---------------------------------------------------------------------
        frac_mat, mean_vec = fractional_residual_matrix(matrix)

        # Diagnostic plot of fractional residual spectra
        plot_fractional_residual_spectra(frac_mat, gain, FRAC_RES_DIR, mhz_per_bin=0.025)

        # ---------------------------------------------------------------------
        # (C) PCA via SVD on fractional residual matrix
        # ---------------------------------------------------------------------
        pcs, eigvecs, eigenvals = compute_pca_svd_from_matrix(frac_mat)

        # ---------------------------------------------------------------------
        # (D) Save PCA artifacts (CSV/NPY/plots)
        # ---------------------------------------------------------------------
        pcs_csv = os.path.join(PCS_DIR, f"noise_pca_{gain}.csv")
        write_pc_csv(pcs_csv, CPT_NAMES, pcs)

        pcs5_csv = os.path.join(PCS5_DIR, f"noise_pca_{gain}_first5.csv")
        write_first_k_pcs(pcs5_csv, CPT_NAMES, pcs, k=5)

        eig_csv = os.path.join(EIGVAL_DIR, f"noise_eigenvalues_{gain}.csv")
        write_eigenvalues_csv(eig_csv, eigenvals)

        eig_png = os.path.join(EIGPLOT_DIR, f"noise_eigenspectrum_{gain}_log.png")
        plot_eigen_spectrum_png(eig_png, eigenvals, gain)

        eig_trim_png = os.path.join(EIGPLOT_DIR, f"noise_eigenspectrum_{gain}_log_exclude_last.png")
        plot_eigen_spectrum_exclude_last_png(eig_trim_png, eigenvals, gain)

        np.save(os.path.join(BASES_DIR, f"{gain}_noise_mean.npy"), mean_vec)
        np.save(os.path.join(BASES_DIR, f"{gain}_noise_eigvecs.npy"), eigvecs)

        plot_mean_spectra_all_bins(matrix, gain)

        # ---------------------------------------------------------------------
        # (E) Regression: predict PC1/PC2/PC3 from telemetry (linear + quadratic)
        # ---------------------------------------------------------------------
        tele_cols = get_telemetry_cols_for_noise_gain(gain)
        tele_path = os.path.join(TELEM_PER_GAIN_DIR, f"{gain}.csv")
        if not os.path.exists(tele_path):
            print(f"[WARN] Missing telemetry for {gain}: {tele_path}. Skipping regression for this gain.")
            continue

        tele_df = pd.read_csv(tele_path)
        missing_cols = ["CPT"] + [c for c in tele_cols if c not in tele_df.columns]
        missing_cols = [c for c in missing_cols if c not in tele_df.columns]
        if missing_cols:
            print(f"[WARN] {gain}: telemetry file missing columns {missing_cols}. Skipping regression for this gain.")
            continue

        # Align telemetry rows to CPT_NAMES
        tele_df = tele_df.set_index("CPT").reindex(CPT_NAMES).reset_index()

        # Drop CPTs with missing telemetry values
        if tele_df[tele_cols].isna().any().any():
            bad = tele_df[tele_cols].isna().any(axis=1)
            bad_cpts = tele_df.loc[bad, "CPT"].tolist()
            print(f"[WARN] {gain}: missing telemetry values for CPTs {bad_cpts}. Dropping those CPTs.")
            tele_df = tele_df.dropna(subset=tele_cols).reset_index(drop=True)

        keep_cpts = tele_df["CPT"].tolist()
        idx_map = {c: i for i, c in enumerate(CPT_NAMES)}
        keep_idx = [idx_map[c] for c in keep_cpts]

        # Keep only CPTs used in regression
        matrix_keep = matrix[keep_idx, :]
        pcs_keep = pcs[keep_idx, :]

        mean_vec_keep = mean_vec

        Xtele = tele_df[tele_cols].astype(float).to_numpy()
        n = Xtele.shape[0]

        reliable_terms = {}  # (pc_name, model_name) -> set(term labels)

        # ---------------------------------------------------------------------
        # (E1) Jackknife: estimate stderr for each coefficient, mark unreliable
        # ---------------------------------------------------------------------
        for pc_name in PC_COMPONENTS:
            pc_idx = int(pc_name.replace("PC", "")) - 1
            if pc_idx >= pcs_keep.shape[1]:
                continue
            y = pcs_keep[:, pc_idx].astype(float)

            for order in [1, 2]:
                model_name = "linear" if order == 1 else "quadratic"
                Z, labels = build_feature_matrix(Xtele, tele_cols, order=order)
                alpha_full = np.linalg.lstsq(Z, y, rcond=None)[0]

                # Jackknife leave-one-out coefficients
                jack = []
                for i in range(n):
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    Zi = Z[mask, :]
                    yi = y[mask]
                    ai = np.linalg.lstsq(Zi, yi, rcond=None)[0]
                    jack.append(ai)
                jack = np.vstack(jack)

                var_jack = np.var(jack, axis=0, ddof=1) if n > 1 else np.zeros_like(alpha_full)
                stderr = np.sqrt(var_jack / n) if n > 0 else np.zeros_like(alpha_full)

                for lbl, a, se in zip(labels, alpha_full, stderr):
                    ratio = (abs(a / se) if se != 0 else np.inf)
                    rec = {
                        "gain_setting": gain,
                        "component": pc_name,
                        "model": model_name,
                        "term": lbl,
                        "alpha": float(a),
                        "stderr": float(se),
                        "ratio": float(ratio),
                    }
                    all_alpha_records.append(rec)
                    if ratio <= THRESHOLD_RATIO:
                        unreliable_alpha_records.append(rec)
                    else:
                        reliable_terms.setdefault((pc_name, model_name), set()).add(lbl)

        # ---------------------------------------------------------------------
        # (E2) Refit filtered models and generate predicted PCs
        # ---------------------------------------------------------------------
        preds_pc = {pc: {"mean": None, "linear": None, "quadratic": None} for pc in PC_COMPONENTS}

        for pc_name in PC_COMPONENTS:
            pc_idx = int(pc_name.replace("PC", "")) - 1
            if pc_idx >= pcs_keep.shape[1]:
                continue
            y = pcs_keep[:, pc_idx].astype(float)

            # Mean-only predictor
            preds_pc[pc_name]["mean"] = np.full_like(y, float(np.mean(y)))
            pc_rms_mean = float(np.sqrt(np.mean((y - preds_pc[pc_name]["mean"]) ** 2)))
            pc_rms_records.append({
                "gain_setting": gain,
                "component": pc_name,
                "model": "mean",
                "pc_rms": pc_rms_mean,
                "n_samples": int(n),
            })

            # Linear + Quadratic filtered fits
            for order in [1, 2]:
                model_name = "linear" if order == 1 else "quadratic"
                Z, labels = build_feature_matrix(Xtele, tele_cols, order=order)

                keep = reliable_terms.get((pc_name, model_name), set())
                keep_idx_terms = [i for i, lbl in enumerate(labels) if lbl in keep]
                if not keep_idx_terms:
                    continue

                Zf = Z[:, keep_idx_terms]
                af = np.linalg.lstsq(Zf, y, rcond=None)[0]
                y_pred = Zf @ af
                preds_pc[pc_name][model_name] = y_pred

                # Save refit coefficients
                kept_labels = [labels[i] for i in keep_idx_terms]
                for term, aval in zip(kept_labels, af):
                    final_alpha_records.append({
                        "gain_setting": gain,
                        "component": pc_name,
                        "model": model_name,
                        "term": term,
                        "alpha_refit": float(aval),
                    })

                pc_rms = float(np.sqrt(np.mean((y - y_pred) ** 2)))
                pc_rms_records.append({
                    "gain_setting": gain,
                    "component": pc_name,
                    "model": model_name,
                    "pc_rms": pc_rms,
                    "n_samples": int(n),
                })

                out_scatter = os.path.join(REG_PC_SCATTER_DIR, f"{gain}_{model_name}_{pc_name.lower()}_scatter.png")
                plot_pc_scatter(out_scatter, gain, pc_name, model_name.capitalize(), y, y_pred, pc_rms)

        # ---------------------------------------------------------------------
        # (F) Reconstruct predicted spectra and evaluate (masked-bin nRMS)
        # ---------------------------------------------------------------------
        valid_bins = (mean_vec_keep != 0.0)

        # Best-possible reconstruction using true PC1-3 for kept CPTs
        pcs_true_k = pcs_keep[:, :K_RECON]
        actual_pc123_mat = reconstruct_noise_from_pcs(pcs_true_k, mean_vec_keep, eigvecs, k=K_RECON)

        save_spectrum_csv(
            os.path.join(REG_SPECTRA_CSV_DIR, f"{gain}_actual_PC123_reconstruction.csv"),
            keep_cpts,
            actual_pc123_mat
        )

        # Record truncation-only nRMS (Actual PC123 vs measured)
        for i, cpt in enumerate(keep_cpts):
            nr = nrms_masked(matrix_keep[i, :], actual_pc123_mat[i, :], valid_bins)
            nrms_records.append({"gain": gain, "model": "Actual", "case": "PC123", "CPT": cpt, "nRMS_masked": nr})
        nr_global = global_nrms_masked(matrix_keep, actual_pc123_mat, valid_bins)
        nrms_records.append({"gain": gain, "model": "Actual", "case": "PC123", "CPT": "GLOBAL", "nRMS_masked": nr_global})

        # Predicted spectra from mean/linear/quadratic PC predictions
        for model_name in ["mean", "linear", "quadratic"]:
            pc_preds = []
            for pc_name in PC_COMPONENTS:
                v = preds_pc.get(pc_name, {}).get(model_name, None)
                if v is None:
                    v = np.zeros(len(keep_cpts), dtype=float)
                pc_preds.append(v.reshape(-1, 1))
            pcs_hat_k = np.hstack(pc_preds)  # (n_keep, 3)

            pred_mat = reconstruct_noise_from_pcs(pcs_hat_k, mean_vec_keep, eigvecs, k=K_RECON)

            save_spectrum_csv(
                os.path.join(REG_SPECTRA_CSV_DIR, f"{gain}_{model_name}_PC123_predicted.csv"),
                keep_cpts,
                pred_mat
            )

            model_label = "Mean only" if model_name == "mean" else model_name.capitalize()

            for i, cpt in enumerate(keep_cpts):
                nr_pred = nrms_masked(matrix_keep[i, :], pred_mat[i, :], valid_bins)
                nr_act = nrms_masked(matrix_keep[i, :], actual_pc123_mat[i, :], valid_bins)
                nrms_records.append({"gain": gain, "model": model_label, "case": "PC123", "CPT": cpt, "nRMS_masked": nr_pred})

                out_png = os.path.join(REG_PLOTS_PER_CPT_DIR, f"{gain}_{model_name}_PC123_{cpt}.png")
                plot_cpt_noise_comparison(
                    out_png=out_png,
                    gain=gain,
                    cpt=cpt,
                    measured=matrix_keep[i, :],
                    predicted=pred_mat[i, :],
                    actual_pc123=actual_pc123_mat[i, :],
                    mean_vec=mean_vec_keep,
                    model_label=model_label,
                    nrms_pred=nr_pred,
                    nrms_actual=nr_act,
                )

            nr_global_pred = global_nrms_masked(matrix_keep, pred_mat, valid_bins)
            nrms_records.append({"gain": gain, "model": model_label, "case": "PC123", "CPT": "GLOBAL", "nRMS_masked": nr_global_pred})

        print(f"[Noise SVD-PCA] Finished {gain}.")

    # =============================================================================
    # WRITE REGRESSION OUTPUT TABLES
    # =============================================================================

    if unreliable_alpha_records:
        txt_path = os.path.join(REG_JACK_DIR, "unreliable_alphas.txt")
        with open(txt_path, "w") as f:
            for r in unreliable_alpha_records:
                f.write(
                    f"{r['component']}, {r['gain_setting']}, {r['model']}, {r['term']}, "
                    f"alpha={r['alpha']:.6g}, stderr={r['stderr']:.6g}, alpha/stderr={r['ratio']:.3g}\n"
                )

    if all_alpha_records:
        pd.DataFrame(all_alpha_records).to_csv(
            os.path.join(REG_JACK_DIR, "jackknife_alphas_with_errors.csv"),
            index=False
        )

    if final_alpha_records:
        pd.DataFrame(final_alpha_records).to_csv(
            os.path.join(REG_JACK_DIR, "alpha_refit.csv"),
            index=False
        )

    if pc_rms_records:
        pd.DataFrame(pc_rms_records).to_csv(
            os.path.join(REG_JACK_DIR, "pc_rms_errors.csv"),
            index=False
        )

    if nrms_records:
        # IMPORTANT: masked-bin nRMS values (bins where mean_vec==0 excluded)
        pd.DataFrame(nrms_records).to_csv(
            os.path.join(REG_DIR, "nrms_errors_maskedbins.csv"),
            index=False
        )

    print(f"[Done] PCA outputs in: {PCA_DIR}")
    print(f"[Done] Regression + predicted noise outputs in: {REG_DIR}")


if __name__ == "__main__":
    main()

