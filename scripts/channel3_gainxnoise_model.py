#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Noise PCA (channel 3) with gain-weighting (predicted gain curves) + cleaning
===============================================================================

This script performs PCA (via SVD) on **channel 3 noise spectra** (L3/M3/H3),
but **first gain-weights** each CPT spectrum by multiplying it by a
**predicted gain curve** evaluated at every noise bin.

Pipeline per gain (L3/M3/H3):
  1) For each CPT directory:
       - Load power_zero_<L/M/H>_3.dat
       - Take column-wise mean over rows -> noise spectrum (length n_bins)
  2) For the same CPT and gain:
       - Load predicted gain curve anchors from:
           ~/gain_model/outputs/gain_pca_model/corrected/phase2/spectra_predictions/predicted_gains/
             <gain>_quadratic_PC12_predicted.csv
         where:
           * row 0 = MHz anchors (K values)
           * rows 1..14 = predicted gain values at those anchors, one row per CPT
  3) Convert MHz anchors to bin anchors:
         bin_anchor = MHz_anchor / 0.025
     and evaluate a **piecewise-linear** gain curve on integer bins 0..n_bins-1.
     IMPORTANT: We **truncate** strictly to the available noise bins (no extrapolation
     beyond the noise array length). (Extrapolation may still occur at the ends
     of the anchor range, but only for x in [0, n_bins-1].)
  4) Multiply:
         weighted_spectrum[bin] = noise_spectrum[bin] * gain_curve[bin]
  5) Apply the same cleaning/masking rules (IN PLACE) after weighting:
       - zero bins where (bin % 8) == 0   (includes bin 0)
       - zero bins listed in bad_bins_filecol0based_<gain>_002.csv
       - zero hard-coded manual bin ranges
  6) Stack CPT weighted spectra into matrix X (n_CPT x n_bins)
  7) Compute fractional residual matrix R:
         mu[j] = mean_cpt X[:, j]
         R[:, j] = (X[:, j] - mu[j]) / mu[j]   if mu[j] != 0
                = 0                            if mu[j] == 0  (masked/constant bins)
  8) Run SVD on R:
         R = U S V^T
         PCs = U*S
         eigvecs = V
         eigenvals = S^2/(n-1)
  9) Save PCA artifacts + diagnostic plots.

Output directory organization
-----------------------------------------------------------
Root output:
  ~/gain_model/outputs/noise_pca_ch3_gainweighted_cleaned/

1) PCA artifacts:
  pca/
    pcs/                  noise_pca_gainweighted_L3.csv, ...
    pcs_first5/           noise_pca_gainweighted_L3_first5.csv, ...
    eigenvalues/          noise_eigenvalues_gainweighted_L3.csv, ...
    eigenspectra/         noise_eigenspectrum_gainweighted_L3_log.png, ...
    bases/                L3_noise_mean_gainweighted.npy, L3_noise_eigvecs_gainweighted.npy, ...
    mean_spectra/         mean_power_gainweighted_L3.png, ...
    fractional_residuals/ fractional_residual_spectra_L3.png, ...

Notes
-----
- This script is PCA-only (no telemetry regression).
- CPT ordering must match CPT_directories.txt ordering.
- Predicted gains CSV must have the MHz row + 14 CPT rows in CPT_NAMES order.

===============================================================================
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

CPT_NAMES = [
    "CPT2", "CPT3", "CPT4", "CPT5", "CPT6", "CPT7", "CPT8",
    "CPT9", "CPT10", "CPT11", "CPT12", "CPT13", "CPT15", "CPT16",
]

GAINS_TO_RUN = ["L3", "M3", "H3"]

CPT_DIRS_TXT = os.path.expanduser("~/gain_model/scripts/CPT_directories.txt")

BAD_BINS_DIR_002 = os.path.expanduser(
    "~/gain_model/outputs/noise_ratio_cpt2_over_cpt1/bad_bins_lists_002"
)

PRED_GAIN_DIR = os.path.expanduser(
    "~/gain_model/outputs/gain_pca_model/corrected/phase2/spectra_predictions"
)

# MHz per bin
MHZ_PER_BIN = 0.025

# Manual mask ranges (inclusive, 0-based bins)
MANUAL_MASK_RANGES = [
    (375, 690),
    (765, 950),
    (1050, 1515),
]

# =============================================================================
# OUTPUT DIRECTORY ORGANIZATION
# =============================================================================

OUT_DIR = os.path.expanduser("~/gain_model/outputs/noise_pca_ch3_gainweighted_cleaned")
os.makedirs(OUT_DIR, exist_ok=True)

PCA_DIR = os.path.join(OUT_DIR, "pca")
PCS_DIR = os.path.join(PCA_DIR, "pcs")
PCS5_DIR = os.path.join(PCA_DIR, "pcs_first5")
EIGVAL_DIR = os.path.join(PCA_DIR, "eigenvalues")
EIGPLOT_DIR = os.path.join(PCA_DIR, "eigenspectra")
BASES_DIR = os.path.join(PCA_DIR, "bases")
MEAN_SPEC_DIR = os.path.join(PCA_DIR, "mean_spectra")
FRAC_RES_DIR = os.path.join(PCA_DIR, "fractional_residuals")

for d in [PCA_DIR, PCS_DIR, PCS5_DIR, EIGVAL_DIR, EIGPLOT_DIR, BASES_DIR, MEAN_SPEC_DIR, FRAC_RES_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# HELPERS: PCA MATH
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
    PCA via SVD on a matrix X that is already the data you want to decompose.

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

# =============================================================================
# HELPERS: FILE I/O + MASKING
# =============================================================================

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
    and returns a sorted unique array of ints (0-based file columns).
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
            if row and row[0].strip():
                vals.append(int(row[0]))

    return np.array(sorted(set(vals)), dtype=int) if vals else np.array([], dtype=int)


def load_noise_colmean(cpt_dir: str, gain: str) -> np.ndarray:
    """
    Load power_zero file (2D), take column-wise mean over rows -> col_means.
    """
    path = os.path.join(os.path.expanduser(cpt_dir), gain_to_power_filename(gain))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file for {gain}: {path}")

    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 1:
        raise ValueError(f"Expected at least 1 column in {path}, got {data.shape[1]}")

    return data.mean(axis=0).astype(float)


def apply_masks_inplace(vec: np.ndarray, bad_bins: np.ndarray) -> None:
    """
    Apply cleaning/masking rules IN PLACE:
      - bins % 8 == 0
      - bad_bins from CSV
      - manual ranges
    """
    n = vec.size
    idx = np.arange(n)
    vec[(idx % 8) == 0] = 0.0

    if bad_bins.size > 0:
        valid = (bad_bins >= 0) & (bad_bins < n)
        vec[bad_bins[valid]] = 0.0

    for lo, hi in MANUAL_MASK_RANGES:
        lo_eff = max(lo, 0)
        hi_eff = min(hi, n - 1)
        if lo_eff <= hi_eff:
            vec[lo_eff:hi_eff + 1] = 0.0

# =============================================================================
# HELPERS: PREDICTED GAIN CURVES (piecewise-linear in bin space)
# =============================================================================

def read_predicted_gains_csv(gain: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads <gain>_quadratic_PC12_predicted.csv:
      row0 = MHz anchors (length K)
      next 14 rows = gain values per CPT at those anchors (CPT_NAMES order)

    Returns:
      mhz_anchors: (K,)
      gains_by_cpt: (n_cpt, K)
    """
    path = os.path.join(PRED_GAIN_DIR, f"{gain}_quadratic_PC12_predicted.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing predicted gain file: {path}")

    rows = []
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if row and any(cell.strip() for cell in row):
                rows.append([float(x) for x in row])

    if len(rows) < 1 + len(CPT_NAMES):
        raise RuntimeError(
            f"{path} has {len(rows)} rows, expected at least {1 + len(CPT_NAMES)} "
            f"(1 MHz row + {len(CPT_NAMES)} CPT rows)."
        )

    mhz_anchors = np.array(rows[0], dtype=float)
    gains_by_cpt = np.array(rows[1:1 + len(CPT_NAMES)], dtype=float)

    if gains_by_cpt.shape[0] != len(CPT_NAMES):
        raise RuntimeError("CPT row count mismatch in predicted gain CSV.")
    if gains_by_cpt.shape[1] != mhz_anchors.size:
        raise RuntimeError("Anchor count mismatch between MHz row and gain rows.")

    return mhz_anchors, gains_by_cpt


def piecewise_linear_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear interpolation with linear extrapolation on both ends.
    xp must be strictly increasing.

    NOTE: Even though this supports extrapolation, we only ever call it on
    bins that exist in the noise spectrum (0..n_bins-1). That is the truncation rule.
    """
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    if xp.size < 2:
        raise ValueError("Need at least 2 anchor points for linear interpolation/extrapolation.")

    # Interior interpolation (np.interp uses constant end fill; we overwrite ends with linear)
    y = np.interp(x, xp, fp)

    # left linear extrapolation
    m_left = (fp[1] - fp[0]) / (xp[1] - xp[0])
    left = x < xp[0]
    if np.any(left):
        y[left] = fp[0] + m_left * (x[left] - xp[0])

    # right linear extrapolation
    m_right = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    right = x > xp[-1]
    if np.any(right):
        y[right] = fp[-1] + m_right * (x[right] - xp[-1])

    return y


def build_gainweighted_noise_matrix(cpt_dirs, gain: str) -> np.ndarray:
    """
    For each CPT:
      1) noise_vec = mean over rows of power_zero_<L/M/H>_3.dat  -> length n_bins
      2) gain_curve evaluated at bins 0..n_bins-1 by piecewise-linear in bin-space
         using anchors from predicted gain CSV (MHz -> bin)
      3) multiply: weighted = noise_vec * gain_curve
      4) apply masks in place (after weighting)
    Return stacked matrix (n_cpt x n_bins).
    """
    mhz_anchors, gains_by_cpt = read_predicted_gains_csv(gain)
    bin_anchors = mhz_anchors / MHZ_PER_BIN

    if not np.all(np.diff(bin_anchors) > 0):
        raise RuntimeError(f"Non-increasing bin anchors computed from MHz row for {gain}.")

    bad_bins = read_bad_bins_for_gain_002(gain)
    rows = []

    for i, (d, cpt_name) in enumerate(zip(cpt_dirs, CPT_NAMES)):
        # 1) Mean noise spectrum
        noise_vec = load_noise_colmean(d, gain)
        n_bins = noise_vec.size

        # 2) TRUNCATION: only evaluate gain at existing noise bins
        bins = np.arange(n_bins, dtype=float)  # 0..n_bins-1

        gain_fp = gains_by_cpt[i, :]  # values at anchors for this CPT
        gain_curve = piecewise_linear_extrap(bins, bin_anchors, gain_fp)

        # 3) Multiply
        weighted = noise_vec * gain_curve

        # 4) Mask/clean after weighting
        apply_masks_inplace(weighted, bad_bins)

        rows.append(weighted)

    return np.vstack(rows)

# =============================================================================
# HELPERS: PLOTTING + CSV OUTPUTS
# =============================================================================

def write_pc_csv(path: str, cpt_names, pcs: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["CPT"] + [f"PC{i+1}" for i in range(pcs.shape[1])])
        for name, row in zip(cpt_names, pcs):
            w.writerow([name] + row.tolist())


def write_first_k_pcs(path: str, cpt_names, pcs: np.ndarray, k: int = 5):
    k_use = min(k, pcs.shape[1])
    pcs_k = pcs[:, :k_use]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["CPT"] + [f"PC{i+1}" for i in range(k_use)])
        for name, row in zip(cpt_names, pcs_k):
            w.writerow([name] + row.tolist())


def write_eigenvalues_csv(path: str, eigenvals: np.ndarray):
    total = float(eigenvals.sum()) if eigenvals.size else 0.0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "eigenvalue", "explained_variance_ratio"])
        for i, lam in enumerate(eigenvals, start=1):
            ratio = float(lam / total) if total > 0 else 0.0
            w.writerow([i, float(lam), ratio])


def plot_eigen_spectrum_png(path: str, eigenvals: np.ndarray, gain: str, exclude_last: bool = False):
    if eigenvals.size == 0:
        return
    vals = eigenvals[:-1] if (exclude_last and eigenvals.size > 1) else eigenvals
    if vals.size == 0:
        return

    idx = np.arange(1, len(vals) + 1)
    eps = 1e-16
    vals_safe = np.where(vals > 0, vals, eps)

    plt.figure(figsize=(6, 4))
    plt.plot(idx, vals_safe, marker="o")
    plt.yscale("log")
    plt.xlabel("Principal component index")
    plt.ylabel("Eigenvalue (log scale)")
    title = f"Eigenvalue spectrum for {gain} (log scale)"
    if exclude_last:
        title += ", smallest removed"
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def zero_to_nan(y: np.ndarray) -> np.ndarray:
    y_plot = y.astype(float).copy()
    y_plot[y_plot == 0.0] = np.nan
    return y_plot


def _get_cpt_colors():
    return [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
        "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
    ]


def plot_mean_spectra_all_bins(matrix: np.ndarray, gain: str):
    """
    Plot gain-weighted mean spectra per CPT (masked bins hidden).
    """
    n_cpt, n_bins = matrix.shape
    x = np.arange(0, n_bins)
    colors = _get_cpt_colors()

    plt.figure(figsize=(9, 6))
    for row, label, color in zip(matrix, CPT_NAMES, colors):
        plt.plot(x, zero_to_nan(row), label=label, color=color)

    max_mhz = (n_bins - 1) * MHZ_PER_BIN
    plt.xlabel(f"Bin index (0-based); max freq ≈ {max_mhz:.3f} MHz")
    plt.ylabel("Mean power × predicted gain")
    plt.title(f"Gain-weighted mean spectrum per CPT for {gain} (masked bins hidden)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out_path = os.path.join(MEAN_SPEC_DIR, f"mean_power_gainweighted_{gain}.png")
    plt.savefig(out_path)
    plt.close()


def plot_fractional_residual_spectra(frac_mat: np.ndarray, gain: str, out_dir: str, mhz_per_bin: float = MHZ_PER_BIN):
    """
    Plot fractional residual spectra per CPT:
      y = (x - mu)/mu
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
# MAIN
# =============================================================================

def main():
    """
    Loads CPT directories, runs gain-weighted noise PCA for L3/M3/H3,
    and writes PCA artifacts under OUT_DIR.
    """
    with open(CPT_DIRS_TXT, "r") as f:
        cpt_dirs = [line.strip() for line in f if line.strip()]

    if len(cpt_dirs) != len(CPT_NAMES):
        raise RuntimeError(
            f"Mismatch: CPT_directories.txt has {len(cpt_dirs)} dirs, "
            f"but CPT_NAMES has {len(CPT_NAMES)} entries."
        )

    for gain in GAINS_TO_RUN:
        print(f"[Noise SVD-PCA] Processing {gain} (channel 3, gain-weighted, truncated to noise bins)...")

        # ---------------------------------------------------------------------
        # Build gain-weighted + cleaned matrix X
        # ---------------------------------------------------------------------
        matrix = build_gainweighted_noise_matrix(cpt_dirs, gain)
        n_bins = matrix.shape[1]
        max_mhz = (n_bins - 1) * MHZ_PER_BIN
        print(f"  Using n_bins={n_bins} (max freq ≈ {max_mhz:.3f} MHz). No evaluation beyond this.")

        # ---------------------------------------------------------------------
        # Fractional residuals R and mean spectrum mu
        # ---------------------------------------------------------------------
        frac_mat, mean_vec = fractional_residual_matrix(matrix)

        # Diagnostic plot: fractional residual spectra
        plot_fractional_residual_spectra(frac_mat, gain, FRAC_RES_DIR, mhz_per_bin=MHZ_PER_BIN)

        # ---------------------------------------------------------------------
        # PCA via SVD on fractional residual matrix
        # ---------------------------------------------------------------------
        pcs, eigvecs, eigenvals = compute_pca_svd_from_matrix(frac_mat)

        # ---------------------------------------------------------------------
        # Save PCA artifacts
        # ---------------------------------------------------------------------
        pcs_csv = os.path.join(PCS_DIR, f"noise_pca_gainweighted_{gain}.csv")
        write_pc_csv(pcs_csv, CPT_NAMES, pcs)

        pcs5_csv = os.path.join(PCS5_DIR, f"noise_pca_gainweighted_{gain}_first5.csv")
        write_first_k_pcs(pcs5_csv, CPT_NAMES, pcs, k=5)

        eig_csv = os.path.join(EIGVAL_DIR, f"noise_eigenvalues_gainweighted_{gain}.csv")
        write_eigenvalues_csv(eig_csv, eigenvals)

        eig_png = os.path.join(EIGPLOT_DIR, f"noise_eigenspectrum_gainweighted_{gain}_log.png")
        plot_eigen_spectrum_png(eig_png, eigenvals, gain, exclude_last=False)

        eig_trim_png = os.path.join(EIGPLOT_DIR, f"noise_eigenspectrum_gainweighted_{gain}_log_exclude_last.png")
        plot_eigen_spectrum_png(eig_trim_png, eigenvals, gain, exclude_last=True)

        np.save(os.path.join(BASES_DIR, f"{gain}_noise_mean_gainweighted.npy"), mean_vec)
        np.save(os.path.join(BASES_DIR, f"{gain}_noise_eigvecs_gainweighted.npy"), eigvecs)

        # Plot gain-weighted mean spectra per CPT
        plot_mean_spectra_all_bins(matrix, gain)

        print(f"[Noise SVD-PCA] Finished {gain}.")

    print(f"[Done] Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
