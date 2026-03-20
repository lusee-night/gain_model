#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Noise SVD on grouped LuSEE session HDF5 spectra, grouped by notch-defined mod-4 phase,
stacked across all science campaigns.

Grouping rule
-------------
Files are ordered by:
    1. science campaign number   (science_1, science_2, ...)
    2. session number            (session_science_000.h5, 001, ...)

Then phase is assigned sequentially using notch:
    - if any notch == 0 in a session, that session is 0mod4
    - the next valid session is 1mod4
    - the next is 2mod4
    - the next is 3mod4
    - then back to 0mod4 unless another notch==0 session resets the phase

What this script does
---------------------
- Reads session_science_###.h5 files recursively from:
      ~/gain_model/data/h5/Payload_CPTs/Payload_CPTs_Science_h5/
- Ignores empty or malformed files
- Assigns each valid file to one of 0mod4/1mod4/2mod4/3mod4 using notch sequencing
- For each mod-4 group and product:
    - stacks all spectra rows across all campaigns/files in that group
    - removes non-finite rows
    - computes a preliminary mean spectrum
    - detects contaminated bins in the mean spectrum using a sliding-window IQR rule
    - computes first-pass residuals using that mean-stage mask
    - applies a conservative immediate-neighbor defect filter on those residuals
    - unions the mean-stage and neighbor-stage masks
    - computes final residuals
    - runs SVD
    - saves mean plots, residual examples, eigenvalues, eigvecs, PC scores
- Writes one combined PDF of all mean spectra
- Writes PDFs of the first 5 eigenvectors for each product in nmod4=1,2,3
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# NOTE: Run with: PYTHONPATH=receive uv run ~/gain_model/scripts/noise_model.py
from data import Spectra  # provided by luseepy/receive when PYTHONPATH=receive


_SESSION_NUM_RE = re.compile(r"session_science_(\d{3})\.h5$")
_CAMPAIGN_NUM_RE = re.compile(r"science_(\d+)$")

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


# ----------------------------
# Discovery / I/O helpers
# ----------------------------

def discover_session_files(session_dir: str, pattern: str) -> List[str]:
    session_dir = os.path.expanduser(session_dir)
    recursive_pattern = os.path.join(session_dir, "**", pattern)
    paths = sorted(glob.glob(recursive_pattern, recursive=True))
    return [p for p in paths if os.path.isfile(p)]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv_rows(path: str, header: List[str], rows: List[List[object]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def extract_session_number(path: str) -> int:
    m = _SESSION_NUM_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse session number from filename: {path}")
    return int(m.group(1))


def extract_campaign_name(path: str, session_dir: str) -> str:
    session_dir = os.path.abspath(os.path.expanduser(session_dir))
    path = os.path.abspath(path)
    rel = os.path.relpath(path, session_dir)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else "unknown_campaign"


def extract_campaign_number(path: str, session_dir: str) -> int:
    campaign_name = extract_campaign_name(path, session_dir)
    m = _CAMPAIGN_NUM_RE.match(campaign_name)
    if not m:
        return -1
    return int(m.group(1))


def product_dir_name(p: int) -> str:
    raw = PRODUCT_NAMES.get(p, f"prod{p}")
    safe = (
        raw.replace("×", "x")
           .replace(" ", "_")
           .replace("/", "_")
           .replace("-", "_")
    )
    return f"prod_{p:03d}_{safe}"


def product_label(p: int) -> str:
    return PRODUCT_NAMES.get(p, f"Product {p}")


# ----------------------------
# Math helpers
# ----------------------------

def fractional_residuals(
    X: np.ndarray,
    mask_bins: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (n_samples, n_bins)
    mu: mean over samples (n_bins,)
    R: (X - mu)/mu, with mu==0 treated as masked -> residuals forced to 0
    """
    mu = np.mean(X.astype(float), axis=0)

    if mask_bins is not None and mask_bins.size > 0:
        valid = (mask_bins >= 0) & (mask_bins < mu.size)
        mu = mu.copy()
        mu[mask_bins[valid]] = 0.0

    denom = mu.copy()
    zero_like = (denom == 0.0)
    denom_safe = denom.copy()
    denom_safe[zero_like] = 1.0

    R = (X.astype(float) - mu) / denom_safe
    R[:, zero_like] = 0.0
    return R, mu


def sanitize_matrix_rows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    keep_mask = np.all(np.isfinite(X), axis=1)
    X_clean = X[keep_mask]
    return X_clean, keep_mask


def svd_pca(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if R.ndim != 2:
        raise ValueError(f"R must be 2D, got shape {R.shape}")
    if R.shape[0] == 0:
        raise ValueError("R has zero rows after cleaning")
    if not np.all(np.isfinite(R)):
        raise ValueError("R contains NaN or inf before SVD")

    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    pcs = U * S
    eigvecs = Vt.T
    n = R.shape[0]
    eigenvals = (S ** 2) / (n - 1) if n > 1 else (S ** 2)
    return pcs, eigvecs, eigenvals


def detect_iqr_bad_bins_sliding(
    stat_vec: np.ndarray,
    manual_mask_bins: Optional[np.ndarray] = None,
    start_bin: int = 150,
    window_size: int = 500,
    iqr_mult: float = 4.0,
    grow_radius: int = 0,
) -> np.ndarray:
    """
    Detect contaminated bins using a sliding-window IQR rule.

    Method:
    - Keep any manual mask bins.
    - Starting at start_bin, slide a window of width window_size one bin at a time:
          [150:650], [151:651], [152:652], ...
    - In each window, compute Q1, Q3, IQR = Q3 - Q1.
    - For each bin covered by that window, record whether that bin lies outside:
          [Q1 - iqr_mult * IQR, Q3 + iqr_mult * IQR]
    - A bin is masked only if it is outside the IQR bounds in *every* sliding window
      that contains it.
    - Optionally grow the mask by +/- grow_radius bins.

    Returns sorted unique masked bin indices.
    """
    y = np.asarray(stat_vec, dtype=float).copy()
    n = y.size

    mask = np.zeros(n, dtype=bool)

    if manual_mask_bins is not None and manual_mask_bins.size > 0:
        valid = (manual_mask_bins >= 0) & (manual_mask_bins < n)
        mask[manual_mask_bins[valid]] = True

    y[~np.isfinite(y)] = np.nan

    start_bin = max(0, int(start_bin))
    window_size = max(1, int(window_size))

    if start_bin >= n:
        return np.where(mask)[0]

    covered_count = np.zeros(n, dtype=int)
    outside_count = np.zeros(n, dtype=int)

    last_start = n - window_size
    if last_start < start_bin:
        starts = [start_bin]
    else:
        starts = range(start_bin, last_start + 1)

    for lo in starts:
        hi = min(n, lo + window_size)

        window = y[lo:hi]
        finite_mask = np.isfinite(window)
        finite_vals = window[finite_mask]

        if finite_vals.size == 0:
            continue

        q1 = np.percentile(finite_vals, 25.0)
        q3 = np.percentile(finite_vals, 75.0)
        iqr = q3 - q1

        if not np.isfinite(iqr):
            continue

        lower = q1 - iqr_mult * iqr
        upper = q3 + iqr_mult * iqr

        idx = np.arange(lo, hi)
        vals = y[lo:hi]
        valid_vals = np.isfinite(vals)

        covered_count[idx[valid_vals]] += 1
        outside_local = valid_vals & ((vals < lower) | (vals > upper))
        outside_count[idx[outside_local]] += 1

    always_outside = (covered_count > 0) & (outside_count == covered_count)
    mask |= always_outside

    if grow_radius > 0:
        grown = mask.copy()
        bad_idx = np.where(mask)[0]
        for idx in bad_idx:
            lo = max(0, idx - grow_radius)
            hi = min(n, idx + grow_radius + 1)
            grown[lo:hi] = True
        mask = grown

    return np.where(mask)[0]


def mad_sigma(x: np.ndarray, axis=None) -> np.ndarray:
    """
    Robust sigma estimate from MAD.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis)
    return 1.4826 * mad


def immediate_neighbor_defect_matrix(R: np.ndarray) -> np.ndarray:
    """
    Compute per-sample defect relative to immediate neighbors.

    For interior bins:
        D[:, j] = R[:, j] - 0.5 * (R[:, j-1] + R[:, j+1])

    Edge bins are set to 0.
    """
    R = np.asarray(R, dtype=float)
    D = np.zeros_like(R)

    if R.shape[1] < 3:
        return D

    D[:, 1:-1] = R[:, 1:-1] - 0.5 * (R[:, :-2] + R[:, 2:])
    return D


def moving_median_1d(y: np.ndarray, half_window: int) -> np.ndarray:
    """
    Simple 1D moving median.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    out = np.empty(n, dtype=float)

    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        out[i] = np.median(y[lo:hi])

    return out


def detect_neighbor_bad_bins(
    R: np.ndarray,
    base_mask_bins: Optional[np.ndarray] = None,
    start_bin: int = 150,
    sigma_mult: float = 10.0,
    absolute_fraction_floor: float = 0.40,
    local_excess_thresh: float = 0.10,
    median_half_window: int = 25,
    grow_radius: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect bins whose residuals disagree with their immediate neighbors.

    Method:
    - Build defect matrix D relative to immediate-neighbor interpolation:
          D[:, j] = R[:, j] - 0.5 * (R[:, j-1] + R[:, j+1])
    - For each bin j, estimate robust sigma_j across samples using MAD.
    - Flag sample/bin entries where |D[:, j]| > sigma_mult * sigma_j.
    - Compute bad_fraction[j] = fraction of samples flagged at bin j.
    - Build a local moving-median baseline of bad_fraction.
    - Mask bin j only if:
          1) bad_fraction[j] is above an absolute floor, and
          2) bad_fraction[j] is elevated above its local neighborhood baseline.

    Returns:
      bad_bins      : 1D integer array of detected bad bins
      bad_fraction  : fraction of samples flagged at each bin
      defect_stat   : 95th percentile of |defect| per bin (for diagnostics)
    """
    R = np.asarray(R, dtype=float)
    n_samples, n_bins = R.shape

    D = immediate_neighbor_defect_matrix(R)
    absD = np.abs(D)

    sigma = mad_sigma(D, axis=0)
    sigma = np.asarray(sigma, dtype=float)

    eps = 1e-12
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > eps), sigma, np.inf)

    flagged = absD > (sigma_mult * sigma_safe[None, :])
    bad_fraction = np.mean(flagged, axis=0)

    local_base = moving_median_1d(bad_fraction, half_window=median_half_window)
    excess = bad_fraction - local_base

    bad_mask = np.zeros(n_bins, dtype=bool)

    lo = max(1, int(start_bin))
    hi = max(lo, n_bins - 1)

    bad_mask[lo:hi] = (
        (bad_fraction[lo:hi] > absolute_fraction_floor) &
        (excess[lo:hi] > local_excess_thresh)
    )

    if base_mask_bins is not None and len(base_mask_bins) > 0:
        valid = (base_mask_bins >= 0) & (base_mask_bins < n_bins)
        bad_mask[base_mask_bins[valid]] = True

    if grow_radius > 0:
        grown = bad_mask.copy()
        idxs = np.where(bad_mask)[0]
        for idx in idxs:
            a = max(0, idx - grow_radius)
            b = min(n_bins, idx + grow_radius + 1)
            grown[a:b] = True
        bad_mask = grown

    defect_stat = np.percentile(absD, 95.0, axis=0)
    return np.where(bad_mask)[0], bad_fraction, defect_stat


def detect_narrow_mean_spikes(
    mean_vec: np.ndarray,
    base_mask_bins: Optional[np.ndarray] = None,
    start_bin: int = 150,
    baseline_half_window: int = 25,
    center_exclude: int = 3,
    height_sigma_mult: float = 250.0,
    max_width: int = 5,
    return_frac: float = 0.15,
    grow_radius: int = 0,
    edge_guard_bins: int = 12,
) -> np.ndarray:
    """
    Detect narrow spike-like features in the mean spectrum conservatively,
    in both directions:
      - upward spikes
      - downward dips

    Interior bins:
      A candidate excursion bin j is masked if:
        1) it is a strict local maximum (for peaks) or minimum (for dips),
        2) it departs strongly from a local baseline,
        3) it returns toward the baseline within max_width bins total.

    Right-edge bins:
      If j is near the high-bin edge and does not have room to return on the right,
      allow masking if:
        1) it is strongly away from the local baseline,
        2) it returns toward baseline on the left within max_width bins,
        3) it is among the most extreme bins near the edge.
    """
    y = np.asarray(mean_vec, dtype=float)
    n = y.size

    mask = np.zeros(n, dtype=bool)
    if base_mask_bins is not None and len(base_mask_bins) > 0:
        valid = (base_mask_bins >= 0) & (base_mask_bins < n)
        mask[base_mask_bins[valid]] = True

    dy = np.diff(y)
    dy = dy[np.isfinite(dy)]
    if dy.size == 0:
        return np.where(mask)[0]

    rough_sigma = 1.4826 * np.median(np.abs(dy - np.median(dy)))
    if not np.isfinite(rough_sigma) or rough_sigma <= 0:
        rough_sigma = np.std(dy) if dy.size > 1 else 0.0
    if not np.isfinite(rough_sigma) or rough_sigma <= 0:
        return np.where(mask)[0]

    new_mask = np.zeros(n, dtype=bool)

    lo_j = max(start_bin, baseline_half_window + 1)
    hi_j = min(n - 2, n - 1)

    for j in range(lo_j, hi_j + 1):
        if mask[j]:
            continue
        if not np.isfinite(y[j]):
            continue

        lo = max(0, j - baseline_half_window)
        hi = min(n, j + baseline_half_window + 1)

        idx = np.arange(lo, hi)
        keep = (idx < j - center_exclude) | (idx > j + center_exclude)
        bg_vals = y[idx[keep]]
        bg_vals = bg_vals[np.isfinite(bg_vals)]

        if bg_vals.size < 10:
            continue

        baseline = np.median(bg_vals)

        is_peak = (y[j] > y[j - 1]) and (y[j] > y[j + 1])
        is_dip = (y[j] < y[j - 1]) and (y[j] < y[j + 1])

        if is_peak:
            peak_height = y[j] - baseline

            if peak_height > height_sigma_mult * rough_sigma:
                return_level = baseline + return_frac * peak_height

                left = j
                while left > max(0, j - max_width) and y[left] > return_level:
                    left -= 1
                left_returned = y[left] <= return_level

                right = j
                while right < min(n - 1, j + max_width) and y[right] > return_level:
                    right += 1
                right_returned = y[right] <= return_level

                width = right - left

                if left_returned and right_returned and width <= max_width:
                    a = max(0, left - grow_radius)
                    b = min(n, right + grow_radius + 1)
                    new_mask[a:b] = True
                    continue

                near_right_edge = (n - 1 - j) <= edge_guard_bins
                if near_right_edge and left_returned:
                    edge_lo = max(start_bin, n - max_width - edge_guard_bins)
                    edge_vals = y[edge_lo:n]
                    edge_vals = edge_vals[np.isfinite(edge_vals)]

                    if edge_vals.size >= 5:
                        edge_q90 = np.percentile(edge_vals, 90.0)
                        if y[j] >= edge_q90:
                            a = max(0, left - grow_radius)
                            b = n
                            new_mask[a:b] = True
                            continue

        if is_dip:
            dip_depth = baseline - y[j]

            if dip_depth > height_sigma_mult * rough_sigma:
                return_level = baseline - return_frac * dip_depth

                left = j
                while left > max(0, j - max_width) and y[left] < return_level:
                    left -= 1
                left_returned = y[left] >= return_level

                right = j
                while right < min(n - 1, j + max_width) and y[right] < return_level:
                    right += 1
                right_returned = y[right] >= return_level

                width = right - left

                if left_returned and right_returned and width <= max_width:
                    a = max(0, left - grow_radius)
                    b = min(n, right + grow_radius + 1)
                    new_mask[a:b] = True
                    continue

                near_right_edge = (n - 1 - j) <= edge_guard_bins
                if near_right_edge and left_returned:
                    edge_lo = max(start_bin, n - max_width - edge_guard_bins)
                    edge_vals = y[edge_lo:n]
                    edge_vals = edge_vals[np.isfinite(edge_vals)]

                    if edge_vals.size >= 5:
                        edge_q10 = np.percentile(edge_vals, 10.0)
                        if y[j] <= edge_q10:
                            a = max(0, left - grow_radius)
                            b = n
                            new_mask[a:b] = True
                            continue

    mask |= new_mask
    return np.where(mask)[0]


# ----------------------------
# Plot helpers
# ----------------------------

def plot_mean_spectrum(
    mean_vec: np.ndarray,
    out_png: str,
    title: str,
    pdf: Optional[PdfPages] = None
) -> None:
    x = np.arange(mean_vec.size)
    y = mean_vec.astype(float).copy()
    y[y == 0.0] = np.nan

    fig = plt.figure(figsize=(9, 5.5))
    plt.plot(x, y)
    plt.xlabel("Bin index (0-based)")
    plt.ylabel("Mean power")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(out_png)
    if pdf is not None:
        pdf.savefig(fig)

    plt.close(fig)


def plot_stat_with_mask(
    stat_vec: np.ndarray,
    masked_bins: np.ndarray,
    out_png: str,
    title: str,
    ylabel: str,
) -> None:
    x = np.arange(stat_vec.size)
    y = np.asarray(stat_vec, dtype=float).copy()

    plt.figure(figsize=(10, 5.5))
    plt.plot(x, y, label="Statistic")

    if masked_bins.size > 0:
        mb = masked_bins[(masked_bins >= 0) & (masked_bins < stat_vec.size)]
        plt.scatter(mb, y[mb], s=18, label="Masked bins")

    plt.xlabel("Bin index (0-based)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_eigenspectrum_first20(
    eigenvals: np.ndarray,
    out_png: str,
    title: str
) -> None:
    if eigenvals.size == 0:
        return

    n_plot = min(20, eigenvals.size)
    vals = eigenvals[:n_plot]
    idx = np.arange(1, n_plot + 1)

    eps = 1e-16
    vals_safe = np.where(vals > 0, vals, eps)

    plt.figure(figsize=(6, 4))
    plt.plot(idx, vals_safe, marker="o")
    plt.yscale("log")
    plt.xlabel("Principal component index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title(title + " (first 20 components)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_frac_residual_example(
    R: np.ndarray,
    out_png: str,
    title: str,
    max_lines: int = 12
) -> None:
    n_samples, n_bins = R.shape
    x = np.arange(n_bins)

    plt.figure(figsize=(9, 5.5))
    nplot = min(n_samples, max_lines)
    for i in range(nplot):
        y = R[i, :].astype(float).copy()
        y[y == 0.0] = np.nan
        plt.plot(x, y, linewidth=1.0, alpha=0.9)

    plt.xlabel("Bin index (0-based)")
    plt.ylabel("Fractional residual (X - mean)/mean")
    plt.title(title + f" (showing {nplot}/{n_samples} samples)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_first5_eigenvectors_to_pdf(
    eigvecs: np.ndarray,
    title: str,
    pdf: PdfPages,
    max_components: int = 5,
) -> None:
    """
    Plot the first few eigenvectors on a single page and append to a PDF.
    """
    if eigvecs.ndim != 2 or eigvecs.size == 0:
        return

    n_bins, n_comp = eigvecs.shape
    k = min(max_components, n_comp)
    x = np.arange(n_bins)

    fig = plt.figure(figsize=(10, 7))
    for i in range(k):
        plt.plot(x, eigvecs[:, i], label=f"EV{i+1}", linewidth=1.2)

    plt.xlabel("Bin index (0-based)")
    plt.ylabel("Eigenvector amplitude")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)


# ----------------------------
# Notch-based grouping helpers
# ----------------------------

def inspect_file_for_phase_assignment(path: str, session_dir_root: str) -> Optional[Dict[str, object]]:
    campaign_name = extract_campaign_name(path, session_dir_root)
    campaign_num = extract_campaign_number(path, session_dir_root)
    session_num = extract_session_number(path)

    try:
        s = Spectra(path)
        data = np.asarray(s.data)
    except Exception as e:
        print(f"[WARN] Skipping {campaign_name}/session {session_num:03d} ({path}) because it could not be loaded: {e}")
        return None

    if data.ndim != 3:
        print(
            f"[WARN] Skipping {campaign_name}/session {session_num:03d} ({path}) because "
            f"s.data has shape {data.shape}, expected (time, product, bin)"
        )
        return None

    n_time, n_prod, n_bins = data.shape
    if n_time == 0 or n_prod == 0 or n_bins == 0:
        print(
            f"[WARN] Skipping {campaign_name}/session {session_num:03d} ({path}) because "
            f"it has an empty dimension: shape={data.shape}"
        )
        return None

    try:
        notch = np.asarray(getattr(s, "notch", []))
        has_notch_zero = bool(notch.size > 0 and np.any(notch == 0))
        notch_unique = np.unique(notch).tolist() if notch.size else []
    except Exception:
        has_notch_zero = False
        notch_unique = []

    return {
        "campaign_name": campaign_name,
        "campaign_num": campaign_num,
        "session_num": session_num,
        "path": path,
        "has_notch_zero": has_notch_zero,
        "notch_unique": notch_unique,
        "shape": (n_time, n_prod, n_bins),
    }


def assign_phases_by_notch(
    paths: List[str],
    session_dir_root: str
) -> Tuple[Dict[int, List[Tuple[int, str]]], List[Dict[str, object]]]:
    inspected: List[Dict[str, object]] = []
    for path in paths:
        try:
            info = inspect_file_for_phase_assignment(path, session_dir_root)
        except Exception as e:
            print(f"[WARN] Could not inspect {path}: {e}")
            info = None
        if info is not None:
            inspected.append(info)

    inspected.sort(key=lambda d: (int(d["campaign_num"]), int(d["session_num"]), str(d["path"])))

    groups: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    assignment_rows: List[Dict[str, object]] = []

    current_phase: Optional[int] = None

    for info in inspected:
        path = str(info["path"])
        session_num = int(info["session_num"])
        campaign_name = str(info["campaign_name"])
        has_notch_zero = bool(info["has_notch_zero"])

        if has_notch_zero:
            phase = 0
            current_phase = 0
            anchor_reason = "notch==0 reset"
        else:
            if current_phase is None:
                print(f"[WARN] No prior notch==0 anchor before {campaign_name}/session {session_num:03d}; skipping")
                assignment_rows.append({
                    "campaign_name": campaign_name,
                    "campaign_num": int(info["campaign_num"]),
                    "session_num": session_num,
                    "path": path,
                    "has_notch_zero": has_notch_zero,
                    "assigned_phase": "",
                    "status": "skipped_before_first_anchor",
                    "reason": "No prior notch==0 anchor",
                    "shape": info["shape"],
                    "notch_unique": info["notch_unique"],
                })
                continue
            phase = (current_phase + 1) % 4
            current_phase = phase
            anchor_reason = "advanced_from_previous"

        groups[phase].append((session_num, path))
        assignment_rows.append({
            "campaign_name": campaign_name,
            "campaign_num": int(info["campaign_num"]),
            "session_num": session_num,
            "path": path,
            "has_notch_zero": has_notch_zero,
            "assigned_phase": phase,
            "status": "used",
            "reason": anchor_reason,
            "shape": info["shape"],
            "notch_unique": info["notch_unique"],
        })

        print(
            f"[Info] Assigned {campaign_name}/session {session_num:03d} "
            f"(notch_zero={has_notch_zero}) -> nmod4={phase}"
        )

    return dict(groups), assignment_rows


# ----------------------------
# Full data loading for a group
# ----------------------------

def load_group_data(
    group_items: List[Tuple[int, str]],
    session_dir_root: str
) -> Tuple[List[Dict[str, object]], int, int]:
    sessions_info: List[Dict[str, object]] = []
    ref_n_prod = None
    ref_n_bins = None

    for session_num, path in group_items:
        campaign_name = extract_campaign_name(path, session_dir_root)

        try:
            s = Spectra(path)
            data = np.asarray(s.data)
        except Exception as e:
            print(f"[WARN] Skipping {campaign_name}/session {session_num:03d} ({path}) because it could not be loaded: {e}")
            continue

        if data.ndim != 3:
            print(
                f"[WARN] Skipping {campaign_name}/session {session_num:03d} ({path}) because "
                f"s.data has shape {data.shape}, expected (time, product, bin)"
            )
            continue

        n_time, n_prod, n_bins = data.shape

        if n_time == 0 or n_prod == 0 or n_bins == 0:
            print(
                f"[WARN] Skipping {campaign_name}/session {session_num:03d} ({path}) because "
                f"it has an empty dimension: shape={data.shape}"
            )
            continue

        if ref_n_prod is None:
            ref_n_prod = n_prod
            ref_n_bins = n_bins
        else:
            if n_prod != ref_n_prod or n_bins != ref_n_bins:
                print(
                    f"[WARN] Skipping {campaign_name}/session {session_num:03d} ({path}) because "
                    f"its shape ({n_time}, {n_prod}, {n_bins}) does not match "
                    f"the group reference (_, {ref_n_prod}, {ref_n_bins})"
                )
                continue

        try:
            g = np.asarray(getattr(s, "gain", []))
            guniq = np.unique(g) if g.size else np.array([])
        except Exception:
            guniq = np.array([])

        try:
            notch = np.asarray(getattr(s, "notch", []))
            notch_u = np.unique(notch) if notch.size else np.array([])
        except Exception:
            notch_u = np.array([])

        sessions_info.append(
            {
                "campaign_name": campaign_name,
                "session_num": session_num,
                "path": path,
                "data": data,
                "n_time": n_time,
                "n_prod": n_prod,
                "n_bins": n_bins,
                "gain_unique": guniq,
                "notch_unique": notch_u,
            }
        )

    if not sessions_info:
        raise RuntimeError("No valid session files remained in this nmod4 group after filtering")

    return sessions_info, ref_n_prod, ref_n_bins


# ----------------------------
# Core grouped runner
# ----------------------------

def run_one_nmod4_group(
    nmod4: int,
    group_items: List[Tuple[int, str]],
    out_root: str,
    prod_list: Optional[List[int]],
    mean_pdf: Optional[PdfPages],
    eigvec_pdf: Optional[PdfPages],
    session_dir_root: str,
    assignment_rows: List[Dict[str, object]],
) -> None:
    group_name = f"nmod4_{nmod4}"
    group_out = os.path.join(out_root, group_name)
    ensure_dir(group_out)

    sessions_info, n_prod, n_bins = load_group_data(group_items, session_dir_root)

    if prod_list is None:
        products = list(range(n_prod))
    else:
        products = [p for p in prod_list if 0 <= p < n_prod]
        if not products:
            raise ValueError(f"No valid products in {prod_list}; files have {n_prod} products")

    total_rows = sum(int(si["n_time"]) for si in sessions_info)

    with open(os.path.join(group_out, "meta.txt"), "w") as f:
        f.write(f"group: {group_name}\n")
        f.write(f"n_sessions: {len(sessions_info)}\n")
        f.write(f"total_rows_stacked: {total_rows}\n")
        f.write(f"data_shape_per_session: (time, product, bin)\n")
        f.write(f"n_products: {n_prod}\n")
        f.write(f"n_bins: {n_bins}\n")
        f.write(f"products_run: {products}\n")
        f.write("\nSessions in group:\n")
        for si in sessions_info:
            gain_list = si["gain_unique"].tolist() if getattr(si["gain_unique"], "size", 0) else []
            notch_list = si["notch_unique"].tolist() if getattr(si["notch_unique"], "size", 0) else []
            f.write(
                f"  campaign={si['campaign_name']}, "
                f"session_{int(si['session_num']):03d}: "
                f"path={si['path']}, "
                f"n_time={si['n_time']}, "
                f"gain_unique={gain_list}, "
                f"notch_unique={notch_list}\n"
            )

    phase_rows = []
    for r in assignment_rows:
        if r["status"] == "used" and r["assigned_phase"] == nmod4:
            phase_rows.append([
                r["campaign_name"],
                r["campaign_num"],
                r["session_num"],
                r["assigned_phase"],
                r["has_notch_zero"],
                r["status"],
                r["reason"],
                r["path"],
                r["shape"],
                r["notch_unique"],
            ])

    write_csv_rows(
        os.path.join(group_out, "phase_assignment.csv"),
        [
            "campaign_name", "campaign_num", "session_num", "assigned_phase",
            "has_notch_zero", "status", "reason", "path", "shape", "notch_unique"
        ],
        phase_rows,
    )

    manual_mask_bins = np.array([0], dtype=int)

    for p in products:
        prod_out = os.path.join(group_out, product_dir_name(p))
        ensure_dir(prod_out)

        prod_name = product_label(p)

        X_parts: List[np.ndarray] = []
        row_meta: List[List[object]] = []

        global_row = 0
        for si in sessions_info:
            campaign_name = str(si["campaign_name"])
            session_num = int(si["session_num"])
            path = str(si["path"])
            data = np.asarray(si["data"], dtype=np.float64)

            Xi = data[:, p, :]
            X_parts.append(Xi)

            session_base = os.path.basename(path).replace(".h5", "")
            for local_row in range(Xi.shape[0]):
                row_meta.append([global_row, campaign_name, session_num, session_base, local_row])
                global_row += 1

        if not X_parts:
            print(f"[WARN] {group_name} prod {p}: no session data available; skipping")
            continue

        X = np.vstack(X_parts)

        print(f"[Info] {group_name} prod {p}: stacked X shape before cleaning = {X.shape}")
        print(f"[Info] {group_name} prod {p}: finite fraction in X = {np.isfinite(X).mean():.6f}")

        X, keep_mask = sanitize_matrix_rows(X)
        row_meta = [row_meta[i] for i in range(len(row_meta)) if keep_mask[i]]

        print(f"[Info] {group_name} prod {p}: stacked X shape after X-cleaning = {X.shape}")

        if X.shape[0] == 0:
            print(f"[WARN] {group_name} prod {p}: all stacked rows were non-finite; skipping")
            continue

        # ----------------------------
        # Stage 1: mean-spectrum mask
        # ----------------------------
        preliminary_mean = np.mean(X, axis=0)

        mean_mask_bins = detect_iqr_bad_bins_sliding(
            preliminary_mean,
            manual_mask_bins=manual_mask_bins,
            start_bin=150,
            window_size=500,
            iqr_mult=4.0,
            grow_radius=0,
        )

        mean_mask_bins = detect_narrow_mean_spikes(
            preliminary_mean,
            base_mask_bins=mean_mask_bins,
            start_bin=150,
            baseline_half_window=25,
            center_exclude=3,
            height_sigma_mult=250.0,
            max_width=5,
            return_frac=0.15,
            grow_radius=1,
            edge_guard_bins=12,
        )

        write_csv_rows(
            os.path.join(prod_out, "masked_bins_mean_stage.csv"),
            ["bin_index"],
            [[int(b)] for b in mean_mask_bins],
        )

        print(
            f"[Info] {group_name} prod {p}: mean-stage masked {len(mean_mask_bins)} bins "
            f"(including manual mask bin 0)"
        )

        plot_stat_with_mask(
            preliminary_mean,
            mean_mask_bins,
            out_png=os.path.join(prod_out, "mean_with_masked_bins.png"),
            title=f"{group_name} — {prod_name} (prod {p}) — mean-stage masked bins",
            ylabel="Mean power",
        )

        # First-pass residuals using mean-stage mask
        R1, mu1 = fractional_residuals(X, mask_bins=mean_mask_bins)

        # ----------------------------
        # Stage 2: conservative immediate-neighbor mask
        # ----------------------------
        neighbor_mask_bins, bad_fraction, defect_stat = detect_neighbor_bad_bins(
            R1,
            base_mask_bins=mean_mask_bins,
            start_bin=150,
            sigma_mult=10.0,
            absolute_fraction_floor=0.40,
            local_excess_thresh=0.10,
            median_half_window=25,
            grow_radius=0,
        )

        write_csv_rows(
            os.path.join(prod_out, "masked_bins_neighbor_stage.csv"),
            ["bin_index"],
            [[int(b)] for b in neighbor_mask_bins],
        )

        write_csv_rows(
            os.path.join(prod_out, "neighbor_bad_fraction.csv"),
            ["bin_index", "flagged_fraction", "defect_stat_p95"],
            [[i, float(bad_fraction[i]), float(defect_stat[i])] for i in range(len(bad_fraction))]
        )

        print(
            f"[Info] {group_name} prod {p}: neighbor-stage masked {len(neighbor_mask_bins)} bins"
        )

        plot_stat_with_mask(
            defect_stat,
            neighbor_mask_bins,
            out_png=os.path.join(prod_out, "neighbor_defect_with_masked_bins.png"),
            title=f"{group_name} — {prod_name} (prod {p}) — neighbor-defect masked bins",
            ylabel="95th percentile |neighbor defect|",
        )

        plot_stat_with_mask(
            bad_fraction,
            neighbor_mask_bins,
            out_png=os.path.join(prod_out, "neighbor_bad_fraction_with_masked_bins.png"),
            title=f"{group_name} — {prod_name} (prod {p}) — fraction of samples flagged",
            ylabel="Flagged sample fraction",
        )

        # Final union mask
        final_mask_bins = np.unique(
            np.concatenate([mean_mask_bins, neighbor_mask_bins])
        ).astype(int)

        write_csv_rows(
            os.path.join(prod_out, "masked_bins_final.csv"),
            ["bin_index"],
            [[int(b)] for b in final_mask_bins],
        )

        print(
            f"[Info] {group_name} prod {p}: final masked {len(final_mask_bins)} bins"
        )

        # Final residuals and mean for SVD
        R, mu = fractional_residuals(X, mask_bins=final_mask_bins)

        R, keep_mask_R = sanitize_matrix_rows(R)
        X = X[keep_mask_R]
        row_meta = [row_meta[i] for i in range(len(row_meta)) if keep_mask_R[i]]

        print(f"[Info] {group_name} prod {p}: residual matrix shape after R-cleaning = {R.shape}")

        if R.shape[0] == 0:
            print(f"[WARN] {group_name} prod {p}: no finite residual rows remained after cleaning; skipping")
            continue

        np.save(os.path.join(prod_out, "mean.npy"), mu)

        plot_mean_spectrum(
            mean_vec=mu,
            out_png=os.path.join(prod_out, "mean.png"),
            title=f"{group_name} — {prod_name} (prod {p}) — mean spectrum (masked bad bins)",
            pdf=mean_pdf,
        )

        plot_frac_residual_example(
            R,
            out_png=os.path.join(prod_out, "frac_residual_example.png"),
            title=f"{group_name} — {prod_name} (prod {p}) — fractional residuals",
            max_lines=12,
        )

        try:
            pcs, eigvecs, eigenvals = svd_pca(R)
        except Exception as e:
            print(f"[WARN] {group_name} prod {p}: SVD failed even after cleaning: {e}")
            continue

        np.save(os.path.join(prod_out, "eigvecs.npy"), eigvecs)

        if eigvec_pdf is not None:
            plot_first5_eigenvectors_to_pdf(
                eigvecs=eigvecs,
                title=f"{group_name} — {prod_name} (prod {p}) — first 5 eigenvectors",
                pdf=eigvec_pdf,
                max_components=5,
            )

        total = float(np.sum(eigenvals)) if eigenvals.size else 0.0
        eig_rows = []
        for i, lam in enumerate(eigenvals, start=1):
            ratio = float(lam / total) if total > 0 else 0.0
            eig_rows.append([i, float(lam), ratio])

        write_csv_rows(
            os.path.join(prod_out, "eigenvalues.csv"),
            ["component", "eigenvalue", "explained_variance_ratio"],
            eig_rows,
        )

        plot_eigenspectrum_first20(
            eigenvals,
            out_png=os.path.join(prod_out, "eigenspectrum_log_first20.png"),
            title=f"{group_name} — {prod_name} (prod {p}) — eigenvalue spectrum",
        )

        write_csv_rows(
            os.path.join(prod_out, "sample_index_map.csv"),
            ["sample", "campaign_name", "session_num", "session_name", "row_in_session"],
            row_meta,
        )

        pc_header = ["sample", "campaign_name", "session_num", "session_name", "row_in_session"] + [
            f"PC{i+1}" for i in range(pcs.shape[1])
        ]
        pc_rows = []
        for i in range(pcs.shape[0]):
            meta_prefix = row_meta[i]
            pc_rows.append(meta_prefix + pcs[i, :].tolist())

        write_csv_rows(os.path.join(prod_out, "pcs.csv"), pc_header, pc_rows)

        k5 = min(5, pcs.shape[1])
        pc5_header = ["sample", "campaign_name", "session_num", "session_name", "row_in_session"] + [
            f"PC{i+1}" for i in range(k5)
        ]
        pc5_rows = []
        for i in range(pcs.shape[0]):
            meta_prefix = row_meta[i]
            pc5_rows.append(meta_prefix + pcs[i, :k5].tolist())

        write_csv_rows(os.path.join(prod_out, "pcs_first5.csv"), pc5_header, pc5_rows)


# ----------------------------
# CLI main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--session-dir",
        default=os.path.expanduser("~/gain_model/data/h5/Payload_CPTs/Payload_CPTs_Science_h5"),
        help="Root directory containing science_<x> campaign subdirectories.",
    )
    ap.add_argument(
        "--pattern",
        default="session_science_*.h5",
        help="Recursive glob pattern for session HDF5 files beneath --session-dir.",
    )
    ap.add_argument(
        "--out-root",
        default=os.path.expanduser("~/gain_model/outputs/noise_svd_nmod4"),
        help="Root output directory (per-nmod4 subdirs created here).",
    )
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=999)
    ap.add_argument(
        "--products",
        default="all",
        help="Comma-separated product indices to run or 'all'. Default: all",
    )
    args = ap.parse_args()

    session_dir_root = os.path.expanduser(args.session_dir)

    files = discover_session_files(session_dir_root, args.pattern)
    if not files:
        raise SystemExit(f"No files matched recursively beneath: {session_dir_root}")

    kept: List[str] = []
    for p in files:
        try:
            n = extract_session_number(p)
        except ValueError:
            continue
        if args.start <= n <= args.end:
            kept.append(p)

    if not kept:
        raise SystemExit(f"Found files, but none in range {args.start:03d}..{args.end:03d}")

    out_root = os.path.expanduser(args.out_root)
    ensure_dir(out_root)

    if args.products.strip().lower() == "all":
        prod_list = None
    else:
        prod_list = [int(x.strip()) for x in args.products.split(",") if x.strip()]

    groups, assignment_rows = assign_phases_by_notch(kept, session_dir_root)

    global_phase_rows = []
    for r in assignment_rows:
        global_phase_rows.append([
            r["campaign_name"],
            r["campaign_num"],
            r["session_num"],
            r["assigned_phase"],
            r["has_notch_zero"],
            r["status"],
            r["reason"],
            r["path"],
            r["shape"],
            r["notch_unique"],
        ])

    write_csv_rows(
        os.path.join(out_root, "phase_assignment_all.csv"),
        [
            "campaign_name", "campaign_num", "session_num", "assigned_phase",
            "has_notch_zero", "status", "reason", "path", "shape", "notch_unique"
        ],
        global_phase_rows,
    )

    print(f"[Info] Found {len(kept)} session files in range {args.start:03d}..{args.end:03d}")
    print(f"[Info] Input root:  {session_dir_root}")
    print(f"[Info] Outputs:     {out_root}")
    print(f"[Info] Products:    {'all' if prod_list is None else prod_list}")
    print("[Info] notch-defined nmod4 groups:")
    for g in sorted(groups):
        print(f"    nmod4={g}: {len(groups[g])} files")

    mean_pdf_path = os.path.join(out_root, "mean_spectra_all_groups.pdf")

    eigvec_pdfs = {}
    for g in [1, 2, 3]:
        eigvec_pdfs[g] = PdfPages(
            os.path.join(out_root, f"first5_eigenvectors_nmod4_{g}.pdf")
        )

    try:
        with PdfPages(mean_pdf_path) as mean_pdf:
            for g in sorted(groups):
                try:
                    eigvec_pdf = eigvec_pdfs.get(g, None)

                    run_one_nmod4_group(
                        nmod4=g,
                        group_items=groups[g],
                        out_root=out_root,
                        prod_list=prod_list,
                        mean_pdf=mean_pdf,
                        eigvec_pdf=eigvec_pdf,
                        session_dir_root=session_dir_root,
                        assignment_rows=assignment_rows,
                    )
                except Exception as e:
                    print(f"[FAIL] nmod4={g}: {e}")
    finally:
        for pdf in eigvec_pdfs.values():
            pdf.close()

    print(f"[Done] Wrote combined mean-spectra PDF: {mean_pdf_path}")
    for g in [1, 2, 3]:
        if g in groups:
            print(
                f"[Done] Wrote first-5-eigenvectors PDF: "
                f"{os.path.join(out_root, f'first5_eigenvectors_nmod4_{g}.pdf')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())