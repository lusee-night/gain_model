#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pairwise telemetry scatter QA for LuSEE gain-modeling (with CPT1).

This script reads the per-gain-setting telemetry-mean CSVs produced by the
telemetry aggregation step (one CSV per setting: L0..H3). For each setting,
it generates 2D scatter plots for all telemetry column pairs (x,y), colored by
a “Z” value (the gain-column mean for that setting, pulled from a separate table).

Core idea (outlier gating via χ²):
  • Treat all CPTs except targets (default: CPT1 and CPT8) as a “background” cloud.
  • Estimate the background mean μ and covariance Σ in the (x,y) plane.
  • For each target CPT, compute χ² = (vᵀ Σ⁻¹ v), where v = [x,y] - μ.
  • Export a plot if CPT1 or CPT8 exceeds a χ² threshold (default: χ² > 6).

Outputs (three PDFs per run_pipeline call):
  1) A gated PDF: only “interesting” (x,y) pairs passing the χ² gate.
  2) A summary PDF: (a) pairs significant for CPT1 in ALL settings, and
     (b) Top-25 pairs ranked by average χ²(CPT1) across settings (forced export).
  3) A fixed-pair PDF: always plots a specific telemetry pair across all settings.

The script runs two passes:
  • Pass 1: no exclusions (original behavior)
  • Pass 2: excludes any pair involving SPE_1VAD8_C (produces short-named PDFs)
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless (no GUI) backend so this can run on servers/CI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from collections import defaultdict

# -------- Paths / Config --------
# INPUT_DIR: directory containing per-setting telemetry-mean CSVs (with CPT1 included)
INPUT_DIR = os.path.expanduser(
    "~/gain_model/outputs/telemetry_per_gainsetting_withCPT1"
)

# Z_TABLE_PATH: a separate table giving the “Z” color value per CPT for each setting.
# Expected columns: CPT + one column per setting (e.g. L0, M0, ..., H3)
Z_TABLE_PATH = os.path.expanduser(
    "~/gain_model/outputs/other/means/gain_column_means_by_CPT.csv"
)

# -------- Helpers (unchanged) --------
# Regex to infer the setting from filenames like "L0_withCPT1.csv" (case-insensitive)
SETTING_RE = re.compile(r"\b([LMH][0-3])(?:_withCPT1)?\.csv$", re.IGNORECASE)

def infer_setting_from_filename(fname: str) -> str | None:
    """Extract setting label (e.g. L0/M2/H3) from a CSV filename."""
    m = SETTING_RE.search(os.path.basename(fname))
    return m.group(1).upper() if m else None

def normalize_cpt_label(s: str) -> str:
    """
    Normalize CPT labels to a canonical form like 'CPT1', 'CPT8', etc.
    Handles variants like 'cpt 01' or 'CPT0008' → 'CPT1'/'CPT8'.
    """
    s = s.strip()
    m = re.match(r"^(CPT)\s*0*([1-9]\d*)$", s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2))}"
    return s.upper()

def pick_z_column_for_setting(setting: str) -> str:
    """
    Choose which Z-table column should color the scatter for this setting.
    Here it’s a 1:1 mapping: Z column name == setting (e.g. 'L0').
    """
    return setting

def load_z_series(z_table_path: str, z_col: str) -> dict:
    """
    Load the Z-table and return a dict mapping CPT label → Z value for one setting.
    This is used to color points in the (x,y) scatter plot.
    """
    zdf = pd.read_csv(z_table_path)
    if "CPT" not in zdf.columns or z_col not in zdf.columns:
        raise ValueError(
            f"Z table missing required columns. Found: {list(zdf.columns)}; need 'CPT' and '{z_col}'."
        )
    zmap = {}
    for _, row in zdf.iterrows():
        cpt = normalize_cpt_label(str(row["CPT"]))
        zmap[cpt] = row[z_col]
    return zmap

def is_numeric_series(s: pd.Series) -> bool:
    """Return True if the series has a numeric dtype (used to filter telemetry columns)."""
    return pd.api.types.is_numeric_dtype(s)

def compute_chi2_for_targets(x: np.ndarray,
                             y: np.ndarray,
                             labels: np.ndarray,
                             targets=("CPT1", "CPT8"),
                             min_bg_points=3,
                             ridge_eps=1e-9):
    """
    Compute Mahalanobis χ² for selected target CPTs in a 2D (x,y) plane.

    Steps:
      1) Build an (N,2) matrix of points.
      2) Define 'background' as all points NOT in targets.
      3) Estimate background mean μ and covariance Σ from background points.
      4) For each target t, compute v = [x_t, y_t] - μ and χ² = vᵀ Σ⁻¹ v.

    The function returns:
      • chi2[target] values
      • μ, Σ, and Σ⁻¹ (used later for ellipse drawing)
    """
    # Stack x,y into points; remove any rows with NaNs/Infs
    xy = np.column_stack([x, y])
    finite_mask = np.isfinite(xy).all(axis=1)
    xy = xy[finite_mask]
    lab = labels[finite_mask]

    # Default result structure if we can't compute a stable covariance
    result = {"chi2": {t: np.nan for t in targets}, "mu": None, "cov": None, "cov_inv": None}

    # Need enough total points to do anything meaningful
    if xy.shape[0] < min_bg_points:
        return result

    # Background = non-target CPTs
    bg_mask = ~np.isin(lab, list(targets))
    bg_xy = xy[bg_mask]
    if bg_xy.shape[0] < min_bg_points:
        return result

    # Background mean and covariance in 2D
    mu = np.nanmean(bg_xy, axis=0)
    cov = np.cov(bg_xy.T, ddof=1)

    # Validate covariance shape/finite
    if not (isinstance(cov, np.ndarray) and cov.shape == (2, 2)) or not np.all(np.isfinite(cov)):
        return result

    # If covariance is near-singular (tiny determinant), add a small ridge to stabilize inversion
    det = np.linalg.det(cov)
    if not np.isfinite(det) or det < 1e-12:
        ridge = (1e-6 * float(np.trace(cov))) + ridge_eps
        cov = cov + ridge * np.eye(2)

    # Invert covariance (fallback to pseudo-inverse if needed)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # Compute χ² for each target (if present)
    for t in targets:
        idx = np.where(lab == t)[0]
        if idx.size == 0:
            continue
        v = xy[idx[0]] - mu
        chi2 = float(v.T @ cov_inv @ v)
        result["chi2"][t] = chi2

    result["mu"] = mu
    result["cov"] = cov
    result["cov_inv"] = cov_inv
    return result

def draw_gaussian_ellipses(ax, mu: np.ndarray, cov: np.ndarray,
                           levels=(5.99,), facecolor="blue",
                           edgecolor="black", alpha=0.15, linewidth=1.2):
    """
    Draw confidence ellipses for a 2D Gaussian with mean μ and covariance Σ.

    For a 2D Gaussian, contours of constant Mahalanobis distance satisfy:
        (x-μ)ᵀ Σ⁻¹ (x-μ) = c
    where c is a χ² value with 2 DOF.
    Example: c=5.99 corresponds to ~95% region (2 DOF).

    We compute eigenvalues/vectors of Σ to get ellipse axes and orientation.
    """
    if mu is None or cov is None:
        return

    # Eigen-decomposition of covariance; eigenvalues give axis variances
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)  # guard against tiny negative numerical artifacts

    for c in levels:
        # Radii scale like sqrt(c * eigenvalue)
        radii = np.sqrt(c * vals)

        # Orientation from eigenvectors
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        # Matplotlib ellipse uses full width/height (diameters)
        ell = mpatches.Ellipse(
            xy=mu,
            width=2 * radii[0],
            height=2 * radii[1],
            angle=angle,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.add_patch(ell)

def plot_pair_2d(pdf, df: pd.DataFrame, xcol: str, ycol: str, zmap: dict,
                 title_prefix: str, force_export: bool = False,
                 extra_note: str | None = None):
    """
    Create one scatter plot page for a specific (xcol,ycol) telemetry pair.

    • Points = CPTs in this setting's CSV
    • Color = Z value from zmap (typically the gain mean for this setting)
    • Gate = export only if CPT1 or CPT8 has χ² > 6 (unless force_export=True)
    • Annotate CPT1 and CPT8 markers + χ² values
    • Draw a 95% covariance ellipse of the background cloud (non-target CPTs)
    """
    # Pull only the needed columns and drop rows with missing x/y
    sub = df[["CPT", xcol, ycol]].dropna()
    if sub.empty:
        return

    # Normalize CPT labels and extract x,y arrays
    cpts = sub["CPT"].astype(str).map(normalize_cpt_label).values
    x = sub[xcol].values
    y = sub[ycol].values

    # Map CPT → Z color value, and drop CPTs without Z
    z = np.array([zmap.get(cpt, np.nan) for cpt in cpts], dtype=float)
    mask = ~np.isnan(z)
    if mask.sum() < 2:
        return
    x, y, z, cpts = x[mask], y[mask], z[mask], cpts[mask]

    # Skip degenerate cases where one axis is essentially constant zero
    if np.allclose(x, 0.0) or np.allclose(y, 0.0):
        return

    # Compute χ² for CPT1 and CPT8 relative to background (non-target) points
    res = compute_chi2_for_targets(x, y, cpts, targets=("CPT1", "CPT8"))
    chi2_cpt1 = res["chi2"].get("CPT1", np.nan)
    chi2_cpt8 = res["chi2"].get("CPT8", np.nan)

    # Gate: only export if CPT1 or CPT8 is sufficiently “outlying”
    passes_gate = (
        (np.isfinite(chi2_cpt1) and chi2_cpt1 > 6.0) or
        (np.isfinite(chi2_cpt8) and chi2_cpt8 > 6.0)
    )
    if not (passes_gate or force_export):
        return

    # --- Build the figure ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter colored by Z (gain mean for this setting)
    scatter = ax.scatter(x, y, c=z, cmap='viridis', s=50, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f"Z = gain means ({title_prefix.split()[0]})")

    # Draw the 2D Gaussian ellipse for the background covariance
    draw_gaussian_ellipses(ax, res["mu"], res["cov"],
                           levels=(5.99,), facecolor="blue",
                           edgecolor="black", alpha=0.15, linewidth=1.2)

    # Highlight CPT1 and CPT8 with distinctive markers + annotations
    for xi, yi, cpt in zip(x, y, cpts):
        nc = normalize_cpt_label(cpt)
        if nc == "CPT1":
            ax.scatter(xi, yi, facecolors='none', edgecolors='red',
                       s=160, linewidth=2, marker='o')
            label = f"CPT1 (χ²={chi2_cpt1:.2f})" if np.isfinite(chi2_cpt1) else "CPT1"
            ax.annotate(label, (xi, yi), xytext=(8, 8), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='red')
        elif nc == "CPT8":
            ax.scatter(xi, yi, facecolors='none', edgecolors='green',
                       s=160, linewidth=2, marker='s')
            label = f"CPT8 (χ²={chi2_cpt8:.2f})" if np.isfinite(chi2_cpt8) else "CPT8"
            ax.annotate(label, (xi, yi), xytext=(8, 8), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='green')

    # Axis labels + title formatting
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    title = f"{title_prefix}  |  x={xcol}  y={ycol}  z={title_prefix.split()[0]}"
    if extra_note:
        title += f"  —  {extra_note}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Write this page into the active PDF
    pdf.savefig(fig)
    plt.close(fig)

# -------- Pipeline (parametric) --------
def run_pipeline(output_pdf_main: str,
                 output_pdf_all: str,
                 output_pdf_fixed: str,
                 exclude_cols: set[str] | None = None):
    """
    Run the full plotting pipeline and write three PDFs.

    Parameters
    ----------
    output_pdf_main:
        Gated pairwise plots (only pairs where CPT1 or CPT8 is outlying by χ²).
    output_pdf_all:
        “Summary/forced” plots including:
          • pairs where CPT1 is significant in ALL settings
          • Top-25 pairs ranked by average χ²(CPT1) across settings
    output_pdf_fixed:
        Plots of one fixed telemetry pair across all settings (forced export).
    exclude_cols:
        If provided, any (x,y) pair involving any excluded column is skipped.
        This is used to produce a second run without problematic telemetry fields.
    """
    exclude_cols = set(exclude_cols or [])
    os.makedirs(os.path.dirname(output_pdf_main), exist_ok=True)

    # Discover input per-setting CSVs (expected naming: *_withCPT1.csv)
    csv_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*_withCPT1.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No '*_withCPT1.csv' found in {INPUT_DIR}")

    # Bookkeeping structures used across the three PDF exports:
    setting_to_path: dict[str, str] = {}                            # setting → CSV path
    cpt1_sig_pairs_by_setting: dict[str, dict[tuple[str, str], float]] = {}  # setting → {(x,y): χ²}
    significant_pairs_by_setting: dict[str, set[tuple[str, str]]] = {}       # setting → {(x,y)} passing CPT1 or CPT8 gate
    cpt1_chi2_by_setting: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)  # setting → {(x,y): χ²(CPT1)}
    chi2_across_settings: dict[tuple[str, str], list[float]] = defaultdict(list)       # (x,y) → [χ²(CPT1) values over settings]

    # -------- PDF #1: gated plots --------
    # For each setting, iterate over all telemetry column pairs and export plots
    # only when CPT1 or CPT8 is sufficiently far from the background cluster.
    with PdfPages(output_pdf_main) as pdf:
        for path in csv_paths:
            setting = infer_setting_from_filename(path)
            if setting is None:
                print(f"[WARN] Could not infer setting from filename: {path}")
                continue

            setting_to_path[setting] = path
            print(f"[INFO] Processing {os.path.basename(path)} (setting={setting})")

            df = pd.read_csv(path)
            if "CPT" not in df.columns:
                print(f"[WARN] Skipping (no CPT column): {path}")
                continue

            # Normalize CPT labels so "CPT01" etc. become canonical
            df["CPT"] = df["CPT"].astype(str).map(normalize_cpt_label)

            # Z values for this setting come from the corresponding column in the Z-table
            z_col = pick_z_column_for_setting(setting)
            zmap = load_z_series(Z_TABLE_PATH, z_col)

            # Select usable telemetry columns:
            #   • numeric dtype
            #   • at least 2 non-NaN entries (otherwise no scatter)
            telem_cols = [c for c in df.columns if c != "CPT" and is_numeric_series(df[c])]
            telem_cols = [c for c in telem_cols if df[c].notna().sum() >= 2]

            # Optional global exclusion (removes columns from consideration entirely)
            if exclude_cols:
                telem_cols = [c for c in telem_cols if c not in exclude_cols]

            n = len(telem_cols)
            if n < 2:
                print(f"[WARN] Not enough numeric columns for pair plotting: {path}")
                continue

            title_prefix = f"{setting} (withCPT1)"
            cpt1_sig_pairs_by_setting.setdefault(setting, {})
            significant_pairs_by_setting.setdefault(setting, set())

            # Iterate over all unique column pairs (i<j)
            for i in range(n):
                for j in range(i + 1, n):
                    xcol, ycol = telem_cols[i], telem_cols[j]

                    # Extra safety (should be redundant after list filter)
                    if xcol in exclude_cols or ycol in exclude_cols:
                        continue

                    # Compute χ² (CPT1/CPT8) for this pair so we can:
                    #   • gate plots
                    #   • record χ² statistics for later aggregation
                    sub = df[["CPT", xcol, ycol]].dropna()
                    if sub.empty:
                        continue

                    cpts = sub["CPT"].astype(str).map(normalize_cpt_label).values
                    x = sub[xcol].values
                    y = sub[ycol].values

                    # Drop CPTs lacking Z values for this setting
                    z = np.array([zmap.get(cpt, np.nan) for cpt in cpts], dtype=float)
                    mask = ~np.isnan(z)
                    if mask.sum() < 2:
                        continue

                    x_m, y_m, cpts_m = x[mask], y[mask], cpts[mask]

                    # Skip degenerate axis cases
                    if np.allclose(x_m, 0.0) or np.allclose(y_m, 0.0):
                        continue

                    res = compute_chi2_for_targets(x_m, y_m, cpts_m, targets=("CPT1", "CPT8"))
                    chi2_cpt1 = res["chi2"].get("CPT1", np.nan)
                    chi2_cpt8 = res["chi2"].get("CPT8", np.nan)

                    # Record χ²(CPT1) for later ranking/averaging across settings
                    if np.isfinite(chi2_cpt1):
                        cpt1_chi2_by_setting[setting][(xcol, ycol)] = chi2_cpt1
                        chi2_across_settings[(xcol, ycol)].append(float(chi2_cpt1))
                    else:
                        cpt1_chi2_by_setting[setting][(xcol, ycol)] = np.nan

                    # Store pairs where CPT1 is significant (used to find “universal” pairs)
                    if np.isfinite(chi2_cpt1) and chi2_cpt1 > 6.0:
                        cpt1_sig_pairs_by_setting[setting][(xcol, ycol)] = chi2_cpt1

                    # Store pairs where CPT1 OR CPT8 passes the gate (used mainly for bookkeeping)
                    if ((np.isfinite(chi2_cpt1) and chi2_cpt1 > 6.0) or
                        (np.isfinite(chi2_cpt8) and chi2_cpt8 > 6.0)):
                        significant_pairs_by_setting[setting].add((xcol, ycol))

                    # Actually produce the plot page (gated inside plot_pair_2d)
                    try:
                        plot_pair_2d(pdf, df, xcol, ycol, zmap, title_prefix,
                                     force_export=False, extra_note=None)
                    except Exception as e:
                        print(f"[WARN] plot failed for {setting} x={xcol} y={ycol}: {e}")

    print(f"[OK] Wrote PDF → {output_pdf_main}")

    # -------- Aggregations --------
    # 1) universal_pairs: telemetry pairs that are CPT1-significant (χ²>6) in ALL settings.
    #    We compute an intersection across settings of each setting’s CPT1-significant pairs.
    settings_present = sorted(cpt1_sig_pairs_by_setting.keys())
    universal_pairs = None
    for s in settings_present:
        pairs_s = set(cpt1_sig_pairs_by_setting[s].keys())
        universal_pairs = pairs_s if universal_pairs is None else (universal_pairs & pairs_s)

    # 2) avg_chi2_per_pair: average χ²(CPT1) for a given (x,y) across settings (where finite).
    avg_chi2_per_pair = {}
    for pair, vals in chi2_across_settings.items():
        if len(vals) == 0:
            continue
        avg_chi2_per_pair[pair] = float(np.mean(vals))

    # 3) top25_pairs: top 25 by average χ²(CPT1) across settings.
    top25_pairs = []
    if avg_chi2_per_pair:
        top25_pairs = [p for p, _ in sorted(avg_chi2_per_pair.items(),
                                            key=lambda kv: kv[1],
                                            reverse=True)[:25]]

    # -------- PDF #2 --------
    # This PDF has two “forced export” sections:
    #   A) Pairs significant for CPT1 in ALL settings (universal_pairs).
    #   B) Top-25 pairs by average χ²(CPT1) across settings, plotted in every setting.
    with PdfPages(output_pdf_all) as pdf_all:
        # ---- Section A: universal CPT1-significant pairs ----
        if universal_pairs and len(universal_pairs) > 0:
            universal_pairs = sorted(universal_pairs)

            for setting in settings_present:
                # Title page for this setting
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.axis('off')
                ax.text(0.5, 0.6, f"{setting} (withCPT1)",
                        ha='center', va='center', fontsize=22)
                ax.text(0.5, 0.45, f"Pairs CPT1-significant (χ²>6) in ALL settings",
                        ha='center', va='center', fontsize=12)
                pdf_all.savefig(fig); plt.close(fig)

                # Load the setting CSV
                path = setting_to_path.get(setting)
                if not path:
                    print(f"[WARN] Missing CSV path for setting {setting}; skipping.")
                    continue

                df = pd.read_csv(path)
                if "CPT" not in df.columns:
                    print(f"[WARN] Skipping (no CPT column): {path}")
                    continue

                df["CPT"] = df["CPT"].astype(str).map(normalize_cpt_label)
                z_col = pick_z_column_for_setting(setting)
                zmap = load_z_series(Z_TABLE_PATH, z_col)
                title_prefix = f"{setting} (withCPT1)"

                # Force-export every universal pair for this setting
                for (xcol, ycol) in universal_pairs:
                    if xcol in exclude_cols or ycol in exclude_cols:
                        continue
                    chi2_here = cpt1_sig_pairs_by_setting.get(setting, {}).get((xcol, ycol), np.nan)
                    note = f"CPT1 χ²={chi2_here:.2f}" if np.isfinite(chi2_here) else None
                    try:
                        plot_pair_2d(pdf_all, df, xcol, ycol, zmap, title_prefix,
                                     force_export=True, extra_note=note)
                    except Exception as e:
                        print(f"[WARN] ALL-settings plot failed for {setting} x={xcol} y={ycol}: {e}")

            # Summary list page of universal pairs (human-readable checklist)
            kept_pairs = [p for p in universal_pairs if p[0] not in exclude_cols and p[1] not in exclude_cols]
            if kept_pairs:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.axis('off')
                ax.text(0.5, 0.9, "Pairs CPT1-significant (χ²>6) in ALL settings",
                        ha='center', va='center', fontsize=16)
                y = 0.82
                for (xcol, ycol) in kept_pairs:
                    ax.text(0.1, y, f"• {xcol} vs {ycol}", fontsize=10, va='top')
                    y -= 0.035
                    if y < 0.08:
                        pdf_all.savefig(fig); plt.close(fig)
                        fig, ax = plt.subplots(figsize=(8, 6)); ax.axis('off'); y = 0.92
                pdf_all.savefig(fig); plt.close(fig)
                print(f"[OK] Wrote CPT1-significant-in-ALL-settings section.")
            else:
                fig, ax = plt.subplots(figsize=(8, 6)); ax.axis('off')
                ax.text(0.5, 0.5, "No universal pairs remained after exclusions.",
                        ha='center', va='center', fontsize=14)
                pdf_all.savefig(fig); plt.close(fig)
        else:
            # If no universal pairs exist, write a single info page
            fig, ax = plt.subplots(figsize=(8, 6)); ax.axis('off')
            ax.text(0.5, 0.5, "No (x, y) pairs had CPT1 χ²>6 in ALL settings.",
                    ha='center', va='center', fontsize=14)
            pdf_all.savefig(fig); plt.close(fig)
            print("[INFO] No universal CPT1-significant pairs; wrote an info page.")

        # ---- Section B: Top-25 by average χ²(CPT1) ----
        kept_top25 = [p for p in top25_pairs if p[0] not in exclude_cols and p[1] not in exclude_cols]
        if kept_top25:
            # Intro page for this section
            fig, ax = plt.subplots(figsize=(8, 6)); ax.axis('off')
            ax.text(0.5, 0.6, "Top 25 pairs by average χ²(CPT1) across settings",
                    ha='center', va='center', fontsize=18)
            ax.text(0.5, 0.45, "Each pair is plotted for every setting (forced export).",
                    ha='center', va='center', fontsize=11)
            pdf_all.savefig(fig); plt.close(fig)

            # For each top pair: write a “pair header” page, then plot it across all settings
            for (xcol, ycol) in kept_top25:
                avg_val = avg_chi2_per_pair.get((xcol, ycol), np.nan)

                fig, ax = plt.subplots(figsize=(8, 6)); ax.axis('off')
                ax.text(0.5, 0.65, f"Pair: {xcol} vs {ycol}",
                        ha='center', va='center', fontsize=16)
                if np.isfinite(avg_val):
                    ax.text(0.5, 0.5, f"Average χ²(CPT1) across settings: {avg_val:.2f}",
                            ha='center', va='center', fontsize=12)
                pdf_all.savefig(fig); plt.close(fig)

                # Plot this pair for each setting (forced export)
                for setting in settings_present:
                    path = setting_to_path.get(setting)
                    if not path:
                        continue
                    df = pd.read_csv(path)
                    if "CPT" not in df.columns:
                        continue
                    df["CPT"] = df["CPT"].astype(str).map(normalize_cpt_label)
                    z_col = pick_z_column_for_setting(setting)
                    zmap = load_z_series(Z_TABLE_PATH, z_col)
                    title_prefix = f"{setting} (withCPT1)"

                    chi2_here = cpt1_chi2_by_setting.get(setting, {}).get((xcol, ycol), np.nan)
                    note_bits = []
                    if np.isfinite(chi2_here):
                        note_bits.append(f"{setting} CPT1 χ²={chi2_here:.2f}")
                    if np.isfinite(avg_val):
                        note_bits.append(f"avg χ²={avg_val:.2f}")
                    note = " | ".join(note_bits) if note_bits else None

                    try:
                        plot_pair_2d(pdf_all, df, xcol, ycol, zmap, title_prefix,
                                     force_export=True, extra_note=note)
                    except Exception as e:
                        print(f"[WARN] Top-25 plot failed for {setting} x={xcol} y={ycol}: {e}")

            # Final summary list page (ranked)
            fig, ax = plt.subplots(figsize=(8, 6)); ax.axis('off')
            ax.text(0.5, 0.9, "Top 25 by average χ²(CPT1) across settings",
                    ha='center', va='center', fontsize=16)
            y = 0.82; rank = 1
            for pair in kept_top25:
                avg_val = avg_chi2_per_pair.get(pair, np.nan)
                xcol, ycol = pair
                line = f"{rank:>2}. {xcol} vs {ycol} — avg χ²={avg_val:.2f}" if np.isfinite(avg_val) else \
                       f"{rank:>2}. {xcol} vs {ycol} — avg χ²=NaN"
                ax.text(0.1, y, line, fontsize=10, va='top'); y -= 0.035; rank += 1
                if y < 0.08:
                    pdf_all.savefig(fig); plt.close(fig)
                    fig, ax = plt.subplots(figsize=(8, 6)); ax.axis('off'); y = 0.92
            pdf_all.savefig(fig); plt.close(fig)
            print("[OK] Wrote Top-25-by-average-χ² section.")
        else:
            # If we can't rank anything (no finite χ² values), write an info page
            fig, ax = plt.subplots(figsize=(8, 6)); ax.axis('off')
            ax.text(0.5, 0.5, "No pairs had finite χ²(CPT1) to rank for Top 25 (after exclusions).",
                    ha='center', va='center', fontsize=14)
            pdf_all.savefig(fig); plt.close(fig)
            print("[INFO] No Top-25 list produced; insufficient χ² data after exclusions.")

        print(f"[OK] Wrote PDF → {output_pdf_all}")

    # -------- PDF #3: fixed pair (unchanged variables) --------
    # This section is a “standard diagnostic” pair you always want across settings.
    x_fixed, y_fixed = "VMON_1V2D", "SPE_1VAD8_V"
    with PdfPages(output_pdf_fixed) as pdf_fixed:
        # Intro page describing the fixed pair
        fig, ax = plt.subplots(figsize=(8, 6)); ax.axis('off')
        ax.text(0.5, 0.6, f"Fixed pair across all settings:", ha='center', va='center', fontsize=18)
        ax.text(0.5, 0.48, f"x = {x_fixed}    y = {y_fixed}", ha='center', va='center', fontsize=13)
        pdf_fixed.savefig(fig); plt.close(fig)

        # If exclusions remove either variable, skip plotting and document why
        if x_fixed in exclude_cols or y_fixed in exclude_cols:
            fig, ax = plt.subplots(figsize=(8, 6)); ax.axis('off')
            ax.text(0.5, 0.5, "Fixed-pair section skipped by exclusions.",
                    ha='center', va='center', fontsize=14)
            pdf_fixed.savefig(fig); plt.close(fig)
            print(f"[INFO] Skipped fixed-pair plots due to exclusions.")
        else:
            # Plot fixed pair for every setting present
            if not setting_to_path:
                print("[WARN] No settings to plot for fixed pair.")
            for setting in sorted(setting_to_path.keys()):
                path = setting_to_path.get(setting)
                if not path:
                    continue
                df = pd.read_csv(path)
                if "CPT" not in df.columns:
                    continue
                df["CPT"] = df["CPT"].astype(str).map(normalize_cpt_label)
                z_col = pick_z_column_for_setting(setting)
                zmap = load_z_series(Z_TABLE_PATH, z_col)
                title_prefix = f"{setting} (withCPT1)"

                # If we computed χ²(CPT1) for this exact fixed pair earlier, annotate it
                chi2_here = cpt1_chi2_by_setting.get(setting, {}).get((x_fixed, y_fixed), np.nan)
                note = f"CPT1 χ²={chi2_here:.2f}" if np.isfinite(chi2_here) else None

                try:
                    plot_pair_2d(pdf_fixed, df, x_fixed, y_fixed, zmap,
                                 title_prefix, force_export=True, extra_note=note)
                except Exception as e:
                    print(f"[WARN] Fixed-pair plot failed for {setting}: {e}")

        print(f"[OK] Wrote fixed-pair PDF → {output_pdf_fixed}")

# -------- Main: run two passes (original + excluded) --------
def main():
    # Pass 1: original behavior (no exclusions) — writes the canonical 3 PDFs
    run_pipeline(
        output_pdf_main=os.path.expanduser(
            "~/gain_model/outputs/telemetry_pairwise_plots/telemetry_pairwise_scatter2d_withCPT1.pdf"
        ),
        output_pdf_all=os.path.expanduser(
            "~/gain_model/outputs/telemetry_pairwise_plots/telemetry_pairwise_scatter2d_withCPT1_CPT1_sig_in_ALL_settings.pdf"
        ),
        output_pdf_fixed=os.path.expanduser(
            "~/gain_model/outputs/telemetry_pairwise_plots/telemetry_pairwise_scatter2d_withCPT1_VMON1V2D_vs_SPE_1VAD8V.pdf"
        ),
        exclude_cols=set()
    )

    # Pass 2: exclude a problematic column everywhere — writes 3 short-named PDFs
    run_pipeline(
        output_pdf_main=os.path.expanduser(
            "~/gain_model/outputs/telemetry_pairwise_plots/telemetry_pairwise_noC.pdf"
        ),
        output_pdf_all=os.path.expanduser(
            "~/gain_model/outputs/telemetry_pairwise_plots/telemetry_pairwise_noC_ALL.pdf"
        ),
        output_pdf_fixed=os.path.expanduser(
            "~/gain_model/outputs/telemetry_pairwise_plots/telemetry_fixedpair_noC.pdf"
        ),
        exclude_cols={"SPE_1VAD8_C"}
    )

if __name__ == "__main__":
    main()
