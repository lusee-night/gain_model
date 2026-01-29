#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# How to run:
# python preamp_removal_withCLI.py   
# --cpt-list ~/uncrater/scripts/CPT_directories_withCPT1.txt   
# --out-base  ~/uncrater/data/plots/withpreamp1   
# --telemetry-dir ~/uncrater/data/plots/overall/telemetry_per_gainsetting_withCPT1   
# --telemetry-template '{gain}_withCPT1.csv'


import os
import re
import sys
import csv
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from sklearn.metrics import mean_squared_error
import matplotlib.transforms as mtransforms

# Ensure we can import from ~/uncrater/scripts for LUSEE preamp model
sys.path.append(os.path.expanduser('~/uncrater/scripts'))
import get_preamp_gain  # LUSEE_GAIN

# ---------------------------
# Configuration (constants that rarely change)
# ---------------------------

# All gains to process (channels 0..3, gains L/M/H)
GAINS_TO_RUN = [
    "L0","M0","H0",
    "L1","M1","H1",
    "L2","M2","H2",
    "L3","M3","H3",
]

# Base directory for FITS files (unchanged)
CALDB_DIR = os.path.expanduser('~/uncrater/scripts/caldb/caldb')

# Channel → preamp FITS filename
CHANNEL_TO_FITS = {
    0: "fmpre6_gain_temp_freq_dep.fits",  # ch0
    1: "fmpre3_gain_temp_freq_dep.fits",  # ch1
    2: "fmpre4_gain_temp_freq_dep.fits",  # ch2
    3: "fmpre1_gain_temp_freq_dep.fits",  # ch3
}

PC_COMPONENTS   = ["PC1", "PC2"]
THRESHOLD_RATIO = 2.0
GENERATE_SINGLE_PHASE_RMS = False
CLEAN_OLD_SINGLE_PHASE    = True

# ---------------------------
# Small helpers
# ---------------------------

def normalize_cpt(s: str) -> str:
    """
    Normalize any CPT-like label to 'CPT<digits>'.
    Examples:
      'CPT4b' -> 'CPT4'
      'CPT6+Science1' -> 'CPT6'
      '/.../CPT8+Science5/spt/session...' -> 'CPT8'
    """
    if s is None:
        return None
    s = str(s)
    m = re.search(r'CPT\s*(\d+)', s, flags=re.IGNORECASE)
    return f"CPT{m.group(1)}" if m else s.strip()

def _gain_channel(gain: str) -> int:
    """Map a gain label (e.g., 'L0','M2','H3') to channel index 0..3."""
    return int(gain[-1])

def build_gain_to_devfile():
    """Build full gain→FITS file map so L/M/H of a channel share the same device file."""
    return {
        g: os.path.join(CALDB_DIR, CHANNEL_TO_FITS[_gain_channel(g)])
        for g in GAINS_TO_RUN
    }

# ---------------------------
# Output directory management (depends on --out-base)
# ---------------------------

def make_output_dirs(out_base):
    """
    Create the same subdirectory layout under --out-base as before.
    Returns a dict with all resolved output paths.
    """
    PREAMP_DIR = os.path.join(out_base, 'preampremoval')
    JACK_DIR   = os.path.join(out_base, 'jackknife', 'withoutpreamps')
    PRED_DIR   = os.path.join(JACK_DIR, "predicted_gains")
    PLOT_DIR   = os.path.join(PRED_DIR, "plots")
    WITHPREAMP_DIR = os.path.join(out_base, 'jackknife', 'withpreamps')
    COMPARE_DIR    = os.path.join(out_base, 'jackknife', 'compare_preamps')

    for d in (PREAMP_DIR, JACK_DIR, PRED_DIR, PLOT_DIR, WITHPREAMP_DIR, COMPARE_DIR):
        os.makedirs(d, exist_ok=True)

    return dict(
        PREAMP_DIR=PREAMP_DIR,
        JACK_DIR=JACK_DIR,
        PRED_DIR=PRED_DIR,
        PLOT_DIR=PLOT_DIR,
        WITHPREAMP_DIR=WITHPREAMP_DIR,
        COMPARE_DIR=COMPARE_DIR,
    )

# ---------------------------
# Telemetry handling (depends on --telemetry-dir and --telemetry-template)
# ---------------------------

def telemetry_csv_path(gain, telem_dir, telem_template):
    """
    Resolve the telemetry CSV path for a given gain using the configured template.
    Example: telem_template='{gain}_withCPT1.csv'
    """
    fname = telem_template.format(gain=gain)
    return os.path.join(telem_dir, fname)

def get_telemetry_cols_for_gain(gain: str):
    """
    Channel-aware telemetry columns (with preamp removed phase):
      - THERM_FPGA
      - SPE_ADC{0 or 1}_T  (ch0/1 → ADC0_T,  ch2/3 → ADC1_T)
      - SPE_1VAD8_V
      - VMON_1V2D
      - SPE_1VAD8_C
    """
    ch = _gain_channel(gain)
    adc_col = "SPE_ADC0_T" if ch in (0, 1) else "SPE_ADC1_T"
    return ["THERM_FPGA", adc_col, "SPE_1VAD8_V", "VMON_1V2D", "SPE_1VAD8_C", "SPE_N5_C", "SPE_P5_C"]

def get_telemetry_cols_for_gain_withpreamps(gain: str):
    """
    Features (no preamp removal) for the PC-vs-features experiment:
      THERM_FPGA,
      SPE_ADC0_T (ch 0/1) or SPE_ADC1_T (ch 2/3),
      SPE_1VAD8_V, VMON_1V2D,
      PFPS_PA{ch}_T where ch is 0..3 from the gain label,
      SPE_1VAD8_C
    """
    ch = _gain_channel(gain)
    adc_col = "SPE_ADC0_T" if ch in (0, 1) else "SPE_ADC1_T"
    pfp_col = f"PFPS_PA{ch}_T"
    return ["THERM_FPGA", adc_col, "SPE_1VAD8_V", "VMON_1V2D", pfp_col, "SPE_1VAD8_C", "SPE_N5_C", "SPE_P5_C"]

# ---------------------------
# Phase 1 utilities
# ---------------------------

def load_gain_vectors(cpt_dirs, gain_setting):
    """
    Load a matrix of gain curves for one gain setting across all CPT directories.
    Returns: matrix (n_cpt, n_freq), freqs_MHz (n_freq,)
    """
    gain_matrix = []
    freqs = None
    for d in cpt_dirs:
        gain_path = os.path.join(os.path.expanduser(d), "gain.dat")
        with open(gain_path, 'r') as f:
            header = f.readline().strip().split()
            data = np.loadtxt(f)
        if gain_setting not in header:
            raise ValueError(f"{gain_setting} not found in {gain_path}")
        col_idx = header.index(gain_setting)
        if freqs is None:
            freqs = data[:, header.index("freq")]
        gain_column = data[:, col_idx]
        gain_matrix.append(gain_column)
    return np.array(gain_matrix), freqs

def db_to_linear(db_vals):
    """Convert dB (power gain) to linear scale."""
    return 10.0 ** (db_vals / 10.0)

def compute_preamp_corrections(freqs_mhz, temps, devfile):
    """
    For each CPT row temperature, compute preamp gain (dB) vs frequency,
    convert to linear, and compute correction factor relative to 20 °C.
    """
    params = get_preamp_gain.LUSEE_GAIN()
    params.fitsfile = os.path.expanduser(devfile)

    hdul = fits.open(params.fitsfile)
    params.freqs = np.empty(0)
    params.paras = np.empty(0)
    for iext in range(1, len(hdul)):
        para_num = hdul[iext].data.shape[0]
        for idata in range(len(hdul[iext].data[0])):
            d = hdul[iext].data.names[idata].split()
            params.freqs = np.append(params.freqs, float(d[0]))
            for ipara in range(para_num):
                params.paras = np.append(params.paras, hdul[iext].data[ipara][idata])
    params.paras = params.paras.reshape(int(len(params.paras) / para_num), para_num)

    freqs_hz = freqs_mhz * 1e6

    # Reference at 20 °C
    params.calc_gain_table_at_T(20.0)
    ref_db = params.get_gain_at_F(freqs_hz)
    ref_lin = db_to_linear(ref_db)

    all_corr = []
    for temp in temps:
        params.calc_gain_table_at_T(float(temp))
        temp_db = params.get_gain_at_F(freqs_hz)
        temp_lin = db_to_linear(temp_db)
        corr = temp_lin / ref_lin
        all_corr.append(corr)

    return np.array(all_corr)

def compute_pca_abs(matrix):
    """
    Compute PCA on absolute residuals (centered) and return:
    pcs, mean vector, eigenvectors (columns).
    """
    mean = np.mean(matrix, axis=0)
    abs_resid = matrix - mean
    cov_abs = abs_resid.T @ abs_resid / abs_resid.shape[0]
    evals_abs, evecs_abs = np.linalg.eigh(cov_abs)
    idx_abs = np.argsort(evals_abs)[::-1]
    evecs_sorted = evecs_abs[:, idx_abs]
    pcs_abs = abs_resid @ evecs_sorted
    return pcs_abs, mean, evecs_sorted

def write_pc_csv(path, cpt_names, data):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["CPT"] + [f"PC{i+1}" for i in range(data.shape[1])]
        writer.writerow(header)
        for name, row in zip(cpt_names, data):
            writer.writerow([name] + row.tolist())

# ---------------------------
# Phase 2 utilities
# ---------------------------

def build_feature_matrix(X, tele_cols, order=2):
    """
    Feature builder:
      - order >= 0: intercept
      - order >= 1: all linear terms in tele_cols (unchanged)
      - order == 2: ONLY quadratic temperature terms:
            (THERM_FPGA)^2, (SPE_ADCx_T)^2, (THERM_FPGA * SPE_ADCx_T)
        where SPE_ADCx_T is either SPE_ADC0_T or SPE_ADC1_T (whichever is present).
    """
    n_samples, n_features = X.shape
    Z = np.ones((n_samples, 1), dtype=float)
    feature_labels = ["1"]

    # Linear terms (keep all)
    if order >= 1 and n_features > 0:
        Z = np.hstack([Z, X.astype(float)])
        feature_labels += list(tele_cols)

    # Quadratic temperature terms ONLY
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

# ---------------------------
# Plot helpers
# ---------------------------

def _stacked_rms_bar_chart(gain, pc, with_rows, without_rows, outfile):
    order = ["Mean only", "Linear", "Quadratic"]
    color_map = {"Mean only": "gray", "Linear": "skyblue", "Quadratic": "seagreen"}

    w_map = {m: float(e) for m, e in zip(with_rows["model"], with_rows["rms_error"])}
    wo_map = {m: float(e) for m, e in zip(without_rows["model"], without_rows["rms_error"])}

    bars_with    = [m for m in order if m in w_map]
    values_with  = [w_map[m] for m in bars_with]
    bars_without = [m for m in order if m in wo_map]
    values_without = [wo_map[m] for m in bars_without]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7.0, 6.0), constrained_layout=False, sharex=True)

    def draw_panel(ax, bars, vals, title_tag):
        rects = ax.bar(bars, vals, color=[color_map.get(b, "gray") for b in bars])
        ax.set_ylabel("RMS Error")
        ax.set_title(title_tag)
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for r, v in zip(rects, vals):
            x = r.get_x() + r.get_width()/2.0
            label = f"{v:.2e}" if abs(v) < 1e-3 else f"{v:.4f}"
            ax.text(x, -0.14, label, transform=trans, ha="center", va="top", fontsize=9, clip_on=False)
        ax.grid(False)

    draw_panel(ax1, bars_with, values_with,  "(With Preamp Correction)")
    draw_panel(ax2, bars_without, values_without, "(Without Preamp Correction)")

    fig.suptitle(f"{gain}: RMS Comparison ({pc})", y=0.98, fontsize=12)
    fig.subplots_adjust(top=0.88, bottom=0.18, hspace=0.35)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# Phase 1
# ---------------------------

def phase1_preamp_correction_and_pca(cpt_dirs_file, gain_to_devfile, cpt_names, outdirs, telem_dir, telem_template):
    # Load CPT directories (aligned 1:1 with cpt_names)
    with open(os.path.expanduser(cpt_dirs_file)) as f:
        cpt_dirs = [line.strip() for line in f if line.strip()]
    assert len(cpt_dirs) == len(cpt_names), "Mismatch: CPT directories and names"

    PREAMP_DIR = outdirs["PREAMP_DIR"]
    print("[Phase 1] CPT_NAMES (normalized):", cpt_names)

    for gain, devfile in gain_to_devfile.items():
        print(f"[Phase 1] Processing {gain}...")

        tele_path = telemetry_csv_path(gain, telem_dir, telem_template)
        tele_df = pd.read_csv(tele_path)

        # Channel-specific preamp temp column
        ch = _gain_channel(gain)
        pfp_col = f"PFPS_PA{ch}_T"

        if "CPT" not in tele_df.columns:
            raise KeyError(f"{tele_path} must have a 'CPT' column")
        if pfp_col not in tele_df.columns:
            raise KeyError(f"{tele_path} must have column '{pfp_col}' for channel {ch}")

        # Normalize CPT labels from the CSV and align
        tele_df["CPT"] = tele_df["CPT"].map(normalize_cpt)
        tele_df = (tele_df.dropna(subset=["CPT"])
                           .drop_duplicates(subset=["CPT"], keep="first")
                           .set_index("CPT")
                           .reindex(cpt_names))

        # Check for missing channel-specific temps AFTER normalization
        if tele_df[pfp_col].isna().any():
            missing = tele_df[pfp_col].isna()
            missing_cpts = [c for c, m in zip(cpt_names, missing) if m]
            raise ValueError(
                f"Missing {pfp_col} for CPTs {missing_cpts} in {tele_path}"
            )

        temps = tele_df[pfp_col].astype(float).values

        # Load raw measured gain matrices and frequencies
        matrix_full, freqs = load_gain_vectors(cpt_dirs, gain)

        # Apply preamp corrections
        corrections = compute_preamp_corrections(freqs, temps, devfile)
        corrected_matrix = matrix_full / corrections

        # Export corrected matrix (CSV with CPT labels)
        corr_outfile = os.path.join(PREAMP_DIR, f"{gain}_matrix_corrected.csv")
        with open(corr_outfile, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["CPT"] + [f"{freq:.6g}" for freq in freqs]
            writer.writerow(header)
            for name, row in zip(cpt_names, corrected_matrix):
                writer.writerow([name] + row.tolist())

        # Perform PCA (absolute residuals only) on corrected matrix
        pcs_abs, mean_vec, eigvecs = compute_pca_abs(corrected_matrix)
        write_pc_csv(os.path.join(PREAMP_DIR, f"pca_abs_{gain}_corrected.csv"), cpt_names, pcs_abs)

        # Save PCA ingredients for reconstruction
        np.save(os.path.join(PREAMP_DIR, f"{gain}_mean.npy"), mean_vec)
        np.save(os.path.join(PREAMP_DIR, f"{gain}_eigvecs.npy"), eigvecs)
        np.save(os.path.join(PREAMP_DIR, f"{gain}_freqs.npy"), freqs)

        print(f"[Phase 1] Finished {gain}: wrote corrected matrix, PCA, and PCA bases")

# ---------------------------
# Phase 2 (jackknife on corrected PCA; rebuild predicted spectra & nRMS)
# ---------------------------

def phase2_jackknife_on_corrected_pca(cpt_names, outdirs, telem_dir, telem_template):
    PREAMP_DIR      = outdirs["PREAMP_DIR"]
    JACK_DIR        = outdirs["JACK_DIR"]
    PRED_DIR        = outdirs["PRED_DIR"]
    PLOT_DIR        = outdirs["PLOT_DIR"]

    all_alpha_records   = []
    filtered_out_records = []
    rms_error_records   = []

    # predicted PCs storage
    predictions_per_pc = {g: {pc: {} for pc in PC_COMPONENTS} for g in GAINS_TO_RUN}
    reliable_terms_lookup = {}

    # --- Main regression loop ---
    for pc in PC_COMPONENTS:
        for gain in GAINS_TO_RUN:
            tele_cols = get_telemetry_cols_for_gain(gain)
            print(f"[Phase 2] Processing {gain} ({pc}) with telemetry {tele_cols}...")

            pca_path = os.path.join(PREAMP_DIR, f"pca_abs_{gain}_corrected.csv")
            if not os.path.exists(pca_path):
                print(f"[Phase 2] Missing file: {pca_path}")
                continue

            tele_path = telemetry_csv_path(gain, telem_dir, telem_template)
            pca_df  = pd.read_csv(pca_path)[["CPT", pc]]
            tele_df = pd.read_csv(tele_path)[["CPT"] + tele_cols]

            pca_df["CPT"]  = pca_df["CPT"].map(normalize_cpt)
            tele_df["CPT"] = tele_df["CPT"].map(normalize_cpt)

            df = (pd.merge(pca_df, tele_df, on="CPT", how="inner")
                    .set_index("CPT")
                    .reindex([c for c in cpt_names if c in pca_df["CPT"].values])
                    .dropna()
                    .reset_index())

            y = df[pc].values
            X = df[tele_cols].values
            n = len(y)
            if n == 0:
                continue

            for order in [1, 2]:
                model_name = "linear" if order == 1 else "quadratic"
                Z, labels = build_feature_matrix(X, tele_cols, order)
                alpha_orig = np.linalg.lstsq(Z, y, rcond=None)[0]

                # Jackknife LOO
                jackknife_alphas = []
                for i in range(n):
                    mask = np.ones(n, dtype=bool); mask[i] = False
                    Z_i, y_i = Z[mask], y[mask]
                    alpha_i = np.linalg.lstsq(Z_i, y_i, rcond=None)[0]
                    jackknife_alphas.append(alpha_i)
                jackknife_alphas = np.vstack(jackknife_alphas)

                var_jack = np.var(jackknife_alphas, axis=0, ddof=1)
                stderr = np.sqrt(var_jack / n)

                # Store coeffs & reliability
                for (label, a_val, se_val) in zip(labels, alpha_orig, stderr):
                    ratio = (abs(a_val / se_val) if se_val != 0 else np.inf)
                    record = {
                        "gain_setting": gain,
                        "component": pc,
                        "model": model_name,
                        "term": label,
                        "alpha": a_val,
                        "stderr": se_val,
                        "ratio": ratio,
                    }
                    all_alpha_records.append(record)
                    if ratio <= THRESHOLD_RATIO:
                        filtered_out_records.append(record)
                    else:
                        key = (gain, pc, model_name)
                        reliable_terms_lookup.setdefault(key, set()).add(label)

            # Fit final using reliable terms; save predicted PCs
            for order in [1, 2]:
                model_name = "linear" if order == 1 else "quadratic"
                Z, labels = build_feature_matrix(X, tele_cols, order)
                keep = reliable_terms_lookup.get((gain, pc, model_name), set())
                keep_idx = [i for i, lbl in enumerate(labels) if lbl in keep]
                if not keep_idx:
                    continue
                Z_filt = Z[:, keep_idx]
                alpha_filt = np.linalg.lstsq(Z_filt, y, rcond=None)[0]
                y_pred = Z_filt @ alpha_filt
                predictions_per_pc[gain][pc][model_name] = y_pred

    # Export unreliable alphas log
    if filtered_out_records:
        with open(os.path.join(JACK_DIR, "unreliable_alphas.txt"), "w") as f:
            for r in filtered_out_records:
                f.write(f"{r['component']}, {r['gain_setting']}, {r['model']}, {r['term']}, "
                        f"alpha = {r['alpha']:.4f}, stderr = {r['stderr']:.4f}, "
                        f"alpha/stderr = {r['ratio']:.2f}\n")

    # Plots & RMS using reliable terms
    for pc in PC_COMPONENTS:
        for gain in GAINS_TO_RUN:
            tele_cols = get_telemetry_cols_for_gain(gain)
            pca_path = os.path.join(PREAMP_DIR, f"pca_abs_{gain}_corrected.csv")
            if not os.path.exists(pca_path):
                continue
            tele_path = telemetry_csv_path(gain, telem_dir, telem_template)

            pca_df  = pd.read_csv(pca_path)[["CPT", pc]]
            tele_df = pd.read_csv(tele_path)[["CPT"] + tele_cols]

            pca_df["CPT"]  = pca_df["CPT"].map(normalize_cpt)
            tele_df["CPT"] = tele_df["CPT"].map(normalize_cpt)

            df = (pd.merge(pca_df, tele_df, on="CPT", how="inner")
                    .set_index("CPT")
                    .reindex([c for c in cpt_names if c in pca_df["CPT"].values])
                    .dropna()
                    .reset_index())

            y_true = df[pc].values
            if y_true.size == 0:
                continue
            X = df[tele_cols].values

            predictions = {}
            rms_vals = {}

            # Mean-only
            y_mean = np.full_like(y_true, np.mean(y_true))
            predictions["Mean only"] = y_mean
            rms_vals["Mean only"] = np.sqrt(mean_squared_error(y_true, y_mean))

            # Linear model (reliable terms only)
            Z1, labels1 = build_feature_matrix(X, tele_cols, order=1)
            keep1 = reliable_terms_lookup.get((gain, pc, "linear"), set())
            keep_idx1 = [i for i, lbl in enumerate(labels1) if lbl in keep1]
            if keep_idx1:
                Z1_filt = Z1[:, keep_idx1]
                alpha1 = np.linalg.lstsq(Z1_filt, y_true, rcond=None)[0]
                y_pred_1 = Z1_filt @ alpha1
                predictions["Linear"] = y_pred_1
                rms_vals["Linear"] = np.sqrt(mean_squared_error(y_true, y_pred_1))

            # Quadratic model (reliable temp-only quads)
            Z2, labels2 = build_feature_matrix(X, tele_cols, order=2)
            keep2 = reliable_terms_lookup.get((gain, pc, "quadratic"), set())
            keep_idx2 = [i for i, lbl in enumerate(labels2) if lbl in keep2]
            if keep_idx2:
                Z2_filt = Z2[:, keep_idx2]
                alpha2 = np.linalg.lstsq(Z2_filt, y_true, rcond=None)[0]
                y_pred_2 = Z2_filt @ alpha2
                predictions["Quadratic"] = y_pred_2
                rms_vals["Quadratic"] = np.sqrt(mean_squared_error(y_true, y_pred_2))

            # Store RMS records
            for model_name, rms_val in rms_vals.items():
                rms_error_records.append({
                    "component": pc,
                    "gain_setting": gain,
                    "model": model_name,
                    "rms_error": round(float(rms_val), 6),
                })

            # Scatterplots
            for label, y_pred in predictions.items():
                plt.figure(figsize=(6, 6))
                plt.scatter(y_true, y_pred, alpha=0.8)
                minv, maxv = float(np.min(y_true)), float(np.max(y_true))
                plt.plot([minv, maxv], [minv, maxv], 'k--')
                plt.xlabel(f"Actual {pc}")
                plt.ylabel(f"Predicted {pc}")
                plt.title(f"{gain}: {label} Model ({pc})\n(With Preamp Correction)\nRMS = {rms_vals[label]:.4f}")
                for i, cpt in enumerate(df["CPT"]):
                    plt.annotate(cpt, (y_true[i], y_pred[i]),
                                 textcoords="offset points", xytext=(4, 2),
                                 ha='left', fontsize=8, alpha=0.7)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(
                    JACK_DIR,
                    f"{gain}_scatter_{label.replace(' ', '_').lower()}_{pc.lower()}.png"
                ))
                plt.close()

    # Final exports
    pd.DataFrame(all_alpha_records).to_csv(
        os.path.join(JACK_DIR, "jackknife_alphas_with_errors.csv"), index=False
    )
    pd.DataFrame(rms_error_records).to_csv(
        os.path.join(JACK_DIR, "rms_errors.csv"), index=False
    )
    reliable_alpha_records = [
        r for r in all_alpha_records
        if (abs(r["alpha"] / r["stderr"]) > THRESHOLD_RATIO)
    ]
    pd.DataFrame(reliable_alpha_records).to_csv(
        os.path.join(JACK_DIR, "reliable_alphas.csv"), index=False
    )

    print("[Phase 2] Jackknife done. RMS errors, alpha values, reliable terms exported.")

    # ---- Reconstruct predicted gains from predicted PCs (PC1 & PC2 only) ----

    def reconstruct_gains(pred_pcs, mean_vec, eigvecs, k):
        V_k = eigvecs[:, :k]
        return mean_vec + pred_pcs @ V_k.T

    nrms_records = []

    for gain in GAINS_TO_RUN:
        mean_vec = np.load(os.path.join(PREAMP_DIR, f"{gain}_mean.npy"))
        eigvecs  = np.load(os.path.join(PREAMP_DIR, f"{gain}_eigvecs.npy"))
        freqs    = np.load(os.path.join(PREAMP_DIR, f"{gain}_freqs.npy"))

        measured_df = pd.read_csv(os.path.join(PREAMP_DIR, f"{gain}_matrix_corrected.csv"))
        measured_df["CPT"] = measured_df["CPT"].map(normalize_cpt)
        measured_df = measured_df.set_index("CPT").reindex(cpt_names).reset_index()

        measured = measured_df.iloc[:, 1:].to_numpy()

        pcs_path = os.path.join(PREAMP_DIR, f"pca_abs_{gain}_corrected.csv")
        pcs_df = pd.read_csv(pcs_path)
        pcs_df["CPT"] = pcs_df["CPT"].map(normalize_cpt)
        pcs_df = pcs_df.set_index("CPT").reindex(cpt_names).reset_index()

        pcs_12 = pcs_df[["PC1", "PC2"]].to_numpy()
        actual_preds_12 = reconstruct_gains(pcs_12, mean_vec, eigvecs, k=2)

        np.savetxt(os.path.join(PRED_DIR, f"{gain}_actual_PC12_reconstruction.csv"),
                   actual_preds_12, delimiter=",",
                   header=",".join([f"{f:.6g}" for f in freqs]), comments='')

        # Actual-PC12 NRMS
        errors_actual = measured - actual_preds_12
        per_cpt_rms_actual = np.sqrt(np.mean(errors_actual**2, axis=1))
        per_cpt_mean = np.mean(measured, axis=1)
        per_cpt_nrms_actual = per_cpt_rms_actual / per_cpt_mean

        global_rms_actual = np.sqrt(np.mean(errors_actual**2))
        global_mean = np.mean(measured)
        global_nrms_actual = global_rms_actual / global_mean

        for cpt, val in zip(cpt_names, per_cpt_nrms_actual):
            nrms_records.append({
                "gain": gain, "model": "Actual", "case": "PC12", "CPT": cpt, "NRMS": float(val)
            })
        nrms_records.append({
            "gain": gain, "model": "Actual", "case": "PC12", "CPT": "GLOBAL", "NRMS": float(global_nrms_actual)
        })

        for model in ["Linear", "Quadratic"]:
            pc1 = predictions_per_pc[gain]["PC1"].get(model.lower(), None)
            pc2 = predictions_per_pc[gain]["PC2"].get(model.lower(), None)
            if pc1 is None and pc2 is None:
                continue

            pcs_12_pred = np.column_stack([
                pc1 if pc1 is not None else np.zeros(len(cpt_names)),
                pc2 if pc2 is not None else np.zeros(len(cpt_names)),
            ])
            preds_12 = reconstruct_gains(pcs_12_pred, mean_vec, eigvecs, k=2)
            np.savetxt(os.path.join(PRED_DIR, f"{gain}_{model.lower()}_PC12_predicted.csv"),
                       preds_12, delimiter=",",
                       header=",".join([f"{f:.6g}" for f in freqs]), comments='')

            errors = measured - preds_12
            per_cpt_rms  = np.sqrt(np.mean(errors**2, axis=1))
            per_cpt_mean = np.mean(measured, axis=1)
            per_cpt_nrms = per_cpt_rms / per_cpt_mean

            global_rms  = np.sqrt(np.mean(errors**2))
            global_mean = np.mean(measured)
            global_nrms = global_rms / global_mean

            for cpt, val in zip(cpt_names, per_cpt_nrms):
                nrms_records.append({
                    "gain": gain, "model": model, "case": "PC12", "CPT": cpt, "NRMS": float(val)
                })
            nrms_records.append({
                "gain": gain, "model": model, "case": "PC12", "CPT": "GLOBAL", "NRMS": float(global_nrms)
            })

            # Per-frequency RMS for error bars
            sigma_per_freq = np.sqrt(np.mean(errors**2, axis=0))

            # ---- Per-CPT measured / predicted / actual plots----
            CPT_NAMES = cpt_names

            for i, cpt in enumerate(CPT_NAMES):
                plt.figure(figsize=(8, 5))
                # measured[i, :]  -> measured gain curve for this CPT
                plt.plot(freqs, measured[i, :], label="Measured")

                # preds_12[i, :]  -> predicted gain curve from predicted PC1/PC2 (current model)
                # sigma_per_freq  -> per-frequency RMS used as error bars on predicted
                plt.errorbar(
                    freqs,
                    preds_12[i, :],
                    yerr=sigma_per_freq,
                    fmt="--",
                    label="Predicted",
                    alpha=0.85,
                    capsize=3,
                )

                # actual_preds_12[i, :] -> reconstruction from ACTUAL PC1/PC2 (ground truth PC12)
                plt.plot(freqs, actual_preds_12[i, :], label="Actual PC12", linestyle=":")

                plt.xlabel("Frequency (MHz)")
                plt.ylabel("Gain (linear)")
                plt.title(
                    f"{gain} {model} PC12 - {cpt} | "
                    f"nRMS_pred={per_cpt_nrms[i]:.3f} | "
                    f"nRMS_actual={per_cpt_nrms_actual[i]:.3f}"
                )
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(PLOT_DIR, f"{gain}_{model.lower()}_PC12_{cpt}.png"))
                plt.close()


    pd.DataFrame(nrms_records).to_csv(os.path.join(PRED_DIR, "nrms_errors.csv"), index=False)

# ---------------------------
# With-preamps PC plots (no preamp removal)
# ---------------------------

def phase_withpreamps_pc_plots(cpt_names, outdirs, telem_dir, telem_template):
    WITHPREAMP_DIR = outdirs["WITHPREAMP_DIR"]

    # Load CPT directories aligned with names
    # We need them to build PCs on raw matrices
    # (We rebuild here to keep this pass self-contained.)
    # The caller will re-open the cpt list file to preserve alignment
    # and pass both paths & names here instead.
    raise_if = []

def _withpreamps_inner(cpt_dirs_file, cpt_names, outdirs, telem_dir, telem_template):
    WITHPREAMP_DIR = outdirs["WITHPREAMP_DIR"]

    with open(os.path.expanduser(cpt_dirs_file)) as f:
        cpt_dirs = [line.strip() for line in f if line.strip()]
    assert len(cpt_dirs) == len(cpt_names), "Mismatch in withpreamps pass"

    rms_error_records = []

    for gain in GAINS_TO_RUN:
        raw_matrix, _freqs = load_gain_vectors(cpt_dirs, gain)
        pcs_abs, _mean_vec, _eigvecs = compute_pca_abs(raw_matrix)

        pcs_df = pd.DataFrame({
            "CPT": cpt_names,
            "PC1": pcs_abs[:, 0],
            "PC2": pcs_abs[:, 1] if pcs_abs.shape[1] > 1 else np.zeros(len(cpt_names))
        })

        tele_cols = get_telemetry_cols_for_gain_withpreamps(gain)
        tele_path = telemetry_csv_path(gain, telem_dir, telem_template)
        tele_df = pd.read_csv(tele_path)

        missing = [c for c in ["CPT"] + tele_cols if c not in tele_df.columns]
        if missing:
            print(f"[WithPreamps] Skipping {gain}: missing columns {missing} in {tele_path}")
            continue

        pcs_df["CPT"]  = pcs_df["CPT"].map(normalize_cpt)
        tele_df["CPT"] = tele_df["CPT"].map(normalize_cpt)

        merged = (pd.merge(pcs_df, tele_df[["CPT"] + tele_cols], on="CPT", how="inner")
                    .set_index("CPT")
                    .reindex([c for c in cpt_names if c in pcs_df["CPT"].values])
                    .dropna()
                    .reset_index())

        for pc in ["PC1", "PC2"]:
            if pc not in merged.columns:
                continue

            y = merged[pc].values
            if y.size == 0:
                continue
            X = merged[tele_cols].values
            n = len(y)

            predictions = {"Mean only": np.full_like(y, np.mean(y))}
            rms_vals = {"Mean only": float(np.sqrt(mean_squared_error(y, predictions["Mean only"])))}

            all_alpha_records = []
            filtered_out_records = []

            for order in [1, 2]:
                model_name = "linear" if order == 1 else "quadratic"
                Z, labels = build_feature_matrix(X, tele_cols, order)
                alpha_full = np.linalg.lstsq(Z, y, rcond=None)[0]

                # Jackknife LOO
                jack = []
                for i in range(n):
                    mask = np.ones(n, dtype=bool); mask[i] = False
                    Zi, yi = Z[mask], y[mask]
                    ai = np.linalg.lstsq(Zi, yi, rcond=None)[0]
                    jack.append(ai)
                jack = np.vstack(jack)
                var_jack = np.var(jack, axis=0, ddof=1)
                stderr = np.sqrt(var_jack / n)

                keep = []
                for lbl, a, se in zip(labels, alpha_full, stderr):
                    ratio = (abs(a / se) if se != 0 else np.inf)
                    if ratio > THRESHOLD_RATIO:
                        keep.append(lbl)
                    else:
                        filtered_out_records.append(lbl)

                keep_idx = [i for i, lbl in enumerate(labels) if lbl in keep]
                if keep_idx:
                    Zf = Z[:, keep_idx]
                    alpha_f = np.linalg.lstsq(Zf, y, rcond=None)[0]
                    y_pred = Zf @ alpha_f
                    predictions[model_name.capitalize()] = y_pred
                    rms_vals[model_name.capitalize()] = float(np.sqrt(mean_squared_error(y, y_pred)))

            # Predicted vs Actual scatter plots
            for label, y_pred in predictions.items():
                plt.figure(figsize=(6, 6))
                plt.scatter(y, y_pred, alpha=0.8)
                lo, hi = float(np.min(y)), float(np.max(y))
                plt.plot([lo, hi], [lo, hi], 'k--')
                plt.xlabel(f"Actual {pc}")
                plt.ylabel(f"Predicted {pc}")
                plt.title(
                    f"{gain}: {label} Model ({pc})\n"
                    f"(Without Preamp Correction)\n"
                    f"RMS = {rms_vals[label]:.4f}"
                )
                for i, cpt in enumerate(merged["CPT"]):
                    plt.annotate(cpt, (y[i], y_pred[i]),
                                 textcoords="offset points", xytext=(4, 2),
                                 ha='left', fontsize=8, alpha=0.7)
                plt.grid(True); plt.tight_layout()
                out_png = os.path.join(
                    WITHPREAMP_DIR,
                    f"{gain}_scatter_{label.replace(' ', '_').lower()}_{pc.lower()}.png"
                )
                plt.savefig(out_png); plt.close()

            for model_name, rms_val in rms_vals.items():
                rms_error_records.append({
                    "component": pc,
                    "gain_setting": gain,
                    "model": model_name,
                    "rms_error": round(float(rms_val), 6),
                    "preamp_removed": False
                })

    pd.DataFrame(rms_error_records).to_csv(
        os.path.join(WITHPREAMP_DIR, "rms_errors.csv"), index=False
    )
    print("[WithPreamps] PC-level plots and RMS charts exported.")

# ---------------------------
# Compare stacked RMS charts
# ---------------------------

def build_stacked_rms_comparisons(outdirs):
    JACK_DIR      = outdirs["JACK_DIR"]
    WITHPREAMP_DIR = outdirs["WITHPREAMP_DIR"]
    COMPARE_DIR    = outdirs["COMPARE_DIR"]

    with_csv  = os.path.join(JACK_DIR, "rms_errors.csv")
    wo_csv    = os.path.join(WITHPREAMP_DIR, "rms_errors.csv")

    if not (os.path.exists(with_csv) and os.path.exists(wo_csv)):
        print("[Compare] Missing rms_errors.csv for one or both phases; skipping stacked charts.")
        return

    df_with = pd.read_csv(with_csv)
    df_wo   = pd.read_csv(wo_csv)

    needed = {"component", "gain_setting", "model", "rms_error"}
    if not needed.issubset(df_with.columns) or not needed.issubset(df_wo.columns):
        print("[Compare] CSVs missing required columns; skipping.")
        return

    for pc in PC_COMPONENTS:
        for gain in GAINS_TO_RUN:
            w_rows  = df_with[(df_with["component"] == pc) & (df_with["gain_setting"] == gain)][["model","rms_error"]]
            wo_rows = df_wo[(df_wo["component"]   == pc) & (df_wo["gain_setting"] == gain)][["model","rms_error"]]
            if w_rows.empty and wo_rows.empty:
                continue
            outfile = os.path.join(COMPARE_DIR, f"{gain}_rms_comparison_{pc.lower()}_stacked.png")
            _stacked_rms_bar_chart(gain, pc, w_rows, wo_rows, outfile)
            print(f"[Compare] Wrote {outfile}")

# ---------------------------
# Misc
# ---------------------------

def _compute_global_nrms(measured, pred):
    errors = measured - pred
    global_rms  = np.sqrt(np.mean(errors**2))
    global_mean = np.mean(measured)
    return float(global_rms / global_mean)

def _predict_pc_linear(tele_df, pc_df, predictors, cpt_names):
    df = (pd.merge(pc_df, tele_df[["CPT"] + predictors], on="CPT", how="inner")
            .set_index("CPT")
            .reindex([c for c in cpt_names if c in pc_df["CPT"].values])
            .dropna()
            .reset_index())

    y = df.iloc[:, 1].values  # after ["CPT", <PC>]
    Z = np.column_stack([np.ones(len(y)), df[predictors].values])
    alpha = np.linalg.lstsq(Z, y, rcond=None)[0]
    y_pred = Z @ alpha
    return df["CPT"].tolist(), y_pred

def _cleanup_single_phase_rms_pngs(outdirs):
    if not CLEAN_OLD_SINGLE_PHASE:
        return
    for d in (outdirs["JACK_DIR"], outdirs["WITHPREAMP_DIR"]):
        for name in os.listdir(d):
            if name.endswith(".png") and "_rms_comparison_" in name:
                try:
                    os.remove(os.path.join(d, name))
                except Exception as e:
                    print(f"[Cleanup] Could not remove {name} from {d}: {e}")

# ---------------------------
# Orchestration
# ---------------------------

def run_pipeline(cpt_dirs_file, out_base, telem_dir, telem_template):
    # Build output dirs under out_base
    outdirs = make_output_dirs(out_base)

    # Build gain→device file map
    gain_to_devfile = build_gain_to_devfile()

    # Load CPT directory list and derive normalized CPT names
    with open(os.path.expanduser(cpt_dirs_file)) as f:
        cpt_dirs = [line.strip() for line in f if line.strip()]
    cpt_names = [normalize_cpt(p) for p in cpt_dirs]
    print("[Phase 1] CPT_NAMES (normalized):", cpt_names)

    # Phase 1
    phase1_preamp_correction_and_pca(
        cpt_dirs_file=cpt_dirs_file,
        gain_to_devfile=gain_to_devfile,
        cpt_names=cpt_names,
        outdirs=outdirs,
        telem_dir=telem_dir,
        telem_template=telem_template,
    )

    # Phase 2
    phase2_jackknife_on_corrected_pca(
        cpt_names=cpt_names,
        outdirs=outdirs,
        telem_dir=telem_dir,
        telem_template=telem_template,
    )

    # With-preamps PC plots (raw)
    _withpreamps_inner(
        cpt_dirs_file=cpt_dirs_file,
        cpt_names=cpt_names,
        outdirs=outdirs,
        telem_dir=telem_dir,
        telem_template=telem_template,
    )

    _cleanup_single_phase_rms_pngs(outdirs)
    build_stacked_rms_comparisons(outdirs)

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="LuSEE preamp removal + PCA + jackknife pipeline (with CLI).")
    ap.add_argument("--cpt-list", required=True,
                    help="Path to CPT directories list (one path per line).")
    ap.add_argument("--out-base", required=True,
                    help="Base output directory. Subdirs will mirror the previous layout.")
    ap.add_argument("--telemetry-dir", required=True,
                    help="Directory containing telemetry per-gain CSV files.")
    ap.add_argument("--telemetry-template", required=True,
                    help="Filename template for telemetry CSV, use {gain} placeholder (e.g., '{gain}_withCPT1.csv').")
    return ap.parse_args()

def main():
    args = parse_args()
    run_pipeline(
        cpt_dirs_file = args.cpt_list,
        out_base      = os.path.expanduser(args.out_base),
        telem_dir     = os.path.expanduser(args.telemetry_dir),
        telem_template= args.telemetry_template,
    )

if __name__ == "__main__":
    main()
