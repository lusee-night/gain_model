#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gain PCA + telemetry regression pipeline.

What this script does
- Builds per-gain gain matrices across CPT runs from gain.dat.
- Optionally removes preamp temperature dependence using CALDB FITS tables
  (correction relative to 20°C) to produce "corrected" spectra.
- Runs PCA on (matrix - mean) residuals (absolute residual PCA).
- Fits PC1/PC2 vs telemetry using linear + limited quadratic temperature terms,
  using jackknife leave-one-out to filter unreliable regression terms.
- Reconstructs predicted gain spectra from predicted PCs (PC1+PC2) and exports:
    • predicted spectra CSVs
    • per-CPT spectra plots with per-frequency RMS error bars
    • RMS / nRMS metrics tables
- Runs a comparison pipeline "raw_with_preamps" (no correction) for reference.
- Runs a CPT1-only diagnostic that applies the final refit coefficients to CPT1,
  plots measured/predicted ratios, and emits CPT1 spectra comparison plots.

Inputs
- CPT directory list: ~/gain_model/scripts/CPT_directories.txt
- Telemetry per-gain CSVs:
    ~/gain_model/outputs/telemetry_per_gainsetting/{GAIN}.csv
    ~/gain_model/outputs/telemetry_per_gainsetting_withCPT1/{GAIN}_withCPT1.csv
- Preamp CALDB FITS tables: ~/gain_model/scripts/caldb/caldb/<fitsfile>

Outputs (organized under)
  ~/gain_model/outputs/gain_pca_model/
"""

import os
import numpy as np
import csv
import sys
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.transforms as mtransforms

# Make sure we can import from ~/gain_model/scripts
sys.path.append(os.path.expanduser('~/gain_model/scripts'))
import get_preamp_gain  # for LUSEE_GAIN class


# ---------------------------
# Roots / Inputs
# ---------------------------
ROOT = os.path.expanduser('~/gain_model')

CPT_DIR_LIST = os.path.join(ROOT, "scripts", "CPT_directories.txt")

TELEM_PER_GAIN_DIR = os.path.join(ROOT, "outputs", "telemetry_per_gainsetting")
TELEM_WITH_CPT1_DIR = os.path.join(ROOT, "outputs", "telemetry_per_gainsetting_withCPT1")

CALDB_DIR = os.path.join(ROOT, "scripts", "caldb", "caldb")


# ---------------------------
# Outputs
# ---------------------------
OUT_ROOT = os.path.join(ROOT, "outputs", "gain_pca_model")

OUT_CORR = os.path.join(OUT_ROOT, "corrected")
OUT_CORR_PHASE1 = os.path.join(OUT_CORR, "phase1")
OUT_CORR_PHASE2 = os.path.join(OUT_CORR, "phase2")

PREAMP_DIR = os.path.join(OUT_CORR_PHASE1, "pca")                 # PCA ingredients + PCA CSVs
MATRIX_DIR = os.path.join(OUT_CORR_PHASE1, "matrices_corrected")  # corrected gain matrices

ALPHAS_DIR = os.path.join(OUT_CORR_PHASE2, "alphas")
PC_SCATTER_DIR = os.path.join(OUT_CORR_PHASE2, "pc_scatter")
SPECTRA_PRED_DIR = os.path.join(OUT_CORR_PHASE2, "spectra_predictions")
SPECTRA_PLOT_DIR = os.path.join(OUT_CORR_PHASE2, "spectra_plots")
METRICS_DIR = os.path.join(OUT_CORR_PHASE2, "metrics")

OUT_RAW = os.path.join(OUT_ROOT, "raw_with_preamps")
RAW_PC_SCATTER_DIR = os.path.join(OUT_RAW, "pc_scatter")
RAW_METRICS_DIR = os.path.join(OUT_RAW, "metrics")

OUT_COMPARE = os.path.join(OUT_ROOT, "compare_preamps")
COMPARE_STACKED_DIR = os.path.join(OUT_COMPARE, "stacked_rms")

OUT_CPT1 = os.path.join(OUT_ROOT, "cpt1_refit")
CPT1_RATIO_DIR = os.path.join(OUT_CPT1, "ratios")
CPT1_SPECTRA_DIR = os.path.join(OUT_CPT1, "spectra")

for d in [
    PREAMP_DIR, MATRIX_DIR,
    ALPHAS_DIR, PC_SCATTER_DIR, SPECTRA_PRED_DIR, SPECTRA_PLOT_DIR, METRICS_DIR,
    RAW_PC_SCATTER_DIR, RAW_METRICS_DIR,
    COMPARE_STACKED_DIR,
    CPT1_RATIO_DIR, CPT1_SPECTRA_DIR,
]:
    os.makedirs(d, exist_ok=True)


# ---------------------------
# Config
# ---------------------------

# CPT names (aligned with CPT_directories.txt)
CPT_NAMES = [
    "CPT2", "CPT3", "CPT4", "CPT5", "CPT6", "CPT7", "CPT8",
    "CPT9", "CPT10", "CPT11", "CPT12", "CPT13", "CPT15", "CPT16"
]

GAINS_TO_RUN = [
    "L0","M0","H0",
    "L1","M1","H1",
    "L2","M2","H2",
    "L3","M3","H3",
]

CHANNEL_TO_FITS = {
    0: "fmpre6_gain_temp_freq_dep.fits",  # ch0
    1: "fmpre3_gain_temp_freq_dep.fits",  # ch1
    2: "fmpre4_gain_temp_freq_dep.fits",  # ch2
    3: "fmpre1_gain_temp_freq_dep.fits",  # ch3
}

def _gain_channel(gain: str) -> int:
    return int(gain[-1])

GAIN_TO_DEVFILE = {
    g: os.path.join(CALDB_DIR, CHANNEL_TO_FITS[_gain_channel(g)])
    for g in GAINS_TO_RUN
}

PC_COMPONENTS = ["PC1", "PC2"]
THRESHOLD_RATIO = 2.0

# Toggle: write single-phase RMS bar PNGs?
GENERATE_SINGLE_PHASE_RMS = False
CLEAN_OLD_SINGLE_PHASE = True


# ---------------------------
# Phase 1 helpers: load gains, apply preamp correction, PCA
# ---------------------------

def load_gain_vectors(cpt_dirs, gain_setting):
    gain_matrix = []
    for d in cpt_dirs:
        gain_path = os.path.join(os.path.expanduser(d), "gain.dat")
        with open(gain_path, 'r') as f:
            header = f.readline().strip().split()
            data = np.loadtxt(f)
        if gain_setting not in header:
            raise ValueError(f"{gain_setting} not found in {gain_path}")
        col_idx = header.index(gain_setting)
        gain_matrix.append(data[:, col_idx])
    return np.array(gain_matrix), data[:, header.index("freq")]  # MHz

def db_to_linear(db_vals):
    return 10.0 ** (db_vals / 10.0)

def compute_preamp_corrections(freqs_mhz, temps, devfile):
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

    params.calc_gain_table_at_T(20.0)
    ref_db = params.get_gain_at_F(freqs_hz)
    ref_lin = db_to_linear(ref_db)

    all_corr = []
    for temp in temps:
        params.calc_gain_table_at_T(float(temp))
        temp_db = params.get_gain_at_F(freqs_hz)
        temp_lin = db_to_linear(temp_db)
        all_corr.append(temp_lin / ref_lin)

    return np.array(all_corr)

def compute_pca_abs(matrix):
    mean = np.mean(matrix, axis=0)
    resid = matrix - mean
    cov_abs = resid.T @ resid / resid.shape[0]
    evals, evecs = np.linalg.eigh(cov_abs)
    idx = np.argsort(evals)[::-1]
    evecs_sorted = evecs[:, idx]
    pcs = resid @ evecs_sorted
    return pcs, mean, evecs_sorted

def write_pc_csv(path, cpt_names, data):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["CPT"] + [f"PC{i+1}" for i in range(data.shape[1])])
        for name, row in zip(cpt_names, data):
            w.writerow([name] + row.tolist())


# ---------------------------
# Phase 1: preamp correction + PCA
# ---------------------------

def phase1_preamp_correction_and_pca():
    with open(CPT_DIR_LIST) as f:
        cpt_dirs = [line.strip() for line in f if line.strip()]
    assert len(cpt_dirs) == len(CPT_NAMES), "Mismatch: CPT directories and names"

    for gain, devfile in GAIN_TO_DEVFILE.items():
        print(f"[Phase 1] Processing {gain}...")

        # Use the channel-correct PFPS preamp temperature:
        #   ch0 -> PFPS_PA0_T, ch1 -> PFPS_PA1_T, ch2 -> PFPS_PA2_T, ch3 -> PFPS_PA3_T
        ch = _gain_channel(gain)
        pa_col = f"PFPS_PA{ch}_T"

        # PFPS_PA{ch}_T comes from telemetry-per-gain CSVs (training CPTs only)
        tele_path = os.path.join(TELEM_PER_GAIN_DIR, f"{gain}.csv")
        tele_df = pd.read_csv(tele_path)
        if "CPT" not in tele_df.columns or pa_col not in tele_df.columns:
            raise KeyError(f"{tele_path} must have columns 'CPT' and '{pa_col}'")

        # Align telemetry rows to CPT_NAMES order (so temps match gain matrix rows)
        tele_df = tele_df.set_index("CPT").reindex(CPT_NAMES)
        if tele_df[pa_col].isna().any():
            missing_cpts = tele_df[tele_df[pa_col].isna()].index.tolist()
            raise ValueError(f"Missing {pa_col} for CPTs {missing_cpts} in {tele_path}")

        temps = tele_df[pa_col].astype(float).values  # shape: (n_cpt,)

        # Load raw measured gain curves and frequency axis
        matrix, freqs = load_gain_vectors(cpt_dirs, gain)

        # Compute preamp correction factors using the correct channel temp
        corrections = compute_preamp_corrections(freqs, temps, devfile)  # (n_cpt, n_freq)

        # Apply correction: divide raw by preamp gain ratio vs 20C
        corrected_matrix = matrix / corrections

        # corrected matrix CSV
        corr_outfile = os.path.join(MATRIX_DIR, f"{gain}_matrix_corrected.csv")
        with open(corr_outfile, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["CPT"] + [f"{freq:.6g}" for freq in freqs])
            for name, row in zip(CPT_NAMES, corrected_matrix):
                w.writerow([name] + row.tolist())

        # PCA on corrected matrix
        pcs_abs, mean_vec, eigvecs = compute_pca_abs(corrected_matrix)
        write_pc_csv(
            os.path.join(PREAMP_DIR, f"pca_abs_{gain}_corrected.csv"),
            CPT_NAMES,
            pcs_abs
        )

        # Save PCA ingredients for later reconstructions
        np.save(os.path.join(PREAMP_DIR, f"{gain}_mean.npy"), mean_vec)
        np.save(os.path.join(PREAMP_DIR, f"{gain}_eigvecs.npy"), eigvecs)
        np.save(os.path.join(PREAMP_DIR, f"{gain}_freqs.npy"), freqs)

        print(f"[Phase 1] Finished {gain}: corrected matrix + PCA bases written")


# ---------------------------
# Phase 2: telemetry regression w/ jackknife filtering, reconstruct spectra
# ---------------------------

def get_telemetry_cols_for_gain(gain: str):
    ch = _gain_channel(gain)
    adc_col = "SPE_ADC0_T" if ch in (0, 1) else "SPE_ADC1_T"
    return ["THERM_FPGA", adc_col, "SPE_1VAD8_V", "VMON_1V2D", "SPE_1VAD8_C"]

def get_telemetry_cols_for_gain_withpreamps(gain: str):
    ch = _gain_channel(gain)
    adc_col = "SPE_ADC0_T" if ch in (0, 1) else "SPE_ADC1_T"
    pfp_col = f"PFPS_PA{ch}_T"
    return ["THERM_FPGA", adc_col, "SPE_1VAD8_V", "VMON_1V2D", pfp_col, "SPE_1VAD8_C"]

def build_feature_matrix(X, tele_cols, order=2):
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

def phase2_jackknife_on_corrected_pca():
    all_alpha_records = []
    filtered_out_records = []
    rms_error_records = []
    final_alpha_records = []  # (refit) coefficients on kept terms

    predictions_per_pc = {g: {pc: {} for pc in PC_COMPONENTS} for g in GAINS_TO_RUN}
    reliable_terms_lookup = {}

    # ----- regression + jackknife screening -----
    for pc in PC_COMPONENTS:
        for gain in GAINS_TO_RUN:
            tele_cols = get_telemetry_cols_for_gain(gain)
            print(f"[Phase 2] Processing {gain} ({pc})...")

            pca_path = os.path.join(PREAMP_DIR, f"pca_abs_{gain}_corrected.csv")
            if not os.path.exists(pca_path):
                print(f"[Phase 2] Missing PCA file: {pca_path}")
                continue

            pca_df = pd.read_csv(pca_path)[["CPT", pc]]
            tele_df = pd.read_csv(os.path.join(TELEM_PER_GAIN_DIR, f"{gain}.csv"))[["CPT"] + tele_cols]

            df = (pd.merge(pca_df, tele_df, on="CPT", how="inner")
                    .set_index("CPT")
                    .reindex([c for c in CPT_NAMES if c in pca_df["CPT"].values])
                    .dropna()
                    .reset_index())

            y = df[pc].values
            X = df[tele_cols].values
            n = len(y)

            for order in [1, 2]:
                model_name = "linear" if order == 1 else "quadratic"
                Z, labels = build_feature_matrix(X, tele_cols, order)
                alpha_orig = np.linalg.lstsq(Z, y, rcond=None)[0]

                jackknife_alphas = []
                for i in range(n):
                    mask = np.ones(n, dtype=bool); mask[i] = False
                    alpha_i = np.linalg.lstsq(Z[mask], y[mask], rcond=None)[0]
                    jackknife_alphas.append(alpha_i)
                jackknife_alphas = np.vstack(jackknife_alphas)

                var_jack = np.var(jackknife_alphas, axis=0, ddof=1)
                stderr = np.sqrt(var_jack / n)

                for (label, a_val, se_val) in zip(labels, alpha_orig, stderr):
                    ratio = (abs(a_val / se_val) if se_val != 0 else np.inf)
                    rec = {
                        "gain_setting": gain,
                        "component": pc,
                        "model": model_name,
                        "term": label,
                        "alpha": float(a_val),
                        "stderr": float(se_val),
                        "ratio": float(ratio),
                    }
                    all_alpha_records.append(rec)
                    if ratio <= THRESHOLD_RATIO:
                        filtered_out_records.append(rec)
                    else:
                        reliable_terms_lookup.setdefault((gain, pc, model_name), set()).add(label)

            # ----- refit final models on kept terms and store predicted PCs -----
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

                kept_labels = [labels[i] for i in keep_idx]
                for term, a in zip(kept_labels, alpha_filt):
                    final_alpha_records.append({
                        "gain_setting": gain,
                        "component": pc,
                        "model": model_name,
                        "term": term,
                        "alpha_refit": float(a),
                    })

    # ----- exports: alphas + unreliable list -----
    if filtered_out_records:
        with open(os.path.join(ALPHAS_DIR, "unreliable_alphas.txt"), "w") as f:
            for r in filtered_out_records:
                f.write(f"{r['component']}, {r['gain_setting']}, {r['model']}, {r['term']}, "
                        f"alpha={r['alpha']:.6g}, stderr={r['stderr']:.6g}, ratio={r['ratio']:.3f}\n")

    pd.DataFrame(all_alpha_records).to_csv(
        os.path.join(ALPHAS_DIR, "jackknife_alphas_with_errors.csv"), index=False
    )

    reliable_alpha_records = [r for r in all_alpha_records if abs(r["alpha"] / r["stderr"]) > THRESHOLD_RATIO]
    pd.DataFrame(reliable_alpha_records).to_csv(
        os.path.join(ALPHAS_DIR, "reliable_alphas.csv"), index=False
    )

    if final_alpha_records:
        pd.DataFrame(final_alpha_records).to_csv(
            os.path.join(ALPHAS_DIR, "alpha_refit.csv"), index=False
        )

    # ----- plotting PC predicted-vs-actual and RMS tables -----
    for pc in PC_COMPONENTS:
        for gain in GAINS_TO_RUN:
            tele_cols = get_telemetry_cols_for_gain(gain)
            pca_path = os.path.join(PREAMP_DIR, f"pca_abs_{gain}_corrected.csv")
            if not os.path.exists(pca_path):
                continue

            pca_df = pd.read_csv(pca_path)[["CPT", pc]]
            tele_df = pd.read_csv(os.path.join(TELEM_PER_GAIN_DIR, f"{gain}.csv"))[["CPT"] + tele_cols]

            df = (pd.merge(pca_df, tele_df, on="CPT", how="inner")
                    .set_index("CPT")
                    .reindex([c for c in CPT_NAMES if c in pca_df["CPT"].values])
                    .dropna()
                    .reset_index())

            y_true = df[pc].values
            X = df[tele_cols].values

            predictions = {}
            rms_vals = {}

            y_mean = np.full_like(y_true, np.mean(y_true))
            predictions["Mean only"] = y_mean
            rms_vals["Mean only"] = float(np.sqrt(mean_squared_error(y_true, y_mean)))

            # Linear
            Z1, labels1 = build_feature_matrix(X, tele_cols, order=1)
            keep1 = reliable_terms_lookup.get((gain, pc, "linear"), set())
            keep_idx1 = [i for i, lbl in enumerate(labels1) if lbl in keep1]
            if keep_idx1:
                Z1_f = Z1[:, keep_idx1]
                a1 = np.linalg.lstsq(Z1_f, y_true, rcond=None)[0]
                y1 = Z1_f @ a1
                predictions["Linear"] = y1
                rms_vals["Linear"] = float(np.sqrt(mean_squared_error(y_true, y1)))

            # Quadratic
            Z2, labels2 = build_feature_matrix(X, tele_cols, order=2)
            keep2 = reliable_terms_lookup.get((gain, pc, "quadratic"), set())
            keep_idx2 = [i for i, lbl in enumerate(labels2) if lbl in keep2]
            if keep_idx2:
                Z2_f = Z2[:, keep_idx2]
                a2 = np.linalg.lstsq(Z2_f, y_true, rcond=None)[0]
                y2 = Z2_f @ a2
                predictions["Quadratic"] = y2
                rms_vals["Quadratic"] = float(np.sqrt(mean_squared_error(y_true, y2)))

            for model_name, rms_val in rms_vals.items():
                rms_error_records.append({
                    "component": pc,
                    "gain_setting": gain,
                    "model": model_name,
                    "rms_error": round(float(rms_val), 6),
                    "preamp_removed": True
                })

            for label, y_pred in predictions.items():
                plt.figure(figsize=(6, 6))
                plt.scatter(y_true, y_pred, alpha=0.8)
                lo, hi = float(y_true.min()), float(y_true.max())
                plt.plot([lo, hi], [lo, hi], 'k--')
                plt.xlabel(f"Actual {pc}")
                plt.ylabel(f"Predicted {pc}")
                plt.title(f"{gain}: {label} ({pc})\n(Preamp removed)\nRMS={rms_vals[label]:.4f}")
                for i, cpt in enumerate(df["CPT"]):
                    plt.annotate(cpt, (y_true[i], y_pred[i]),
                                 textcoords="offset points", xytext=(4, 2),
                                 ha='left', fontsize=8, alpha=0.7)
                plt.grid(True); plt.tight_layout()
                out_png = os.path.join(PC_SCATTER_DIR, f"{gain}_scatter_{label.replace(' ','_').lower()}_{pc.lower()}.png")
                plt.savefig(out_png); plt.close()

    pd.DataFrame(rms_error_records).to_csv(
        os.path.join(METRICS_DIR, "rms_errors.csv"), index=False
    )

    print("[Phase 2] Jackknife + PC plots done. Alphas/metrics exported.")

    # ----- reconstruct predicted gains from predicted PCs -----
    def reconstruct_gains(pred_pcs, mean_vec, eigvecs, k):
        V_k = eigvecs[:, :k]
        return mean_vec + pred_pcs @ V_k.T

    nrms_records = []

    for gain in GAINS_TO_RUN:
        mean_vec = np.load(os.path.join(PREAMP_DIR, f"{gain}_mean.npy"))
        eigvecs  = np.load(os.path.join(PREAMP_DIR, f"{gain}_eigvecs.npy"))
        freqs    = np.load(os.path.join(PREAMP_DIR, f"{gain}_freqs.npy"))

        measured_df = pd.read_csv(os.path.join(MATRIX_DIR, f"{gain}_matrix_corrected.csv"))
        measured = measured_df.iloc[:, 1:].to_numpy()

        pcs_df = pd.read_csv(os.path.join(PREAMP_DIR, f"pca_abs_{gain}_corrected.csv")).set_index("CPT").loc[CPT_NAMES]
        pcs_12 = pcs_df[["PC1", "PC2"]].to_numpy()

        actual_preds_12 = reconstruct_gains(pcs_12, mean_vec, eigvecs, k=2)

        np.savetxt(
            os.path.join(SPECTRA_PRED_DIR, f"{gain}_actual_PC12_reconstruction.csv"),
            actual_preds_12, delimiter=",",
            header=",".join([f"{f:.6g}" for f in freqs]), comments=''
        )

        errors_actual = measured - actual_preds_12
        per_cpt_rms_actual = np.sqrt(np.mean(errors_actual**2, axis=1))
        per_cpt_mean = np.mean(measured, axis=1)
        per_cpt_nrms_actual = per_cpt_rms_actual / per_cpt_mean

        global_nrms_actual = float(np.sqrt(np.mean(errors_actual**2)) / np.mean(measured))

        for cpt, val in zip(CPT_NAMES, per_cpt_nrms_actual):
            nrms_records.append({"gain": gain, "model": "Actual", "case": "PC12", "CPT": cpt, "NRMS": float(val)})
        nrms_records.append({"gain": gain, "model": "Actual", "case": "PC12", "CPT": "GLOBAL", "NRMS": global_nrms_actual})

        for model in ["Linear", "Quadratic"]:
            pc1 = predictions_per_pc[gain]["PC1"].get(model.lower(), None)
            pc2 = predictions_per_pc[gain]["PC2"].get(model.lower(), None)
            if pc1 is None and pc2 is None:
                continue

            pcs_12_pred = np.column_stack([
                pc1 if pc1 is not None else np.zeros(len(CPT_NAMES)),
                pc2 if pc2 is not None else np.zeros(len(CPT_NAMES)),
            ])

            preds_12 = reconstruct_gains(pcs_12_pred, mean_vec, eigvecs, k=2)

            np.savetxt(
                os.path.join(SPECTRA_PRED_DIR, f"{gain}_{model.lower()}_PC12_predicted.csv"),
                preds_12, delimiter=",",
                header=",".join([f"{f:.6g}" for f in freqs]), comments=''
            )

            errors = measured - preds_12
            per_cpt_rms  = np.sqrt(np.mean(errors**2, axis=1))
            per_cpt_mean = np.mean(measured, axis=1)
            per_cpt_nrms = per_cpt_rms / per_cpt_mean
            global_nrms = float(np.sqrt(np.mean(errors**2)) / np.mean(measured))

            for cpt, val in zip(CPT_NAMES, per_cpt_nrms):
                nrms_records.append({"gain": gain, "model": model, "case": "PC12", "CPT": cpt, "NRMS": float(val)})
            nrms_records.append({"gain": gain, "model": model, "case": "PC12", "CPT": "GLOBAL", "NRMS": global_nrms})

            sigma_per_freq = np.sqrt(np.mean(errors**2, axis=0))

            for i, cpt in enumerate(CPT_NAMES):
                fig, ax = plt.subplots(figsize=(8.5, 5.2))

                ax.plot(freqs, measured[i, :], label="Measured (corrected)", linewidth=1.6)
                ax.errorbar(
                    freqs, preds_12[i, :], yerr=sigma_per_freq,
                    fmt="--", label="Predicted (PC12)", alpha=0.90, capsize=3
                )
                ax.plot(freqs, actual_preds_12[i, :], label="Actual PC12", linestyle=":")

                ax.set_xlabel("Frequency (MHz)")
                ax.set_ylabel("Gain (linear)")

                # Two key metrics
                nrms_pred = per_cpt_nrms[i]
                nrms_pc3p = per_cpt_nrms_actual[i]   # truncation error from excluding PC3+

                fig.suptitle(f"{gain} {model} PC12 — {cpt}", fontsize=12, y=0.98)
                ax.set_title(
                    f"nRMS(pred) = {nrms_pred:.3f}    |    nRMS(actual) = {nrms_pc3p:.3f}",
                    fontsize=10
                )

                ax.legend()
                ax.grid(True)
                fig.tight_layout(rect=(0, 0, 1, 0.94))

                out_png = os.path.join(
                    SPECTRA_PLOT_DIR,
                    f"{gain}_{model.lower()}_PC12_{cpt}.png"
                )
                fig.savefig(out_png)
                plt.close(fig)

    pd.DataFrame(nrms_records).to_csv(os.path.join(METRICS_DIR, "nrms_errors.csv"), index=False)


# ---------------------------
# Raw-with-preamps comparison pipeline
# ---------------------------

def phase_withpreamps_pc_plots():
    """
    No preamp correction:
      - PCA from raw gain matrices (per gain)
      - Fit PC1/PC2 vs telemetry (includes PFPS_PA{ch}_T)
      - Jackknife filter unreliable terms
      - Export predicted-vs-actual scatters + RMS tables
    """
    with open(CPT_DIR_LIST) as f:
        cpt_dirs = [line.strip() for line in f if line.strip()]
    assert len(cpt_dirs) == len(CPT_NAMES), "Mismatch: CPT directories and names"

    rms_error_records = []

    for gain in GAINS_TO_RUN:
        raw_matrix, _freqs = load_gain_vectors(cpt_dirs, gain)
        pcs_abs, _mean_vec, _eigvecs = compute_pca_abs(raw_matrix)

        pcs_df = pd.DataFrame({
            "CPT": CPT_NAMES,
            "PC1": pcs_abs[:, 0],
            "PC2": pcs_abs[:, 1] if pcs_abs.shape[1] > 1 else np.zeros(len(CPT_NAMES))
        })

        tele_cols = get_telemetry_cols_for_gain_withpreamps(gain)
        tele_path = os.path.join(TELEM_PER_GAIN_DIR, f"{gain}.csv")
        tele_df = pd.read_csv(tele_path)

        missing = [c for c in ["CPT"] + tele_cols if c not in tele_df.columns]
        if missing:
            print(f"[Raw] Skipping {gain}: missing {missing} in {tele_path}")
            continue

        merged = (pd.merge(pcs_df, tele_df[["CPT"] + tele_cols], on="CPT", how="inner")
                    .set_index("CPT")
                    .reindex([c for c in CPT_NAMES if c in pcs_df["CPT"].values])
                    .dropna()
                    .reset_index())

        for pc in ["PC1", "PC2"]:
            if pc not in merged.columns:
                continue

            y = merged[pc].values
            X = merged[tele_cols].values
            n = len(y)

            predictions = {"Mean only": np.full_like(y, np.mean(y))}
            rms_vals = {"Mean only": float(np.sqrt(mean_squared_error(y, predictions["Mean only"])))}

            for order in [1, 2]:
                model_name = "linear" if order == 1 else "quadratic"
                Z, labels = build_feature_matrix(X, tele_cols, order)
                alpha_full = np.linalg.lstsq(Z, y, rcond=None)[0]

                jack = []
                for i in range(n):
                    mask = np.ones(n, dtype=bool); mask[i] = False
                    ai = np.linalg.lstsq(Z[mask], y[mask], rcond=None)[0]
                    jack.append(ai)
                jack = np.vstack(jack)
                stderr = np.sqrt(np.var(jack, axis=0, ddof=1) / n)

                keep_idx = []
                for i, (a, se) in enumerate(zip(alpha_full, stderr)):
                    ratio = (abs(a / se) if se != 0 else np.inf)
                    if ratio > THRESHOLD_RATIO:
                        keep_idx.append(i)

                if keep_idx:
                    Zf = Z[:, keep_idx]
                    af = np.linalg.lstsq(Zf, y, rcond=None)[0]
                    y_pred = Zf @ af
                    key = model_name.capitalize()
                    predictions[key] = y_pred
                    rms_vals[key] = float(np.sqrt(mean_squared_error(y, y_pred)))

            for label, y_pred in predictions.items():
                plt.figure(figsize=(6, 6))
                plt.scatter(y, y_pred, alpha=0.8)
                lo, hi = float(np.min(y)), float(np.max(y))
                plt.plot([lo, hi], [lo, hi], 'k--')
                plt.xlabel(f"Actual {pc}")
                plt.ylabel(f"Predicted {pc}")
                plt.title(f"{gain}: {label} ({pc})\n(Raw, no correction)\nRMS={rms_vals[label]:.4f}")
                for i, cpt in enumerate(merged["CPT"]):
                    plt.annotate(cpt, (y[i], y_pred[i]),
                                 textcoords="offset points", xytext=(4, 2),
                                 ha='left', fontsize=8, alpha=0.7)
                plt.grid(True); plt.tight_layout()
                out_png = os.path.join(RAW_PC_SCATTER_DIR, f"{gain}_scatter_{label.replace(' ','_').lower()}_{pc.lower()}.png")
                plt.savefig(out_png); plt.close()

            for model_name, rms_val in rms_vals.items():
                rms_error_records.append({
                    "component": pc,
                    "gain_setting": gain,
                    "model": model_name,
                    "rms_error": round(float(rms_val), 6),
                    "preamp_removed": False
                })

    pd.DataFrame(rms_error_records).to_csv(os.path.join(RAW_METRICS_DIR, "rms_errors.csv"), index=False)
    print("[Raw] PC plots + RMS exported.")


# ---------------------------
# Compare preamp vs raw: stacked RMS charts
# ---------------------------

def _stacked_rms_bar_chart(gain, pc, with_rows, without_rows, outfile):
    order = ["Mean only", "Linear", "Quadratic"]
    color_map = {"Mean only": "gray", "Linear": "skyblue", "Quadratic": "seagreen"}

    w_map = {m: float(e) for m, e in zip(with_rows["model"], with_rows["rms_error"])}
    wo_map = {m: float(e) for m, e in zip(without_rows["model"], without_rows["rms_error"])}

    bars_with = [m for m in order if m in w_map]
    vals_with = [w_map[m] for m in bars_with]
    bars_wo = [m for m in order if m in wo_map]
    vals_wo = [wo_map[m] for m in bars_wo]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7.0, 6.0), sharex=True)

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

    draw_panel(ax1, bars_with, vals_with,  "(Corrected / preamp removed)")
    draw_panel(ax2, bars_wo,   vals_wo,    "(Raw / with preamps)")

    fig.suptitle(f"{gain}: RMS Comparison ({pc})", y=0.98, fontsize=12)
    fig.subplots_adjust(top=0.88, bottom=0.18, hspace=0.35)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

def build_stacked_rms_comparisons():
    corr_csv = os.path.join(METRICS_DIR, "rms_errors.csv")
    raw_csv  = os.path.join(RAW_METRICS_DIR, "rms_errors.csv")

    if not (os.path.exists(corr_csv) and os.path.exists(raw_csv)):
        print("[Compare] Missing RMS CSV(s); skipping stacked comparisons.")
        return

    df_corr = pd.read_csv(corr_csv)
    df_raw  = pd.read_csv(raw_csv)

    needed = {"component", "gain_setting", "model", "rms_error"}
    if not needed.issubset(df_corr.columns) or not needed.issubset(df_raw.columns):
        print("[Compare] RMS CSV missing required columns; skipping.")
        return

    for pc in PC_COMPONENTS:
        for gain in GAINS_TO_RUN:
            c_rows = df_corr[(df_corr["component"] == pc) & (df_corr["gain_setting"] == gain)][["model", "rms_error"]]
            r_rows = df_raw[(df_raw["component"] == pc) & (df_raw["gain_setting"] == gain)][["model", "rms_error"]]
            if c_rows.empty and r_rows.empty:
                continue
            out_png = os.path.join(COMPARE_STACKED_DIR, f"{gain}_rms_comparison_{pc.lower()}_stacked.png")
            _stacked_rms_bar_chart(gain, pc, c_rows, r_rows, out_png)
            print(f"[Compare] Wrote {out_png}")


# ---------------------------
# Cleanup helper (optional)
# ---------------------------

def _cleanup_single_phase_rms_pngs():
    if not CLEAN_OLD_SINGLE_PHASE:
        return
    for d in (PC_SCATTER_DIR, RAW_PC_SCATTER_DIR):
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.endswith(".png") and "_rms_comparison_" in name:
                try:
                    os.remove(os.path.join(d, name))
                except Exception as e:
                    print(f"[Cleanup] Could not remove {name} from {d}: {e}")


# ---------------------------
# CPT1: apply refit α to CPT1, plot ratios & spectra
# ---------------------------

CPT1_DATA_DIR = os.path.join(ROOT, "data", "CPTs", "CPT1", "20250401_130140", "spt", "session_cpt-short_awg")

def _cpt1_corrected_spectrum(gain, freqs_mhz, devfile, t_pfps):
    path = os.path.join(CPT1_DATA_DIR, "gain.dat")
    with open(path, 'r') as f:
        header = f.readline().strip().split()
        data = np.loadtxt(f)
    g = data[:, header.index(gain)]
    corr = compute_preamp_corrections(freqs_mhz, np.array([t_pfps]), devfile)[0]
    return g / corr

def _predict_pc_with_refit(alphas_refit, gain, pc, tele_row, tele_cols):
    sub = alphas_refit[
        (alphas_refit["gain_setting"] == gain) &
        (alphas_refit["component"] == pc) &
        (alphas_refit["model"] == "quadratic")
    ]
    if sub.empty:
        return None

    Z1, labels = build_feature_matrix(tele_row[tele_cols].values.reshape(1, -1), tele_cols, order=2)
    label2x = {lbl: Z1[0, i] for i, lbl in enumerate(labels)}

    yhat = 0.0
    for _, r in sub.iterrows():
        t = str(r["term"])
        if t in label2x:
            yhat += float(r["alpha_refit"]) * label2x[t]
    return float(yhat)

def _predict_training_pcs_with_refit(alphas_refit, gain, X_df, tele_cols):
    preds = {}
    Z, labels = build_feature_matrix(X_df[tele_cols].values, tele_cols, order=2)
    label2i = {lbl: i for i, lbl in enumerate(labels)}

    for pc in PC_COMPONENTS:
        sub = alphas_refit[
            (alphas_refit["gain_setting"] == gain) &
            (alphas_refit["component"] == pc) &
            (alphas_refit["model"] == "quadratic")
        ]
        if sub.empty:
            continue

        a = np.zeros(Z.shape[1])
        for _, r in sub.iterrows():
            t = str(r["term"])
            if t in label2i:
                a[label2i[t]] = float(r["alpha_refit"])
        preds[pc] = Z @ a

    return preds

def cpt1_from_final_refit_alphas():
    refit_path = os.path.join(ALPHAS_DIR, "alpha_refit.csv")
    if not os.path.exists(refit_path):
        print("[CPT1-refit] Missing alpha_refit.csv; run Phase 2 first.")
        return
    alphas_refit = pd.read_csv(refit_path)

    ratio_store = {}

    linestyle_map = {"L": "-", "M": "--", "H": ":"}
    color_map = {0: "C0", 1: "C1", 2: "C2", 3: "C3"}

    for gain in GAINS_TO_RUN:
        mean = np.load(os.path.join(PREAMP_DIR, f"{gain}_mean.npy"))
        V    = np.load(os.path.join(PREAMP_DIR, f"{gain}_eigvecs.npy"))
        freqs= np.load(os.path.join(PREAMP_DIR, f"{gain}_freqs.npy"))
        V2 = V[:, :2]

        pcs_df = pd.read_csv(os.path.join(PREAMP_DIR, f"pca_abs_{gain}_corrected.csv"))[["CPT"] + PC_COMPONENTS]
        telecols = get_telemetry_cols_for_gain(gain)
        tele_df = pd.read_csv(os.path.join(TELEM_PER_GAIN_DIR, f"{gain}.csv"))[["CPT"] + telecols]

        df = (pd.merge(pcs_df, tele_df, on="CPT", how="inner")
                .set_index("CPT")
                .reindex([c for c in CPT_NAMES if c in pcs_df["CPT"].values])
                .dropna()
                .reset_index())
        if df.empty:
            print(f"[CPT1-refit] {gain}: no training rows.")
            continue

        corrmat = (pd.read_csv(os.path.join(MATRIX_DIR, f"{gain}_matrix_corrected.csv"))
                    .set_index("CPT").loc[df["CPT"].tolist()].to_numpy())

        tr_preds = _predict_training_pcs_with_refit(
            alphas_refit, gain, df[["CPT"] + telecols].drop(columns="CPT"), telecols
        )
        if not tr_preds:
            print(f"[CPT1-refit] {gain}: no kept terms.")
            continue

        pc1_tr = tr_preds.get("PC1", np.zeros(len(df)))
        pc2_tr = tr_preds.get("PC2", np.zeros(len(df)))
        corr_pred = mean + np.column_stack([pc1_tr, pc2_tr]) @ V2.T
        sigma_f = np.sqrt(np.mean((corrmat - corr_pred) ** 2, axis=0))

        # CPT1 telemetry row: prefer "{gain}_withCPT1.csv" (telemetry script output)
        cpt1_csv = os.path.join(TELEM_WITH_CPT1_DIR, f"{gain}_withCPT1.csv")
        if not os.path.exists(cpt1_csv):
            cpt1_csv = os.path.join(TELEM_WITH_CPT1_DIR, f"{gain}.csv")

        cpt1 = pd.read_csv(cpt1_csv)
        row1 = cpt1[cpt1["CPT"] == "CPT1"].iloc[0]

        pc1_hat = _predict_pc_with_refit(alphas_refit, gain, "PC1", row1, telecols) or 0.0
        pc2_hat = _predict_pc_with_refit(alphas_refit, gain, "PC2", row1, telecols) or 0.0

        pred_corr = (mean + np.array([[pc1_hat, pc2_hat]]) @ V2.T).reshape(-1)

        ch = _gain_channel(gain)
        pa_col = f"PFPS_PA{ch}_T"
        t_pfps = float(row1[pa_col])
        meas_corr = _cpt1_corrected_spectrum(gain, freqs, GAIN_TO_DEVFILE[gain], t_pfps)

        coeffs = (meas_corr - mean) @ V
        actual_pc12 = (mean + coeffs[:2] @ V2.T).reshape(-1)

        nrms_pred = float(np.sqrt(np.mean((meas_corr - pred_corr) ** 2)) / np.mean(meas_corr))
        nrms_actual = float(np.sqrt(np.mean((meas_corr - actual_pc12) ** 2)) / np.mean(meas_corr))

        eps = 1e-6
        valid = np.abs(pred_corr) > eps
        ratio = np.ones_like(pred_corr)
        ratio[valid] = meas_corr[valid] / pred_corr[valid]

        ratio_store[gain] = {"freqs": freqs.copy(), "ratio": ratio.copy()}

        pred_corr_ratio = pred_corr * ratio
        nrms_pred_ratio = float(np.sqrt(np.mean((meas_corr - pred_corr_ratio) ** 2)) / np.mean(meas_corr))

        ratio_df = pd.DataFrame({
            "freq_MHz": freqs,
            "ratio_meas_over_pred": ratio,
            "pred_corr_ratio": pred_corr_ratio,
        })
        ratio_df.to_csv(os.path.join(CPT1_RATIO_DIR, f"{gain}_CPT1_ratio_meas_over_pred.csv"), index=False)

        plt.figure(figsize=(8, 4.5))
        plt.plot(freqs, ratio, label="Measured / Predicted", linewidth=2)
        plt.axhline(1.0, linestyle="--", color="k", alpha=0.6)
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Ratio (meas / pred)")
        plt.title(f"{gain} CPT1 meas/pred vs frequency\nnRMS_pred={nrms_pred:.3f}, nRMS_pred_ratio={nrms_pred_ratio:.3f}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(CPT1_RATIO_DIR, f"{gain}_CPT1_ratio_meas_over_pred.png"))
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(freqs, meas_corr, label="Measured (corrected)", linewidth=1.5)
        ax.errorbar(freqs, pred_corr, yerr=sigma_f, fmt="--", label="Predicted (PC12)", alpha=0.9, capsize=3)
        ax.plot(freqs, actual_pc12, linestyle=":", label="Actual PC12")
        ax.plot(freqs, pred_corr_ratio, linestyle="-.", label="Predicted × ratio")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Gain (corrected, linear)")
        fig.suptitle(f"{gain} Quadratic PC12 – CPT1", fontsize=12, y=0.98)
        ax.set_title(f"nRMS_pred={nrms_pred:.3f}  nRMS_pred_ratio={nrms_pred_ratio:.3f}  nRMS_actual={nrms_actual:.3f}", fontsize=10)
        ax.legend(); ax.grid(True)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(os.path.join(CPT1_SPECTRA_DIR, f"{gain}_CPT1_PC12_pred_vs_measured_corrected.png"))
        plt.close(fig)

    if ratio_store:
        fig, ax = plt.subplots(figsize=(10, 6))
        for gain, d in ratio_store.items():
            freqs = d["freqs"]; ratio = d["ratio"]
            ch = _gain_channel(gain)
            gl = gain[0].upper()
            ax.plot(freqs, ratio,
                    color=color_map.get(ch, "C0"),
                    linestyle=linestyle_map.get(gl, "-"),
                    linewidth=1.8,
                    label=gain)
        ax.axhline(1.0, linestyle="--", color="k", alpha=0.6)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Ratio (meas / pred)")
        fig.suptitle("CPT1 Measured/Predicted Ratio vs Frequency (All Gains)", fontsize=13, y=0.98)
        ax.legend(ncol=3, fontsize=9, frameon=True)
        ax.grid(True)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_all = os.path.join(CPT1_RATIO_DIR, "CPT1_ratio_all_gains.png")
        fig.savefig(out_all)
        plt.close(fig)
        print(f"[CPT1-refit] Wrote combined ratio plot: {out_all}")


# ---------------------------
# Main
# ---------------------------

def main():
    phase1_preamp_correction_and_pca()
    phase2_jackknife_on_corrected_pca()
    phase_withpreamps_pc_plots()
    _cleanup_single_phase_rms_pngs()
    build_stacked_rms_comparisons()
    cpt1_from_final_refit_alphas()

if __name__ == "__main__":
    main()
