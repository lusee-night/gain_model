import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from sklearn.metrics import mean_squared_error

# === Configuration ===
BASE_PCA_DIR = os.path.expanduser("~/uncrater/data/plots/overall")
TELEMETRY_PATH = os.path.join(BASE_PCA_DIR, "means.csv")
OUTPUT_BASE = os.path.join(BASE_PCA_DIR, "gainalphas", "plots")
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Telemetry to use for channel-based regression
TELEMETRY_COLS = ["TFPGA", "SPE_ADC0_T", "PFPS_PA0_T"]

# Gain settings grouped by channel
CHANNELS = {
    0: ["L0", "M0", "H0"],
    1: ["L1", "M1", "H1"],
    2: ["L2", "M2", "H2"],
    3: ["L3", "M3", "H3"],
}

# === Feature Expansion ===
def build_feature_matrix(X, order=2):
    n_samples, n_features = X.shape
    Z = np.ones((n_samples, 1))  # constant term
    feature_labels = ["1"]

    if order >= 1:
        Z = np.hstack([Z, X])
        feature_labels += TELEMETRY_COLS

    if order == 2:
        quad_terms = []
        quad_labels = []
        for i, j in combinations_with_replacement(range(n_features), 2):
            quad_terms.append((X[:, i] * X[:, j]).reshape(-1, 1))
            quad_labels.append(f"{TELEMETRY_COLS[i]}*{TELEMETRY_COLS[j]}")
        Z = np.hstack([Z] + quad_terms)
        feature_labels += quad_labels

    return Z, feature_labels

# === Main Loop Over Channels ===
for ch, gain_settings in CHANNELS.items():
    print(f"\n=== Processing Channel {ch} ===")
    plot_dir = os.path.join(OUTPUT_BASE, f"channel{ch}_analysis")
    os.makedirs(plot_dir, exist_ok=True)
    alpha_table = []

    for gain in gain_settings:
        print(f"  - {gain}")
        pca_path = os.path.join(BASE_PCA_DIR, f"pca_abs_{gain}.csv")
        if not os.path.exists(pca_path):
            print(f"    [!] Missing PCA file: {pca_path}")
            continue

        # Load PC1 and telemetry
        pca_df = pd.read_csv(pca_path)[["CPT", "PC1"]]
        tele_df = pd.read_csv(TELEMETRY_PATH)[["CPT"] + TELEMETRY_COLS]
        df = pd.merge(pca_df, tele_df, on="CPT")
        y_true = df["PC1"].values
        X = df[TELEMETRY_COLS].values

        rms_values = []
        predictions = {}

        # --- Model 0: Mean Only ---
        y_pred_0 = np.full_like(y_true, np.mean(y_true))
        rms_values.append(np.sqrt(mean_squared_error(y_true, y_pred_0)))
        predictions["Mean only"] = y_pred_0

        # --- Model 1: Linear ---
        Z1, labels1 = build_feature_matrix(X, order=1)
        alpha1 = np.linalg.lstsq(Z1, y_true, rcond=None)[0]
        y_pred_1 = Z1 @ alpha1
        rms_values.append(np.sqrt(mean_squared_error(y_true, y_pred_1)))
        predictions["Linear"] = y_pred_1
        alpha_table.append({
            "gain_setting": gain,
            "model": "linear",
            **{label: coeff for label, coeff in zip(labels1, alpha1)}
        })

        # --- Model 2: Quadratic ---
        Z2, labels2 = build_feature_matrix(X, order=2)
        alpha2 = np.linalg.lstsq(Z2, y_true, rcond=None)[0]
        y_pred_2 = Z2 @ alpha2
        rms_values.append(np.sqrt(mean_squared_error(y_true, y_pred_2)))
        predictions["Quadratic"] = y_pred_2
        alpha_table.append({
            "gain_setting": gain,
            "model": "quadratic",
            **{label: coeff for label, coeff in zip(labels2, alpha2)}
        })

        # --- Scatter Plots with Labels ---
        for label, y_pred in predictions.items():
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true, y_pred, color="dodgerblue", alpha=0.8)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
            plt.xlabel("Actual PC1")
            plt.ylabel("Predicted PC1")
            plt.title(f"{gain}: {label} Model\nRMS = {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
            plt.grid(True)

            # Add CPT labels
            for i, cpt in enumerate(df["CPT"]):
                plt.annotate(cpt, (y_true[i], y_pred[i]), textcoords="offset points",
                             xytext=(4, 2), ha='left', fontsize=8, alpha=0.8)

            plt.tight_layout()
            figname = f"{gain}_scatter_{label.replace(' ', '_').lower()}.png"
            plt.savefig(os.path.join(plot_dir, figname))
            plt.close()

        # --- Bar Plot of RMS Errors ---
        plt.figure(figsize=(6, 4))
        plt.bar(["Mean only", "Linear", "Quadratic"], rms_values, color=["gray", "skyblue", "seagreen"])
        plt.ylabel("RMS Error")
        plt.title(f"{gain}: RMS Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{gain}_rms_comparison.png"))
        plt.close()

    # Save alpha values for this channel
    alpha_df = pd.DataFrame(alpha_table)
    alpha_df.to_csv(os.path.join(plot_dir, "alphas.csv"), index=False)

print("\nAll channels processed.")
