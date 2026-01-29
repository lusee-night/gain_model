import os
import pandas as pd
import matplotlib.pyplot as plt

# Map each gain to the corresponding PFPS column
GAIN_TO_PFPS = {
    "L0": "PFPS_PA0_T", "M0": "PFPS_PA0_T", "H0": "PFPS_PA0_T",
    "L1": "PFPS_PA1_T", "M1": "PFPS_PA1_T", "H1": "PFPS_PA1_T",
    "L2": "PFPS_PA2_T", "M2": "PFPS_PA2_T", "H2": "PFPS_PA2_T",
    "L3": "PFPS_PA3_T", "M3": "PFPS_PA3_T", "H3": "PFPS_PA3_T",
}

GAINS_ORDERED = ["L0","M0","H0","L1","M1","H1","L2","M2","H2","L3","M3","H3"]
PCS = ["PC1", "PC2", "PC3"]

def plot_PCs_vs_PFPS(pca_dir, means_csv, variant):
    """
    For each gain in GAINS_ORDERED, read `<pca_dir>/<variant>_<gain>_top3.csv`
    (must contain CPT, PC1, PC2, PC3), join with PFPS_PA*_T from means_csv by CPT,
    and plot: x = PFPS_PA*_T, y = PC1/PC2/PC3.
    """
    means_df = pd.read_csv(os.path.expanduser(means_csv))
    if "CPT" not in means_df.columns:
        raise ValueError(f"'CPT' column not found in {means_csv}")
    means_df.set_index("CPT", inplace=True)

    outdir = os.path.expanduser("~/uncrater/data/plots/overall/plot_pca_vs_means")
    os.makedirs(outdir, exist_ok=True)

    for gain in GAINS_ORDERED:
        pca_file = os.path.join(os.path.expanduser(pca_dir), f"{variant}_{gain}_top3.csv")
        if not os.path.exists(pca_file):
            print(f"Missing PCA file for {gain}: {pca_file}")
            continue

        pca_df = pd.read_csv(pca_file)
        if "CPT" not in pca_df.columns:
            print(f"'CPT' not found in {pca_file} — skipping {gain}")
            continue
        pca_df.set_index("CPT", inplace=True)

        pfps_col = GAIN_TO_PFPS[gain]
        if pfps_col not in means_df.columns:
            print(f"{pfps_col} not found in {means_csv} — skipping {gain}")
            continue

        merged = pca_df.join(means_df[[pfps_col]], how="inner")
        if merged.empty:
            print(f"No matching CPTs after join for {gain} ({pfps_col})")
            continue

        for pc in PCS:
            if pc not in merged.columns:
                print(f"{pc} not found in {pca_file} — skipping")
                continue

            plt.figure()
            # x = PFPS temperature; y = PC component
            plt.scatter(merged[pfps_col], merged[pc], marker='o')

            # Label points by CPT
            for label, row in merged.iterrows():
                plt.annotate(label, (row[pfps_col], row[pc]), fontsize=8)

            plt.xlabel(pfps_col)
            plt.ylabel(pc)
            plt.title(f"{variant}: {pc} vs {pfps_col} for {gain}")
            plt.grid(True)

            outname = f"{variant}_{pc}_vs_{pfps_col}_{gain}.png"
            outpath = os.path.join(outdir, outname)
            plt.tight_layout()
            plt.savefig(outpath)
            print(f"Saved {outpath}")
            plt.close()

if __name__ == "__main__":
    plot_PCs_vs_PFPS(
        pca_dir="~/uncrater/data/plots/overall",
        means_csv="~/uncrater/data/plots/overall/means.csv",
        variant="pca_abs"
    )
    plot_PCs_vs_PFPS(
        pca_dir="~/uncrater/data/plots/overall",
        means_csv="~/uncrater/data/plots/overall/means.csv",
        variant="pca_frac"
    )
