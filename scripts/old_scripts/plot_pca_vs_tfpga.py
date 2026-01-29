import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_TFPGA_vs_PCs(pca_dir, tfpga_csv, variant):
    tfpga_df = pd.read_csv(os.path.expanduser(tfpga_csv))
    tfpga_df.set_index("CPT", inplace=True)

    gain_settings = [col for col in tfpga_df.columns if col not in ["mean TFPGA"]]

    pcs = ["PC1", "PC2", "PC3"]

    # Make sure output directory exists
    outdir = os.path.expanduser("~/uncrater/data/plots/overall/plot_pca_vs_tfpga")
    os.makedirs(outdir, exist_ok=True)

    for gain in gain_settings:
        pca_file = os.path.join(os.path.expanduser(pca_dir), f"{variant}_{gain}_top3.csv")
        if not os.path.exists(pca_file):
            print(f"Missing PCA file for {gain}: {pca_file}")
            continue

        pca_df = pd.read_csv(pca_file)
        pca_df.set_index("CPT", inplace=True)

        merged = pca_df.join(tfpga_df[["mean TFPGA"]], how="inner")

        if merged.empty:
            print(f"No matching CPTs for {gain}")
            continue

        for pc in pcs:
            if pc not in merged.columns:
                print(f"{pc} not found in {pca_file}")
                continue

            plt.figure()
            # mean TFPGA on x-axis, PC on y-axis
            plt.scatter(merged["mean TFPGA"], merged[pc], marker='o')
            for label, row in merged.iterrows():
                plt.annotate(label, (row["mean TFPGA"], row[pc]), fontsize=8)

            plt.xlabel("Mean THERM_FPGA")
            plt.ylabel(pc)
            plt.title(f"{variant} {pc} vs TFPGA for {gain}")
            plt.grid(True)

            outname = f"{variant}_{pc}_vs_TFPGA_{gain}.png"
            outpath = os.path.join(outdir, outname)
            plt.tight_layout()
            plt.savefig(outpath)
            print(f"Saved {outpath}")
            plt.close()

if __name__ == "__main__":
    plot_TFPGA_vs_PCs(
        pca_dir="~/uncrater/data/plots/overall",
        tfpga_csv="~/uncrater/data/plots/overall/mean_TFPGA.csv",
        variant="pca_abs"
    )

    plot_TFPGA_vs_PCs(
        pca_dir="~/uncrater/data/plots/overall",
        tfpga_csv="~/uncrater/data/plots/overall/mean_TFPGA.csv",
        variant="pca_frac"
    )
