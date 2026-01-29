import os
import pandas as pd
import matplotlib.pyplot as plt

def read_lines(path):
    with open(os.path.expanduser(path), 'r') as f:
        return [line.strip() for line in f if line.strip()]

def plot_gain_vs_frequency_per_gain(input_dirs, output_dirs):
    for in_dir, out_dir in zip(input_dirs, output_dirs):
        gain_path = os.path.join(os.path.expanduser(in_dir), "gain.dat")
        if not os.path.exists(gain_path):
            print(f"Missing file: {gain_path}")
            continue

        df = pd.read_csv(gain_path, delim_whitespace=True)

        if "freq" not in df.columns:
            print(f"Missing 'freq' column in {gain_path}")
            continue

        freq = df["freq"]
        gain_cols = [col for col in df.columns if col != "freq"]

        os.makedirs(os.path.expanduser(out_dir), exist_ok=True)

        for col in gain_cols:
            plt.figure(figsize=(8, 5))
            plt.plot(freq, df[col], label=col)
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("Gain")
            plt.title(f"{col} vs Frequency\n{os.path.basename(in_dir)}")
            plt.grid(True)
            plt.tight_layout()

            out_path = os.path.join(os.path.expanduser(out_dir), f"gain_vs_frequency_{col}.png")
            plt.savefig(out_path)
            print(f"Saved {out_path}")
            plt.close()

if __name__ == "__main__":
    input_dirs = read_lines("~/uncrater/scripts/CPT_directories.txt")
    output_dirs = read_lines("~/uncrater/scripts/output_directories.txt")
    if len(input_dirs) != len(output_dirs):
        print("Mismatch: CPT_directories.txt and output_directories.txt must have the same number of lines")
    else:
        plot_gain_vs_frequency_per_gain(input_dirs, output_dirs)
