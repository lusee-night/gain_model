from pathlib import Path
import csv
import matplotlib.pyplot as plt

# File paths
cpt_file = Path.home() / "uncrater/scripts/CPT_directories.txt"
output_csv = Path.home() / "uncrater/data/plots/overall/mean_TFPGA.csv"
plot_dir = Path.home() / "uncrater/data/plots/overall"
gain_columns = ["L0", "M0", "H0", "L1", "M1", "H1", "L2", "M2", "H2", "L3", "M3", "H3"]

# Step 1: Clean CSV to keep only "CPT" and "mean TFPGA"
with open(output_csv) as f:
    reader = csv.DictReader(f)
    cleaned_rows = [{"CPT": row.get("CPT", "").strip(), "mean TFPGA": row.get("mean TFPGA", "").strip()} for row in reader]

# Step 2: Load directories
with open(cpt_file) as f:
    gain_dirs = [Path(line.strip()).expanduser() for line in f if line.strip()]

# Step 3: Compute gain means from gain.dat
gain_means_by_dir = []
for d in gain_dirs:
    gain_file = d / "gain.dat"
    try:
        with open(gain_file) as f:
            reader = csv.reader(f, delimiter="\t")
            headers = next(reader)
            indices = [headers.index(col) for col in gain_columns]
            sums = [0.0] * len(indices)
            count = 0
            for row in reader:
                for i, idx in enumerate(indices):
                    sums[i] += float(row[idx])
                count += 1
            means = [s / count if count > 0 else float('nan') for s in sums]
            gain_means_by_dir.append(means)
    except Exception as e:
        print(f"Failed to read {gain_file}: {e}")
        gain_means_by_dir.append([float("nan")] * len(gain_columns))

# Step 4: Merge into final CSV
final_header = ["CPT", "mean TFPGA"] + gain_columns
final_rows = []
for row, means in zip(cleaned_rows, gain_means_by_dir):
    final_rows.append([row["CPT"], row["mean TFPGA"]] + [f"{v:.8f}" for v in means])

with open(output_csv, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(final_header)
    writer.writerows(final_rows)

print(f"\nCSV updated: {output_csv}")

# Step 5: Generate and save labeled scatter plots (overwriting previous plots)
with open(output_csv) as f:
    reader = csv.DictReader(f)
    data = list(reader)

tfpga_vals = [float(row["mean TFPGA"]) for row in data]
cpt_labels = [row["CPT"] for row in data]

for col in gain_columns:
    try:
        gain_vals = [float(row[col]) for row in data]
        plt.figure(figsize=(8, 6))
        plt.scatter(gain_vals, tfpga_vals)
        for x, y, label in zip(gain_vals, tfpga_vals, cpt_labels):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 2), ha='left', fontsize=8)
        plt.xlabel(col)
        plt.ylabel("mean TFPGA")
        plt.title(f"{col} vs mean TFPGA")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{col}_vs_TFPGA.png")
        plt.close()
    except Exception as e:
        print(f"Failed to plot {col}: {e}")
