import csv
from pathlib import Path

# Define paths
base_path = Path.home() / "uncrater/data/gain_decode"
output_path = Path.home() / "uncrater/data/plots/overall"
output_path.mkdir(parents=True, exist_ok=True)
output_csv = output_path / "mean_TFPGA.csv"

# Collect (CPT name, mean value)
results = []

for csv_file in sorted(base_path.glob("CPT*/DCB_telemetry_decoded.csv")):
    cpt_name = csv_file.parent.name
    if cpt_name == "CPT1":
        continue  # Skip CPT1
    try:
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            values = [float(row[0]) for row in reader if len(row) > 0]
            mean_val = sum(values) / len(values) if values else float("nan")
            results.append((cpt_name, mean_val))
    except Exception as e:
        print(f"Failed to process {csv_file}: {e}")
        results.append((cpt_name, float("nan")))

# Sort results numerically by CPT number
def cpt_key(item):
    name = item[0]
    return int(name.replace("CPT", ""))

results.sort(key=cpt_key)

# Write results to output CSV
with open(output_csv, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["CPT", "mean TFPGA"])
    writer.writerows(results)

print(f"Mean TFPGA values (excluding CPT1) written to: {output_csv}")
