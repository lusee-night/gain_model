#!/usr/bin/env python3
"""
gain_vs_tfpga.py  —  linear *and* quadratic fits, with z-scores
===============================================================
For every (frequency, gain-setting) across all CPTs:

1. Fit a linear model:        gain = m·T + b
2. Fit a quadratic model:     gain = a·T² + b·T + c        

For each coefficient we store:
    • value
    • σ   
    • z-score  = value / σ
and write per-coefficient CSVs in the familiar 13-column layout.

Outputs (all in  ~/uncrater/data/plots/overall/ ):
    gain_vs_tfpga_slopes.csv,
    gain_vs_tfpga_z_linear.csv,
    gain_vs_tfpga_quad_a|b|c.csv,
    gain_vs_tfpga_z_quad_a|b|c.csv,
    gain_vs_tfpga_quad_r2.csv

PNGs (scatter + solid linear + dashed quadratic) go to:
    /mnt/c/users/gtspe/Desktop/gain_vs_tfpga
"""

from collections import defaultdict
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────── configuration ─────────────────────────
CPT_DIR_TXT = Path("~/uncrater/scripts/CPT_directories.txt").expanduser()
GAIN_FILE   = "gain.dat"

TFPGA_CSV   = Path("~/uncrater/data/plots/overall/mean_TFPGA.csv").expanduser()
CSV_OUT_DIR = Path("~/uncrater/data/plots/overall").expanduser()
PNG_OUT_DIR = Path("/mnt/c/users/gtspe/Desktop/gain_vs_tfpga")

CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)
PNG_OUT_DIR.mkdir(parents=True, exist_ok=True)

GAIN_SETTINGS = [
    "L0", "M0", "H0",
    "L1", "M1", "H1",
    "L2", "M2", "H2",
    "L3", "M3", "H3",
]

# ────────────────── helper: canonical CPT id ─────────────────────
def get_cpt_id(path: Path) -> str | None:
    for part in path.parts:
        if part.lower().startswith("cpt"):
            core = part.split("+", 1)[0]
            m = re.match(r"^(cpt)(\d+)", core, re.IGNORECASE)
            if m:
                return f"CPT{m.group(2)}"
    return None

def r2_score(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot else np.nan

# ─────────────────── load mean_TFPGA table ──────────────────────
tfpga_df  = pd.read_csv(TFPGA_CSV)
tfpga_col = [c for c in tfpga_df.columns if c.lower().startswith("mean")][0]
tfpga_map = dict(zip(tfpga_df["CPT"].astype(str), tfpga_df[tfpga_col]))

# ─────────────────── gather data ────────────────────────────────
data = defaultdict(list)         # (freq, gset) → list[(T, gain)]

with CPT_DIR_TXT.open() as f:
    for raw in f:
        session_dir = Path(raw.strip()).expanduser()
        if not session_dir:
            continue
        cpt_id = get_cpt_id(session_dir)
        if not cpt_id or cpt_id not in tfpga_map:
            print(f"Skipping {session_dir} — CPT not in mean_TFPGA.csv")
            continue
        gain_path = session_dir / GAIN_FILE
        if not gain_path.is_file():
            print(f"{GAIN_FILE} missing in {session_dir}")
            continue

        tfpga = float(tfpga_map[cpt_id])
        df = pd.read_csv(gain_path, delim_whitespace=True,
                         comment="#", engine="python")
        freq_col = df.columns[0]

        for _, row in df.iterrows():
            freq = float(row[freq_col])
            for g in GAIN_SETTINGS:
                if g in row and not pd.isna(row[g]):
                    data[(freq, g)].append((tfpga, float(row[g])))

# ─────────────────── fit, plot, collect coeffs & z’s ────────────
# linear
slopes_by_freq   = defaultdict(dict)
z_lin_by_freq    = defaultdict(dict)
# quadratic
quad_a_by_freq   = defaultdict(dict);   z_quad_a = defaultdict(dict)
quad_b_by_freq   = defaultdict(dict);   z_quad_b = defaultdict(dict)
quad_c_by_freq   = defaultdict(dict);   z_quad_c = defaultdict(dict)
quad_r2_by_freq  = defaultdict(dict)

for (freq, gset), pts in data.items():
    pts = np.asarray(pts)
    if pts.shape[0] < 2:
        continue
    x, y = pts[:, 0], pts[:, 1]

    # ‣ linear fit + σ_m
    (m, b_lin), cov_lin = np.polyfit(x, y, 1, cov=True)
    sigma_m = np.sqrt(cov_lin[0, 0])
    z_m     = m / sigma_m if sigma_m > 0 else np.nan
    slopes_by_freq[freq][gset] = m
    z_lin_by_freq[freq][gset]  = z_m

    # ‣ quadratic fit for *all* freqs  (needs ≥3 points)
    if pts.shape[0] >= 3:
        (a, b, c), cov_quad = np.polyfit(x, y, 2, cov=True)
        sigma_a, sigma_b, sigma_c = np.sqrt(np.diag(cov_quad))
        z_a = a / sigma_a if sigma_a > 0 else np.nan
        z_b = b / sigma_b if sigma_b > 0 else np.nan
        z_c = c / sigma_c if sigma_c > 0 else np.nan
        quad_a_by_freq[freq][gset] = a;   z_quad_a[freq][gset] = z_a
        quad_b_by_freq[freq][gset] = b;   z_quad_b[freq][gset] = z_b
        quad_c_by_freq[freq][gset] = c;   z_quad_c[freq][gset] = z_c
        quad_r2_by_freq[freq][gset] = r2_score(y, a*x**2 + b*x + c)

    # ‣ PNG
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=20, label="data")
    x_fit = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_fit, m*x_fit + b_lin, lw=1.2,
            label=f"lin  z={z_m:+.2f}")
    if pts.shape[0] >= 3:
        ax.plot(x_fit, a*x_fit**2 + b*x_fit + c, lw=1.2, ls="--",
                label=f"quad  zₐ={z_a:+.2f}")
    ax.set_xlabel("mean TFPGA (°C)")
    ax.set_ylabel("Gain")
    ax.set_title(f"{gset}  @  {freq:.3g} Hz")
    ax.legend()
    fig.tight_layout()
    fname = f"{gset}_{freq:.6g}Hz.png".replace("+", "").replace("-", "m")
    fig.savefig(PNG_OUT_DIR / fname, dpi=300)
    plt.close(fig)

# ─────────────────── CSV writer helper ──────────────────────────
def write_table(freq_dict, suffix):
    if not freq_dict:
        return None
    df = (pd.DataFrame.from_dict(freq_dict, orient="index")
            .reindex(columns=GAIN_SETTINGS)
            .reset_index().rename(columns={"index": "frequency_Hz"})
            .sort_values("frequency_Hz"))
    out = CSV_OUT_DIR / f"gain_vs_tfpga_{suffix}.csv"
    df.to_csv(out, index=False)
    return out

# linear
write_table(slopes_by_freq, "slopes")
write_table(z_lin_by_freq,  "z_linear")

# quadratic
write_table(quad_a_by_freq, "quad_a")
write_table(quad_b_by_freq, "quad_b")
write_table(quad_c_by_freq, "quad_c")
write_table(z_quad_a,       "z_quad_a")
write_table(z_quad_b,       "z_quad_b")
write_table(z_quad_c,       "z_quad_c")
write_table(quad_r2_by_freq,"quad_r2")

print("✓ Completed — PNGs in", PNG_OUT_DIR, "and CSVs in", CSV_OUT_DIR)
