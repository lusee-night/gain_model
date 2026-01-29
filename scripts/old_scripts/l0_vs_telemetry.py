#!/usr/bin/env python3
"""
l0_vs_telemetry.py – L0 gain vs telemetry means, freq-by-freq
Adds quadratic fit (zₐ, z_b, z_c) and shows R² in subtitle.
Only keeps plots where |zₘ| ≥ Z_THRESH or freq = 0.1 Hz.
Deletes variables that only show 0.1 Hz plots.
"""

from collections import defaultdict
from pathlib import Path
import re, shutil, numpy as np, pandas as pd, matplotlib.pyplot as plt
import scipy.stats as st

# ─── configuration ───────────────────────────────────────────
CPT_DIR_TXT = Path("~/uncrater/scripts/CPT_directories.txt").expanduser()
GAIN_FILE   = "gain.dat"

MEANS_CSV   = Path("~/uncrater/data/plots/overall/means.csv").expanduser()
PNG_DIR     = Path("~/uncrater/data/plots/L1").expanduser()

GAIN_SETTING = "L1"
Z_THRESH     = 2.0
FREQ_KEEP    = 0.1
# ─────────────────────────────────────────────────────────────

# start fresh
if PNG_DIR.exists(): shutil.rmtree(PNG_DIR)
PNG_DIR.mkdir(parents=True, exist_ok=True)

def get_cpt_id(p: Path) -> str|None:
    for part in p.parts:
        m = re.match(r"^cpt(\d+)", part, re.I)
        if m: return f"CPT{m.group(1)}"
    return None

def slug(t: str) -> str: return re.sub(r"[^\w\-]", "_", t)

# ─── 1. load means.csv ───────────────────────────────────────
means = pd.read_csv(MEANS_CSV)
for c in means.columns:
    if c.strip().lower().replace(" ", "") == "meantfpga":
        means = means.rename(columns={c: "TFPGA"}); break

it, il = means.columns.get_loc("TFPGA"), means.columns.get_loc("L1")
tele_cols = means.columns[it:il]  # includes TFPGA now

if tele_cols.empty:
    raise SystemExit("No telemetry columns.")

tele_by_cpt = {str(means.loc[i,"CPT"]): means.loc[i,tele_cols].to_dict()
               for i in means.index}
slug2orig = {slug(c):c for c in tele_cols}

# ─── 2. gather (tele, gain) pairs ────────────────────────────
data = defaultdict(list)   # (freq, slug) → [(tele, gain)]

with CPT_DIR_TXT.open() as fh:
    for line in fh:
        sess = Path(line.strip()).expanduser();  cid = get_cpt_id(sess)
        if not cid or cid not in tele_by_cpt: continue
        gfile = sess/GAIN_FILE
        if not gfile.is_file(): continue

        tvals = tele_by_cpt[cid]
        dg = pd.read_csv(gfile, sep=r"\s+", comment="#", engine="python")
        fcol = dg.columns[0]

        for _, r in dg.iterrows():
            freq = float(r[fcol]);  gval = r.get(GAIN_SETTING)
            if pd.isna(gval): continue
            for col, tval in tvals.items():
                if pd.isna(tval): continue
                data[(freq, slug(col))].append((float(tval), float(gval)))

# ─── 3. fit, plot, and filter ────────────────────────────────
created = defaultdict(list)

for (freq, s), pts in data.items():
    pts = np.asarray(pts);   x, y = pts[:,0], pts[:,1]
    if len(x) < 2: continue

    try:
        (m, b_lin), cov_lin = np.polyfit(x, y, 1, cov=True)
        z_m = m / np.sqrt(cov_lin[0, 0]) if cov_lin[0, 0] > 0 else np.nan
    except Exception:
        z_m = np.nan; m = b_lin = 0

    # Skip if z is weak and not 0.1 Hz
    if abs(z_m) <= Z_THRESH and not np.isclose(freq, FREQ_KEEP, atol=1e-6):
        continue

    # Quadratic fit
    have_quad = len(x) >= 3 and np.unique(x).size >= 3
    if have_quad:
        try:
            coeff, covq = np.polyfit(x, y, 2, cov=True)
            have_quad &= covq.shape == (3, 3)
        except (np.linalg.LinAlgError, ValueError):
            have_quad = False
    if have_quad:
        a, bq, c = coeff
        sa, sb, sc = np.sqrt(np.diag(covq))
        z_a = a / sa if sa > 0 else np.nan
        z_b = bq / sb if sb > 0 else np.nan
        z_c = c / sc if sc > 0 else np.nan
        y_quad = a * x**2 + bq * x + c

        resid = y - y_quad
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_quad = 1 - ss_res / ss_tot
    else:
        z_a = z_b = z_c = r2_quad = np.nan

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=30, label="data")
    xf = np.linspace(x.min(), x.max(), 200)
    ax.plot(xf, m * xf + b_lin, lw=1.2, label=f"lin z={z_m:+.2f}")
    if have_quad:
        ax.plot(xf, a * xf**2 + bq * xf + c, lw=1.2, ls="--",
                label=f"quad zₐ={z_a:+.2f} z_b={z_b:+.2f} z_c={z_c:+.2f}")
    ax.set_xlabel(f"Mean {slug2orig[s]}")
    ax.set_ylabel("Gain (L1)")
    ax.set_title(f"L1 @ {freq:.3g} Hz  vs {slug2orig[s]}")
    subtitle = (f"lin z={z_m:+.2f}    quad: zₐ={z_a:+.2f} z_b={z_b:+.2f} "
                f"z_c={z_c:+.2f}    R²={r2_quad:.3f}")
    ax.text(0.5, -0.13, subtitle, transform=ax.transAxes,
            ha="center", va="top", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()

    fname = f"L1_{s}_{freq:.6g}Hz.png".replace("+", "").replace("-", "m")
    fig.savefig(PNG_DIR / fname, dpi=300)
    plt.close(fig)

    created[s].append(freq)
    print("✓", fname)

# ─── 4. prune only-0.1 Hz columns ────────────────────────────
uncorr=[]
for s, freqs in created.items():
    if all(np.isclose(freqs, FREQ_KEEP, atol=1e-6)):
        for fp in PNG_DIR.glob(f"L1_{s}_*Hz.png"):
            fp.unlink(True)
        uncorr.append(slug2orig[s])

if uncorr:
    out = PNG_DIR / "uncorrelated_columns.txt"
    out.write_text("\n".join(uncorr) + "\n")
    print("Uncorrelated cols (only 0.1 Hz):", ", ".join(uncorr))
    print("→", out)
else:
    print("All kept columns showed significance at ≥1 freq ≠0.1 Hz.")

print("\nINTERPRETATION:")
print(" • R² close to 1 ⇒ quadratic model explains data well.")
print(" • |zₐ|, |z_b|, |z_c| ≥ 2 ⇒ individual coefficients significant.")
print("Finished — PNGs in", PNG_DIR)
