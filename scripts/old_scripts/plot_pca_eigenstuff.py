import os
import numpy as np
import matplotlib.pyplot as plt

CPT_NAMES = [
    "CPT2", "CPT3", "CPT4", "CPT5", "CPT6", "CPT7", "CPT8",
    "CPT9", "CPT10", "CPT11", "CPT12", "CPT13", "CPT15", "CPT16"
]

GAIN_SETTINGS = ["L0", "M0", "H0", "L1", "M1", "H1", "L2", "M2", "H2", "L3", "M3", "H3"]

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
        gain_column = data[:, col_idx]
        gain_matrix.append(gain_column)
    return np.array(gain_matrix), data[:, header.index("freq")]

def plot_eigenvectors_and_spectrum(matrix, freqs, gain, output_dir):
    mean = np.mean(matrix, axis=0)
    abs_resid = matrix - mean

    cov = abs_resid.T @ abs_resid / abs_resid.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Plot eigenvectors
    for i in range(min(3, evecs.shape[1])):
        plt.figure()
        plt.plot(freqs, evecs[:, i], marker='o')
        plt.title(f"{gain} - Eigenvector PC{i+1}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Component Weight")
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(output_dir, f"{gain}_eigenvector_PC{i+1}.png")
        plt.savefig(path)
        print(f"Saved {path}")
        plt.close()

    # Plot eigenvalue spectrum (log y-scale)
    plt.figure()
    plt.plot(range(1, len(evals)+1), evals, marker='o')
    plt.yscale('log')
    plt.xlabel("Component Index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title(f"{gain} - Eigenvalue Spectrum")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{gain}_eigenvalue_spectrum.png")
    plt.savefig(path)
    print(f"Saved {path}")
    plt.close()

def main():
    with open(os.path.expanduser('~/uncrater/scripts/CPT_directories.txt')) as f:
        cpt_dirs = [line.strip() for line in f if line.strip()]
    assert len(cpt_dirs) == len(CPT_NAMES), "Mismatch: CPT directories and names"

    output_dir = os.path.expanduser('~/uncrater/data/plots/overall')
    os.makedirs(output_dir, exist_ok=True)

    for gain in GAIN_SETTINGS:
        print(f"Analyzing {gain}...")
        matrix, freqs = load_gain_vectors(cpt_dirs, gain)
        plot_eigenvectors_and_spectrum(matrix, freqs, gain, output_dir)

if __name__ == "__main__":
    main()
