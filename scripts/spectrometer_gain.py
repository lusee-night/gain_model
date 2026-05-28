#!/usr/bin/env python3
"""
Importable module for predicting corrected gain spectra from PCA model artifacts.

Usage
-----
    from spectrometer_gain import SpectrometerGain

    sg = SpectrometerGain()          # uses default model dir
    freq, gain = sg.get_corrected_gain('H', 3, {
        'THERM_FPGA':  30.4,
        'SPE_ADC1_T':  29.8,
        'SPE_1VAD8_V': 1.799,
        'VMON_1V2D':   1.201,
        'SPE_1VAD8_C': 0.045,
    })
    # freq : np.ndarray shape (16,) — anchor frequencies in MHz
    # gain : np.ndarray shape (16,) — predicted corrected gain at each frequency

Dependencies: numpy, csv (stdlib), pathlib (stdlib).  No pandas required.
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "outputs" / "gain_pca_model" / "corrected"


class SpectrometerGain:
    """
    Load PCA gain model artifacts once and predict corrected gain spectra.

    Parameters
    ----------
    model_dir : str or Path, optional
        Path to the ``corrected/`` directory that contains ``phase1/`` and
        ``phase2/`` sub-directories.  Defaults to the standard repo location
        relative to this file.
    """

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = _DEFAULT_MODEL_DIR
        self._model_dir = Path(model_dir).expanduser().resolve()

        alpha_path = self._model_dir / "phase2" / "alphas" / "alpha_refit.csv"
        if not alpha_path.exists():
            raise FileNotFoundError(f"alpha_refit.csv not found at {alpha_path}")

        # _alphas[gain_setting][pc] = list of (term, alpha_float)
        # Only quadratic-model rows are kept (mirrors get_corrected_gain.py behaviour).
        self._alphas = defaultdict(lambda: defaultdict(list))
        with open(alpha_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row["model"] != "quadratic":
                    continue
                self._alphas[row["gain_setting"]][row["component"]].append(
                    (row["term"], float(row["alpha_refit"]))
                )

        # Per-gain numpy artifacts loaded lazily on first use.
        self._cache = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_corrected_gain(self, level: str, channel: int, telemetry: dict):
        """
        Predict corrected gain spectrum.

        Parameters
        ----------
        level : str
            Gain level: 'L', 'M', or 'H' (case-insensitive).
        channel : int
            Channel index: 0, 1, 2, or 3.
        telemetry : dict
            Must contain:
              - 'THERM_FPGA'
              - 'SPE_ADC0_T' (channels 0/1) **or** 'SPE_ADC1_T' (channels 2/3)
              - 'SPE_1VAD8_V'
              - 'VMON_1V2D'
              - 'SPE_1VAD8_C'

        Returns
        -------
        freq : np.ndarray, shape (16,)
            Anchor frequencies in MHz.
        gain_spectrum : np.ndarray, shape (16,)
            Predicted corrected gain at each anchor frequency.
        """
        gain_key = self._make_gain_key(level, channel)
        mean_vec, eigvecs, freqs = self._load_npy(gain_key)
        features = self._build_features(gain_key, telemetry)

        pc1 = self._predict_pc(gain_key, "PC1", features)
        pc2 = self._predict_pc(gain_key, "PC2", features)

        pred = mean_vec + pc1 * eigvecs[:, 0] + pc2 * eigvecs[:, 1]
        return freqs, pred

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_gain_key(level: str, channel: int) -> str:
        level = str(level).strip().upper()
        if level not in {"L", "M", "H"}:
            raise ValueError(f"level must be L, M, or H; got {level!r}")
        if channel not in {0, 1, 2, 3}:
            raise ValueError(f"channel must be 0–3; got {channel!r}")
        return f"{level}{channel}"

    def _load_npy(self, gain_key: str):
        if gain_key not in self._cache:
            pca_dir = self._model_dir / "phase1" / "pca"
            paths = {
                "mean":    pca_dir / f"{gain_key}_mean.npy",
                "eigvecs": pca_dir / f"{gain_key}_eigvecs.npy",
                "freqs":   pca_dir / f"{gain_key}_freqs.npy",
            }
            missing = [str(p) for p in paths.values() if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    "Missing model files:\n" + "\n".join(missing)
                )
            self._cache[gain_key] = (
                np.load(paths["mean"]),
                np.load(paths["eigvecs"]),
                np.load(paths["freqs"]),
            )
        return self._cache[gain_key]

    @staticmethod
    def _build_features(gain_key: str, telemetry: dict) -> dict:
        """Build the feature dict that mirrors build_feature_matrix() in get_corrected_gain.py."""
        ch = int(gain_key[-1])
        adc_key = "SPE_ADC0_T" if ch <= 1 else "SPE_ADC1_T"

        required = ["THERM_FPGA", adc_key, "SPE_1VAD8_V", "VMON_1V2D", "SPE_1VAD8_C"]
        missing = [k for k in required if k not in telemetry or telemetry[k] is None]
        if missing:
            raise ValueError(f"Missing telemetry keys for {gain_key}: {missing}")

        T  = float(telemetry["THERM_FPGA"])
        Tc = float(telemetry[adc_key])

        return {
            "1":                        1.0,
            "THERM_FPGA":               T,
            adc_key:                    Tc,
            "SPE_1VAD8_V":              float(telemetry["SPE_1VAD8_V"]),
            "VMON_1V2D":               float(telemetry["VMON_1V2D"]),
            "SPE_1VAD8_C":              float(telemetry["SPE_1VAD8_C"]),
            "THERM_FPGA*THERM_FPGA":    T * T,
            f"{adc_key}*{adc_key}":     Tc * Tc,
            f"THERM_FPGA*{adc_key}":    T * Tc,
        }

    def _predict_pc(self, gain_key: str, pc: str, features: dict) -> float:
        return sum(
            alpha * features.get(term, 0.0)
            for term, alpha in self._alphas[gain_key][pc]
        )
