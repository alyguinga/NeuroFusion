"""
v3/features/spectral.py
═══════════════════════════════════════════════════════════════════════════
Spectral biomarker extraction for EEG-based Alzheimer's disease detection.

AD Spectral Signature ("spectral slowing"):
  - δ (1-4 Hz) and θ (4-8 Hz) power INCREASE
  - α (8-13 Hz) and β (13-30 Hz) power DECREASE
  - θ/α ratio INCREASES (primary AD index)
  - Peak Alpha Frequency (PAF) SLOWS below 8 Hz

Mathematical Foundations:
─────────────────────────
1. Power Spectral Density via Welch's method:
   P(f) = (1/KMU) * Σ_k |Σ_n x_k[n] w[n] e^{-j2πfn}|²
   where K = number of segments, M = segment length, U = window power,
   w[n] = Hanning window, x_k = k-th overlapping segment.

2. Absolute Band Power:
   P_band = ∫_{f_low}^{f_high} P(f) df
   Approximated via Simpson's rule on discrete PSD.

3. Relative Band Power (RBP):
   RBP_band = P_band / P_total
   where P_total = ∫_{1}^{45} P(f) df

4. Peak Alpha Frequency (PAF):
   PAF = argmax_{f ∈ [8,13]} P(f)
   Correlates with MMSE cognitive score (Clin Neurophysiol 2024).

5. θ/α Ratio:
   R_{θ/α} = P_θ / P_α
   Primary EEG marker for AD — elevated in MCI and AD vs healthy.

References:
  - Al-Nuaimi et al., 2021: 11 discriminative band-power ratios
  - Frontiers in Neuroscience, 2024: Spectral slowing review
  - Clinical Neurophysiology, 2024: PAF-MMSE correlation
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config as CFG


def extract_spectral_features(epoch_data, sfreq=None):
    """
    Extract spectral biomarkers from a single EEG epoch.

    Parameters
    ----------
    epoch_data : ndarray, shape (n_channels, n_times)
        Single epoch of EEG data in Volts.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    features : ndarray, shape (n_features,)
        Feature vector containing:
        - 5 × n_ch absolute band powers (δ, θ, α, β, γ per channel)
        - 5 × n_ch relative band powers
        - n_ch PAF values
        - n_ch θ/α ratios
        - 11 × n_ch Al-Nuaimi band ratios
    feature_names : list of str
        Names of each feature for interpretability.
    """
    if sfreq is None:
        sfreq = CFG.SFREQ

    n_ch, n_times = epoch_data.shape
    bands = CFG.BANDS  # {'delta': (1,4), 'theta': (4,8), ...}
    band_names = list(bands.keys())
    n_bands = len(band_names)

    # Compute PSD via Welch's method for all channels
    # nperseg = min(256, n_times) for good frequency resolution
    nperseg = min(256, n_times)
    freqs, psd_all = welch(epoch_data, fs=sfreq, nperseg=nperseg, axis=-1)
    # psd_all shape: (n_ch, n_freqs)

    # ── Absolute Band Power ──────────────────────────────────────────
    # P_band = ∫_{f_low}^{f_high} P(f) df  (Simpson's rule)
    abs_power = np.zeros((n_ch, n_bands))
    for bi, bname in enumerate(band_names):
        flo, fhi = bands[bname]
        idx = (freqs >= flo) & (freqs <= fhi)
        if idx.sum() < 2:
            abs_power[:, bi] = 0
        else:
            abs_power[:, bi] = simpson(psd_all[:, idx], x=freqs[idx], axis=-1)

    # ── Total Power (1-45 Hz) ────────────────────────────────────────
    total_idx = (freqs >= 1.0) & (freqs <= 45.0)
    total_power = simpson(psd_all[:, total_idx], x=freqs[total_idx], axis=-1)
    total_power = np.maximum(total_power, 1e-20)  # avoid division by zero

    # ── Relative Band Power ──────────────────────────────────────────
    # RBP_band = P_band / P_total
    rel_power = abs_power / total_power[:, np.newaxis]

    # ── Peak Alpha Frequency (PAF) — Center of Gravity ──────────────
    # PAF_cog = Σ(f × P(f)) / Σ(P(f))  for f ∈ [8, 13]
    # Center-of-gravity gives sub-bin frequency precision, which is
    # needed to detect AD-range PAF shifts (9→7 Hz). Argmax with
    # nperseg=256 gives only ~1 Hz resolution — too coarse.
    # (Klimesch 1999; Clinical Neurophysiology 2024)
    alpha_idx = (freqs >= 8.0) & (freqs <= 13.0)
    alpha_freqs = freqs[alpha_idx]
    paf = np.zeros(n_ch)
    for c in range(n_ch):
        alpha_power = psd_all[c, alpha_idx]
        total_alpha = alpha_power.sum()
        if alpha_idx.sum() > 0 and total_alpha > 0:
            paf[c] = np.sum(alpha_freqs * alpha_power) / total_alpha
        else:
            paf[c] = 10.0  # default healthy PAF

    # ── θ/α Ratio ────────────────────────────────────────────────────
    # R_{θ/α} = P_θ / P_α  (elevated in AD)
    theta_idx = band_names.index('theta')
    alpha_idx_band = band_names.index('alpha')
    theta_alpha_ratio = abs_power[:, theta_idx] / np.maximum(abs_power[:, alpha_idx_band], 1e-20)

    # ── Al-Nuaimi Band Ratios (11 discriminative ratios) ─────────────
    # Each ratio = P_band_a / P_band_b (per channel)
    band_ratios = np.zeros((n_ch, len(CFG.BAND_RATIOS)))
    for ri, (bname_a, bname_b) in enumerate(CFG.BAND_RATIOS):
        idx_a = band_names.index(bname_a)
        idx_b = band_names.index(bname_b)
        band_ratios[:, ri] = abs_power[:, idx_a] / np.maximum(abs_power[:, idx_b], 1e-20)

    # ── Assemble Feature Vector ──────────────────────────────────────
    features = np.concatenate([
        abs_power.flatten(),          # n_ch * 5
        rel_power.flatten(),          # n_ch * 5
        paf,                          # n_ch
        theta_alpha_ratio,            # n_ch
        band_ratios.flatten(),        # n_ch * 11
    ])

    return features.astype(np.float32)


def batch_extract_spectral(X_1d, sfreq=None):
    """
    Extract spectral features for all epochs.

    Parameters
    ----------
    X_1d : ndarray, shape (N, n_channels, n_times)
    sfreq : float

    Returns
    -------
    X_spectral : ndarray, shape (N, n_features)
    """
    if sfreq is None:
        sfreq = CFG.SFREQ

    N = X_1d.shape[0]
    # Get feature size from first epoch
    f0 = extract_spectral_features(X_1d[0], sfreq)
    n_feat = len(f0)

    X_spectral = np.zeros((N, n_feat), dtype=np.float32)
    X_spectral[0] = f0

    for i in range(1, N):
        X_spectral[i] = extract_spectral_features(X_1d[i], sfreq)

    # Replace NaN/Inf with 0
    np.nan_to_num(X_spectral, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  [spectral] Extracted {n_feat} features per epoch ({N} epochs)")
    return X_spectral


def get_spectral_feature_names(n_channels):
    """Return human-readable feature names for plotting/XAI."""
    bands = list(CFG.BANDS.keys())
    names = []

    # Absolute band powers
    for ch in range(n_channels):
        for b in bands:
            names.append(f"abs_{b}_ch{ch}")

    # Relative band powers
    for ch in range(n_channels):
        for b in bands:
            names.append(f"rbp_{b}_ch{ch}")

    # PAF
    for ch in range(n_channels):
        names.append(f"paf_ch{ch}")

    # θ/α ratio
    for ch in range(n_channels):
        names.append(f"theta_alpha_ratio_ch{ch}")

    # Al-Nuaimi ratios
    for ch in range(n_channels):
        for (ba, bb) in CFG.BAND_RATIOS:
            names.append(f"ratio_{ba}_{bb}_ch{ch}")

    return names
