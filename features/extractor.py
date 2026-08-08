"""
v3/features/extractor_v3.py
═══════════════════════════════════════════════════════════════════════════
Main feature extractor orchestrator for V3 NeuroFusion-AD pipeline.

Combines three feature domains:
  1. Spectral  — PSD, band power, PAF, θ/α ratio, band ratios (linear freq. domain)
  2. Complexity — SampEn, PE, HFD, LZC, Hjorth (non-linear time domain)
  3. Connectivity — PLI, graph metrics (inter-channel relationships)

All features are z-score normalized across epochs.

Architecture Context:
  These features feed into BiomarkerNet (Path D of the 4-branch FusionNet),
  encoding domain knowledge that deep learning branches may not capture
  from raw signals alone. This is particularly valuable for AD detection
  because spectral slowing and complexity reduction are well-validated
  clinical biomarkers.
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import time
import os
from .spectral import batch_extract_spectral, get_spectral_feature_names
from .complexity import batch_extract_complexity, get_complexity_feature_names
from .connectivity_features import batch_extract_connectivity, get_connectivity_feature_names


def extract_all_biomarkers(X_1d, sfreq=None, verbose=True, normalize=True, mean=None, std=None, save_scaler_to=None):
    """
    Extract all biomarker features from raw EEG epochs.

    Parameters
    ----------
    X_1d : ndarray, shape (N, n_channels, n_times)
        Raw EEG epoch data (standardized).
    sfreq : float
        Sampling frequency.
    verbose : bool
    normalize : bool
    mean : ndarray or None
    std : ndarray or None
    save_scaler_to : str or None

    Returns
    -------
    X_bio : ndarray, shape (N, n_total_features)
        Combined, z-score normalized biomarker features.
    X_pli : ndarray, shape (N, n_ch, n_ch)
        PLI connectivity matrices for GNN input.
    feature_names : list of str
        Names of all features for XAI/visualization.
    """
    N, n_ch, n_times = X_1d.shape

    if verbose:
        print("\n  Extracting biomarker features:")
        print(f"    Epochs: {N} | Channels: {n_ch} | Samples: {n_times}")

    # ── 1. Spectral Features ─────────────────────────────────────────
    t0 = time.time()
    X_spectral = batch_extract_spectral(X_1d, sfreq)
    if verbose:
        print(f"    Spectral: {X_spectral.shape[1]} features ({time.time()-t0:.1f}s)")

    # ── 2. Complexity Features ───────────────────────────────────────
    t0 = time.time()
    X_complexity = batch_extract_complexity(X_1d)
    if verbose:
        print(f"    Complexity: {X_complexity.shape[1]} features ({time.time()-t0:.1f}s)")

    # ── 3. Connectivity Features + PLI Matrices ──────────────────────
    t0 = time.time()
    X_conn, X_pli = batch_extract_connectivity(X_1d)
    if verbose:
        print(f"    Connectivity: {X_conn.shape[1]} features ({time.time()-t0:.1f}s)")

    # ── Combine All Features ─────────────────────────────────────────
    X_bio = np.concatenate([X_spectral, X_complexity, X_conn], axis=1)

    # ── Z-Score Normalization ────────────────────────────────────────
    if normalize:
        if mean is None:
            mean = X_bio.mean(axis=0, keepdims=True)
            if save_scaler_to is not None:
                os.makedirs(save_scaler_to, exist_ok=True)
                np.save(os.path.join(save_scaler_to, "X_bio_mean.npy"), mean)
        if std is None:
            std = X_bio.std(axis=0, keepdims=True) + 1e-8
            if save_scaler_to is not None:
                np.save(os.path.join(save_scaler_to, "X_bio_std.npy"), std)
        X_bio = (X_bio - mean) / std

    # Final NaN cleanup
    np.nan_to_num(X_bio, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Feature Names ────────────────────────────────────────────────
    feature_names = (
        get_spectral_feature_names(n_ch) +
        get_complexity_feature_names(n_ch) +
        get_connectivity_feature_names(n_ch)
    )

    if verbose:
        print(f"\n  Total biomarker features: {X_bio.shape[1]}")
        print(f"  PLI matrices: {X_pli.shape}")

    return X_bio.astype(np.float32), X_pli.astype(np.float32), feature_names
