"""
v3/data/tf_representations.py
═══════════════════════════════════════════════════════════════════════════
Unified Time-Frequency Representation (TFR) module.

Implements two TFR methods with a single API:
  1. CWT  — Continuous Wavelet Transform (Morlet, via MNE)
  2. STFT — Short-Time Fourier Transform (via scipy)

Both methods produce output of shape (N, C, n_freqs, n_times_out) so the
CViT branch of FusionNet can be used unchanged for fair comparison.

Literature basis:
  - PMC/NIH 2024–2025 systematic comparison: STFT achieved up to 98.8%
    accuracy in multi-channel fusion frameworks for AD classification.
  - JMIR, Frontiers 2025: CWT (Morlet) remains gold standard for
    multi-resolution analysis of non-stationary EEG signals.

Mathematical background:
  CWT:  W(a,b) = (1/√a) ∫ x(t) ψ*((t-b)/a) dt
        Uses Morlet wavelet ψ with adaptive n_cycles.

  STFT: S(τ,f) = ∫ x(t) w(t-τ) e^{-j2πft} dt
        Uses Hann window with fixed width (nperseg).
        Fixed time-frequency resolution (Heisenberg trade-off).
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import RectBivariateSpline


# ═════════════════════════════════════════════════════════════════════════
# CWT — Continuous Wavelet Transform (existing logic, refactored)
# ═════════════════════════════════════════════════════════════════════════

def _compute_cwt(X, sfreq, freqs, chunk_size=50, decim=4, n_jobs=1):
    """
    Compute Morlet wavelet scalograms using MNE.

    Parameters
    ----------
    X : ndarray, shape (N, C, T)
    sfreq : float
    freqs : ndarray — center frequencies to evaluate
    chunk_size : int — epochs per batch (memory management)
    decim : int — decimation factor for time axis
    n_jobs : int

    Returns
    -------
    X_tf : ndarray, shape (N, C, len(freqs), T//decim)
    """
    from mne.time_frequency import tfr_array_morlet

    n_epochs, n_channels, n_times = X.shape
    n_freqs = len(freqs)
    n_times_out = int(np.ceil(n_times / decim))

    n_cycles = freqs / 2.0
    n_cycles[n_cycles < 2] = 2

    X_tf = np.zeros((n_epochs, n_channels, n_freqs, n_times_out), dtype=np.float32)

    for start in range(0, n_epochs, chunk_size):
        end = min(start + chunk_size, n_epochs)
        power_chunk = tfr_array_morlet(
            X[start:end], sfreq=sfreq, freqs=freqs,
            n_cycles=n_cycles, output='power',
            n_jobs=n_jobs, use_fft=True, decim=decim
        )
        X_tf[start:end] = np.sqrt(power_chunk).astype(np.float32)

        if end == n_epochs or (start + chunk_size) % (chunk_size * 5) == 0:
            print(f"    CWT: {end}/{n_epochs} epochs")

    return X_tf


# ═════════════════════════════════════════════════════════════════════════
# STFT — Short-Time Fourier Transform
# ═════════════════════════════════════════════════════════════════════════

def _compute_stft(X, sfreq, freqs, chunk_size=50, decim=1, **kwargs):
    """
    Compute STFT spectrograms using scipy.

    Uses a Hann window with nperseg chosen to give ~1 Hz frequency
    resolution at the given sampling rate, with 75% overlap for smooth
    temporal evolution.

    The output is cropped to the target frequency range (matching the
    freqs array from CWT) and resized to match the CWT output shape
    for fair comparison.

    Parameters
    ----------
    X : ndarray, shape (N, C, T)
    sfreq : float
    freqs : ndarray — target frequency bins (used to determine output shape)
    chunk_size : int — epochs per batch
    decim : int — not used for STFT (overlap controls time resolution)

    Returns
    -------
    X_tf : ndarray, shape (N, C, n_target_freqs, n_target_times)
        Matches the shape that CWT would produce with the same freqs and decim=4.
    """
    n_epochs, n_channels, n_times = X.shape

    # STFT parameters
    # nperseg controls frequency resolution: df = sfreq / nperseg
    # With nperseg=sfreq (256 for 256 Hz), we get 1 Hz resolution
    # But that's very fine — use sfreq//4 = 64 for 4 Hz resolution
    # (similar to CWT at low frequencies)
    nperseg = min(int(sfreq) // 4, n_times)
    noverlap = int(nperseg * 0.75)  # 75% overlap for smooth time axis
    window = 'hann'

    # Determine target output shape (matching CWT with decim=4)
    n_target_freqs = len(freqs)
    n_target_times = int(np.ceil(n_times / 4))  # match CWT decim=4

    X_tf = np.zeros((n_epochs, n_channels, n_target_freqs, n_target_times),
                    dtype=np.float32)

    # Frequency and time range of interest
    f_min, f_max = freqs[0], freqs[-1]

    for start in range(0, n_epochs, chunk_size):
        end = min(start + chunk_size, n_epochs)

        for i in range(start, end):
            for ch in range(n_channels):
                # Compute STFT
                f_stft, t_stft, Zxx = scipy_signal.stft(
                    X[i, ch, :], fs=sfreq,
                    window=window, nperseg=nperseg,
                    noverlap=noverlap, return_onesided=True
                )

                # Take magnitude (amplitude spectrum)
                magnitude = np.abs(Zxx)

                # Crop to frequency range of interest
                freq_mask = (f_stft >= f_min) & (f_stft <= f_max)
                f_cropped = f_stft[freq_mask]
                mag_cropped = magnitude[freq_mask, :]

                # Resize to match target shape using bilinear interpolation
                if mag_cropped.shape[0] > 1 and mag_cropped.shape[1] > 1:
                    interp = RectBivariateSpline(
                        np.arange(mag_cropped.shape[0]),
                        np.arange(mag_cropped.shape[1]),
                        mag_cropped, kx=1, ky=1
                    )
                    new_f = np.linspace(0, mag_cropped.shape[0] - 1, n_target_freqs)
                    new_t = np.linspace(0, mag_cropped.shape[1] - 1, n_target_times)
                    X_tf[i, ch] = interp(new_f, new_t).astype(np.float32)
                else:
                    # Edge case: very short signals
                    X_tf[i, ch] = 0.0

        if end == n_epochs or (start + chunk_size) % (chunk_size * 5) == 0:
            print(f"    STFT: {end}/{n_epochs} epochs")

    return X_tf


# ═════════════════════════════════════════════════════════════════════════
# CQT — Constant-Q Transform
# ═════════════════════════════════════════════════════════════════════════

def _compute_cqt(X, sfreq, freqs, chunk_size=50, decim=1, **kwargs):
    import librosa
    
    n_epochs, n_channels, n_times = X.shape
    n_target_freqs = len(freqs)
    n_target_times = int(np.ceil(n_times / 4))
    
    X_tf = np.zeros((n_epochs, n_channels, n_target_freqs, n_target_times), dtype=np.float32)
    
    fmin = freqs[0]
    n_bins = len(freqs)
    
    for start in range(0, n_epochs, chunk_size):
        end = min(start + chunk_size, n_epochs)
        
        for i in range(start, end):
            for ch in range(n_channels):
                sig = X[i, ch, :]
                C = librosa.cqt(sig, sr=sfreq, fmin=fmin, n_bins=n_bins, bins_per_octave=8)
                mag = np.abs(C)
                
                if mag.shape[0] > 1 and mag.shape[1] > 1:
                    interp = RectBivariateSpline(
                        np.arange(mag.shape[0]),
                        np.arange(mag.shape[1]),
                        mag, kx=1, ky=1
                    )
                    new_f = np.linspace(0, mag.shape[0] - 1, n_target_freqs)
                    new_t = np.linspace(0, mag.shape[1] - 1, n_target_times)
                    X_tf[i, ch] = interp(new_f, new_t).astype(np.float32)
                else:
                    X_tf[i, ch] = 0.0
                    
        if end == n_epochs or (start + chunk_size) % (chunk_size * 5) == 0:
            print(f"    CQT: {end}/{n_epochs} epochs")
            
    return X_tf


# ═════════════════════════════════════════════════════════════════════════
# WVD — Pseudo Wigner-Ville Distribution
# ═════════════════════════════════════════════════════════════════════════

def _compute_wvd(X, sfreq, freqs, chunk_size=50, decim=1, **kwargs):
    from scipy.signal import hilbert
    from scipy.fft import fft, fftshift
    
    n_epochs, n_channels, n_times = X.shape
    n_target_freqs = len(freqs)
    n_target_times = int(np.ceil(n_times / 4))
    
    X_tf = np.zeros((n_epochs, n_channels, n_target_freqs, n_target_times), dtype=np.float32)
    
    f_min, f_max = freqs[0], freqs[-1]
    
    for start in range(0, n_epochs, chunk_size):
        end = min(start + chunk_size, n_epochs)
        
        for i in range(start, end):
            for ch in range(n_channels):
                sig = X[i, ch, :]
                z = hilbert(sig)
                N = len(z)
                
                t_idx = np.linspace(0, N-1, n_target_times, dtype=int)
                max_lag = min(N // 4, 128) # Limit max lag to speed up computation
                mag = np.zeros((max_lag * 2, len(t_idx)))
                
                for t_out, t_in in enumerate(t_idx):
                    tau_max = min(t_in, N - 1 - t_in, max_lag - 1)
                    if tau_max < 1:
                        continue
                        
                    tau = np.arange(-tau_max, tau_max + 1)
                    R = z[t_in + tau] * np.conj(z[t_in - tau])
                    
                    W = np.abs(fftshift(fft(R, n=max_lag*2)))
                    mag[:, t_out] = W
                
                f_wvd = np.linspace(0, sfreq/2, max_lag * 2)
                freq_mask = (f_wvd >= f_min) & (f_wvd <= f_max)
                f_cropped = f_wvd[freq_mask]
                mag_cropped = mag[freq_mask, :]
                
                if mag_cropped.shape[0] > 1 and mag_cropped.shape[1] > 1:
                    interp = RectBivariateSpline(
                        np.arange(mag_cropped.shape[0]),
                        np.arange(mag_cropped.shape[1]),
                        mag_cropped, kx=1, ky=1
                    )
                    new_f = np.linspace(0, mag_cropped.shape[0] - 1, n_target_freqs)
                    new_t = np.linspace(0, mag_cropped.shape[1] - 1, n_target_times)
                    X_tf[i, ch] = interp(new_f, new_t).astype(np.float32)
                else:
                    X_tf[i, ch] = 0.0

        if end == n_epochs or (start + chunk_size) % (chunk_size * 5) == 0:
            print(f"    WVD: {end}/{n_epochs} epochs")
            
    return X_tf


# ═════════════════════════════════════════════════════════════════════════
# UNIFIED API
# ═════════════════════════════════════════════════════════════════════════

SUPPORTED_METHODS = ['cwt', 'stft', 'cqt', 'wvd']

def batch_compute_tf(X, sfreq, method='cwt', freqs=None, chunk_size=50,
                     decim=4, n_jobs=1):
    """
    Compute time-frequency representations using the specified method.

    Both methods produce the same output shape (N, C, n_freqs, n_times_out)
    so downstream models (CViT) work unchanged.

    Parameters
    ----------
    X : ndarray, shape (N, C, T)
        Raw EEG epochs (standardized).
    sfreq : float
        Sampling frequency in Hz.
    method : str
        'cwt' — Continuous Wavelet Transform (Morlet via MNE)
        'stft' — Short-Time Fourier Transform (Hann window via scipy)
        'cqt' — Constant-Q Transform (via librosa)
        'wvd' — Pseudo Wigner-Ville Distribution (via scipy)
    freqs : ndarray or None
        Target frequency bins. Default: np.arange(2, 45, 2) → 22 bins.
    chunk_size : int
        Number of epochs per processing chunk.
    decim : int
        Decimation factor (CWT only; STFT matches this output size).
    n_jobs : int
        Parallel jobs (CWT only).

    Returns
    -------
    X_tf : ndarray, shape (N, C, n_freqs, n_times_out)
        Time-frequency representation (amplitude).

    Raises
    ------
    ValueError
        If method is not in SUPPORTED_METHODS.
    """
    if freqs is None:
        freqs = np.arange(2, 45, 2)

    method = method.lower().strip()
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown TFR method '{method}'. "
            f"Supported: {SUPPORTED_METHODS}"
        )

    print(f"\n  Computing TFR using method: {method.upper()}")
    print(f"    Epochs: {X.shape[0]} | Channels: {X.shape[1]} | "
          f"Samples: {X.shape[2]} | Freqs: {len(freqs)}")

    if method == 'cwt':
        X_tf = _compute_cwt(X, sfreq, freqs, chunk_size, decim, n_jobs)
    elif method == 'stft':
        X_tf = _compute_stft(X, sfreq, freqs, chunk_size, decim)
    elif method == 'cqt':
        X_tf = _compute_cqt(X, sfreq, freqs, chunk_size, decim)
    elif method == 'wvd':
        X_tf = _compute_wvd(X, sfreq, freqs, chunk_size, decim)

    print(f"    Output shape: {X_tf.shape}")

    # NaN/Inf cleanup
    if np.isnan(X_tf).any() or np.isinf(X_tf).any():
        print(f"    WARNING: NaN/Inf detected in {method.upper()} output — fixing")
        np.nan_to_num(X_tf, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return X_tf
