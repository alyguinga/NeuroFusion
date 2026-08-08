"""
v3/features/complexity.py
═══════════════════════════════════════════════════════════════════════════
Non-linear complexity biomarkers — GPU-accelerated where possible.

GPU acceleration targets Sample Entropy (the main bottleneck):
  - Old: Python loop over ~500K pairs per channel → ~22s for 10 epochs
  - New: Vectorized pairwise distance on GPU → ~0.5s for 10 epochs

Metrics: SampEn, PE, HFD, LZC, Hjorth (Activity, Mobility, Complexity)

Band-filtered SampEn (added for biomarker enhancement):
  AD-specific entropy changes are strongest in individual frequency bands.
  Broadband SampEn dilutes the signal. Computing per-band SampEn for
  theta (4-8 Hz), alpha (8-13 Hz), and beta (13-30 Hz) gives BiomarkerNet
  access to complexity information the DL branches can't easily learn.
  (Abásolo et al. 2006; Hornero et al. 2009)
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import math
import sys, os
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config as CFG

# Try importing torch for GPU acceleration
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═════════════════════════════════════════════════════════════════════════
# 1. SAMPLE ENTROPY — GPU-ACCELERATED
# ═════════════════════════════════════════════════════════════════════════
def _sampen_all_channels_gpu(epoch_data, m=2, r_factor=0.2, device='cuda'):
    """
    GPU-vectorized SampEn for ALL channels of one epoch simultaneously.

    SampEn(m, r, N) = -ln(A / B)
    where B = template matches at dim m, A = matches at dim m+1.

    Instead of Python loops, uses torch broadcasting for pairwise
    Chebyshev distance computation on GPU.

    Memory: ~200MB per epoch (16 channels, 1024 samples). Fine for Kaggle GPU.
    """
    n_ch, N = epoch_data.shape
    x = torch.tensor(epoch_data, dtype=torch.float32, device=device)
    r = r_factor * x.std(dim=1)  # (n_ch,) — per-channel threshold
    results = np.zeros(n_ch, dtype=np.float32)

    counts = {}
    for dim in [m, m + 1]:
        # Create templates: (n_ch, n_templates, dim) via unfold
        # unfold returns N - dim + 1 windows
        templates = x.unfold(dimension=1, size=dim, step=1)
        n_tmpl = templates.shape[1]  # actual count from unfold

        if n_tmpl < 2:
            return results

        # Pairwise Chebyshev distance: max|t_i - t_j| over embedding dims
        # (n_ch, n_tmpl, 1, dim) - (n_ch, 1, n_tmpl, dim)
        diffs = (templates.unsqueeze(2) - templates.unsqueeze(1)).abs()
        distances = diffs.max(dim=-1).values  # (n_ch, n_tmpl, n_tmpl)

        # Upper triangle mask (exclude self-matches)
        mask = torch.triu(torch.ones(n_tmpl, n_tmpl, dtype=torch.bool, device=device), diagonal=1)

        # Count matches per channel where distance < r
        masked = distances[:, mask]  # (n_ch, n_pairs)
        counts[dim] = (masked < r.unsqueeze(1)).sum(dim=1).cpu().float().numpy()

        del diffs, distances, templates  # free GPU memory

    B_c, A_c = counts[m], counts[m + 1]
    valid = (B_c > 0) & (A_c > 0)
    results[valid] = -np.log(A_c[valid] / B_c[valid])
    return results


_TRIU_CACHE = {}

def _sampen_all_channels_cpu(epoch_data, m=2, r_factor=0.2):
    """
    Numpy-vectorized SampEn fallback (no GPU). Still ~10x faster than
    the pure Python loop version thanks to broadcasting.
    """
    n_ch, N = epoch_data.shape
    results = np.zeros(n_ch, dtype=np.float32)

    for c in range(n_ch):
        x = epoch_data[c]
        r = r_factor * np.std(x)
        if r == 0 or N < m + 2:
            continue

        # Precompute the diff matrices for all k once at max length N - m
        L = N - m
        diffs = [np.abs(x[k:L+k, np.newaxis] - x[k:L+k]) for k in range(m + 1)]

        counts = {}
        for dim in [m, m + 1]:
            n_tmpl = N - dim
            # Slice precomputed diffs to n_tmpl
            dist = diffs[0][:n_tmpl, :n_tmpl]
            for k in range(1, dim):
                dist = np.maximum(dist, diffs[k][:n_tmpl, :n_tmpl])

            if n_tmpl not in _TRIU_CACHE:
                _TRIU_CACHE[n_tmpl] = np.triu_indices(n_tmpl, k=1)
            triu = _TRIU_CACHE[n_tmpl]
            counts[dim] = np.sum(dist[triu] < r)

        B, A = counts[m], counts[m + 1]
        if B > 0 and A > 0:
            results[c] = -np.log(A / B)

    return results


# ═════════════════════════════════════════════════════════════════════════
# 2. PERMUTATION ENTROPY (numpy-vectorized, already fast)
# ═════════════════════════════════════════════════════════════════════════
def permutation_entropy(x, m=3, delay=1):
    """PE(m,τ) = -Σ p(π) log₂(p(π)) / log₂(m!)"""
    N = len(x)
    n_pat = N - (m - 1) * delay
    if n_pat <= 0:
        return 0.0
    indices = np.arange(m) * delay
    patterns = np.array([x[i + indices] for i in range(n_pat)])
    sorted_idx = np.argsort(patterns, axis=1)
    base = np.array([m**i for i in range(m)])
    hashes = np.sum(sorted_idx * base, axis=1)
    _, counts = np.unique(hashes, return_counts=True)
    p = counts / len(hashes)
    pe = -np.sum(p * np.log2(p + 1e-12))
    return pe / np.log2(math.factorial(m))


# ═════════════════════════════════════════════════════════════════════════
# 3. HIGUCHI FRACTAL DIMENSION
# ═════════════════════════════════════════════════════════════════════════
def higuchi_fd(x, kmax=10):
    """HFD = slope of log(L(k)) vs log(1/k)."""
    N = len(x)
    kmax = min(kmax, N // 4)
    if kmax < 2:
        return 1.0
    x = np.asarray(x, dtype=np.float64)
    L = np.zeros(kmax)
    for k in range(1, kmax + 1):
        Lk = np.zeros(k)
        for m_idx in range(k):
            idxs = np.arange(m_idx, N, k)
            if len(idxs) < 2:
                continue
            Lmk = np.sum(np.abs(np.diff(x[idxs])))
            n_int = len(idxs) - 1
            if n_int > 0:
                Lk[m_idx] = Lmk * (N - 1) / (n_int * k * k)
        L[k - 1] = np.mean(Lk) if np.any(Lk > 0) else 1e-10
    valid = L > 0
    if valid.sum() < 2:
        return 1.0
    k_vals = np.arange(1, kmax + 1, dtype=np.float64)
    coeffs = np.polyfit(np.log(k_vals[valid]), np.log(L[valid]), 1)
    return max(-coeffs[0], 0.0)


# ═════════════════════════════════════════════════════════════════════════
# 4. LEMPEL-ZIV COMPLEXITY
# ═════════════════════════════════════════════════════════════════════════
def lempel_ziv_complexity(x):
    """LZC = c(n) / (n / log₂(n)), binarized at median."""
    N = len(x)
    if N < 4:
        return 0.0
    median_val = np.median(x)
    binary = ''.join(['1' if v >= median_val else '0' for v in x])
    s, n = binary, len(binary)
    i, k, l, c = 0, 1, 1, 1
    while True:
        if k + l > n:
            break
        if s[i:i+l] == s[k:k+l]:
            k += 1; l = 1
        else:
            i += 1
            if i == k:
                c += 1; k += l; i = 0; l = 1
            else:
                l += 1
    b = n / np.log2(n) if n > 1 else 1.0
    return c / b


# ═════════════════════════════════════════════════════════════════════════
# 5. HJORTH PARAMETERS
# ═════════════════════════════════════════════════════════════════════════
def hjorth_parameters(x):
    """Activity=var(x), Mobility=sqrt(var(dx)/var(x)), Complexity=mob(dx)/mob(x)."""
    var_x = np.var(x)
    if var_x < 1e-20:
        return 0.0, 0.0, 0.0
    dx = np.diff(x)
    var_dx = np.var(dx)
    ddx = np.diff(dx)
    var_ddx = np.var(ddx)
    activity = var_x
    mobility = np.sqrt(var_dx / var_x)
    mob_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0 else 0.0
    complexity = mob_dx / mobility if mobility > 0 else 0.0
    return activity, mobility, complexity


# ═════════════════════════════════════════════════════════════════════════
# 6. BAND-FILTERED SAMPLE ENTROPY
# ═════════════════════════════════════════════════════════════════════════
_BAND_SAMPEN_BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
}


def _bandpass_filter(data, low, high, sfreq=256, order=4):
    """
    Zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data : ndarray (n_ch, n_times)
    low, high : float — cutoff frequencies
    sfreq : float
    order : int

    Returns
    -------
    filtered : ndarray (n_ch, n_times)
    """
    nyq = sfreq / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def _band_sampen(epoch_data, device=None, sfreq=256):
    """
    Compute SampEn on band-filtered signals (theta, alpha, beta).

    AD-specific entropy reductions are strongest in individual bands,
    not broadband. (Abásolo et al. 2006; Hornero et al. 2009)

    Parameters
    ----------
    epoch_data : ndarray (n_ch, n_times)
    device : str or None — 'cuda' for GPU, 'cpu' for CPU
    sfreq : float

    Returns
    -------
    band_sampen : ndarray (n_ch, 3) — SampEn for [theta, alpha, beta]
    """
    n_ch = epoch_data.shape[0]
    band_sampen = np.zeros((n_ch, 3), dtype=np.float32)

    for bi, (bname, (flo, fhi)) in enumerate(_BAND_SAMPEN_BANDS.items()):
        filtered = _bandpass_filter(epoch_data, flo, fhi, sfreq)
        if device is not None and device != 'cpu' and HAS_TORCH:
            band_sampen[:, bi] = _sampen_all_channels_gpu(
                filtered, m=2, r_factor=0.2, device=device
            )
        else:
            band_sampen[:, bi] = _sampen_all_channels_cpu(
                filtered, m=2, r_factor=0.2
            )

    return band_sampen


# ═════════════════════════════════════════════════════════════════════════
# BATCH EXTRACTOR — GPU SampEn + CPU rest
# ═════════════════════════════════════════════════════════════════════════
# Features per channel: 8 original + 3 band-filtered SampEn = 11
_FEATS_PER_CH = 11


def extract_complexity_features(epoch_data, device=None, sfreq=256):
    """Extract all complexity features for a single epoch (n_ch, n_times)."""
    n_ch = epoch_data.shape[0]
    feats = np.zeros(n_ch * _FEATS_PER_CH, dtype=np.float32)

    # Broadband SampEn — GPU-accelerated (all channels at once)
    if device is not None and device != 'cpu' and HAS_TORCH:
        sampen_vals = _sampen_all_channels_gpu(epoch_data, m=2, r_factor=0.2, device=device)
    else:
        sampen_vals = _sampen_all_channels_cpu(epoch_data, m=2, r_factor=0.2)

    # Band-filtered SampEn (theta, alpha, beta)
    band_sampen = _band_sampen(epoch_data, device=device, sfreq=sfreq)

    for c in range(n_ch):
        offset = c * _FEATS_PER_CH
        sig = epoch_data[c]
        feats[offset + 0] = sampen_vals[c]
        feats[offset + 1] = permutation_entropy(sig, m=3, delay=1)
        feats[offset + 2] = higuchi_fd(sig, kmax=CFG.HFD_KMAX)
        feats[offset + 3] = lempel_ziv_complexity(sig)
        act, mob, comp = hjorth_parameters(sig)
        feats[offset + 4] = act
        feats[offset + 5] = mob
        feats[offset + 6] = comp
        feats[offset + 7] = feats[offset + 1] * (2.0 - feats[offset + 2])
        # Band-filtered SampEn (theta, alpha, beta)
        feats[offset + 8] = band_sampen[c, 0]   # sampen_theta
        feats[offset + 9] = band_sampen[c, 1]   # sampen_alpha
        feats[offset + 10] = band_sampen[c, 2]  # sampen_beta

    return feats


def batch_extract_complexity(X_1d, sfreq=None):
    """
    Extract complexity features for all epochs.
    Uses GPU for SampEn if available, CPU for the rest.
    """
    if sfreq is None:
        sfreq = CFG.SFREQ

    N = X_1d.shape[0]
    n_ch = X_1d.shape[1]
    n_feat = n_ch * _FEATS_PER_CH

    # Detect GPU
    device = None
    if HAS_TORCH and torch.cuda.is_available():
        device = 'cuda'
        print(f"    [complexity] Using GPU acceleration for SampEn")
    else:
        print(f"    [complexity] GPU not available, using numpy vectorized (CPU)")
        device = 'cpu'

    X_complexity = np.zeros((N, n_feat), dtype=np.float32)

    for i in range(N):
        X_complexity[i] = extract_complexity_features(X_1d[i], device=device, sfreq=sfreq)
        if (i + 1) % 100 == 0:
            print(f"    [complexity] Processed {i+1}/{N} epochs")

    # Free GPU cache
    if device == 'cuda' and HAS_TORCH:
        torch.cuda.empty_cache()

    np.nan_to_num(X_complexity, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  [complexity] Extracted {n_feat} features per epoch ({N} epochs)")
    return X_complexity


def get_complexity_feature_names(n_channels):
    """Return human-readable feature names."""
    names = []
    metrics = ['sampen', 'pe', 'hfd', 'lzc', 'hjorth_act', 'hjorth_mob', 'hjorth_comp',
               'composite_cx', 'sampen_theta', 'sampen_alpha', 'sampen_beta']
    for ch in range(n_channels):
        for m in metrics:
            names.append(f"{m}_ch{ch}")
    return names
