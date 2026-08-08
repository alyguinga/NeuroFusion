"""
v3/data/augment_v3.py
═══════════════════════════════════════════════════════════════════════════
Enhanced data augmentation suite for EEG-AD deep learning.

All transforms are applied ONLINE during training only. They increase
effective dataset size and prevent overfitting on small cohorts (N~60).

Transforms (applied to raw 1D signal):
───────────────────────────────────────
1. Gaussian Noise — x' = x + ε, ε ~ N(0, σ²)
   Simulates recording noise variability across sessions/systems.

2. Channel Dropout — randomly zero entire channels with probability p.
   Forces the model to use distributed spatial patterns, not single
   "loud" channels. Regularization effect similar to spatial dropout.

3. Amplitude Scaling — x' = α × x, α ~ Uniform(0.85, 1.15)
   Invariance to amplitude differences across recording systems
   and subject head impedances.

4. Temporal Masking — zero out random contiguous time segments.
   Prevents over-reliance on specific temporal positions.
   Similar to SpecAugment (Park et al., 2019) for audio.

References:
  - Frontiers 2024: EEG augmentation for AD
  - Park et al., 2019: SpecAugment (adapted for EEG)
  - MDPI 2024: Channel dropout for BCI robustness
═══════════════════════════════════════════════════════════════════════════
"""

import torch


def augment_batch(x_1d, x_scalo=None, x_adj=None, x_bio=None,
                  noise_std=0.08, chan_drop_prob=0.1,
                  amp_scale_range=(0.85, 1.15),
                  temporal_mask_ratio=0.05):
    """
    Apply online augmentation to a training batch.

    Parameters
    ----------
    x_1d : Tensor (B, 1, C, T) — raw EEG signal
    x_scalo : Tensor (B, C, F, T) — scalograms (not augmented)
    x_adj : Tensor (B, C, C) — adjacency (not augmented)
    x_bio : Tensor (B, n_bio) — biomarker features (not augmented)
    noise_std : float — Gaussian noise standard deviation
    chan_drop_prob : float — probability of dropping each channel
    amp_scale_range : tuple — (min, max) for amplitude scaling
    temporal_mask_ratio : float — fraction of time points to mask

    Returns
    -------
    Augmented tensors (same shapes as input).
    Only x_1d is modified; other modalities are returned unchanged.
    """
    B = x_1d.shape[0]
    device = x_1d.device

    # ── 1. Gaussian Noise ────────────────────────────────────────────
    # x' = x + ε, ε ~ N(0, σ²)
    # Only applied to raw 1D signal — derived features (scalograms,
    # adjacency, biomarkers) should not be randomly perturbed.
    if noise_std > 0:
        x_1d = x_1d + noise_std * torch.randn_like(x_1d)

    # ── 2. Channel Dropout ───────────────────────────────────────────
    # Randomly zero entire channels with probability p
    if chan_drop_prob > 0:
        # Mask shape: (B, 1, C, 1) — broadcast across time
        mask = (torch.rand(B, 1, x_1d.shape[2], 1, device=device) > chan_drop_prob).float()
        x_1d = x_1d * mask

    # ── 3. Amplitude Scaling ─────────────────────────────────────────
    # x' = α × x, α ~ Uniform(low, high) per sample
    lo, hi = amp_scale_range
    scale = torch.empty(B, 1, 1, 1, device=device).uniform_(lo, hi)
    x_1d = x_1d * scale

    # ── 4. Temporal Masking ──────────────────────────────────────────
    # Zero out contiguous time segments (SpecAugment-style)
    if temporal_mask_ratio > 0:
        T = x_1d.shape[-1]
        mask_len = max(1, int(T * temporal_mask_ratio))
        for b in range(B):
            start = torch.randint(0, T - mask_len, (1,)).item()
            x_1d[b, :, :, start:start + mask_len] = 0

    return x_1d, x_scalo, x_adj, x_bio
