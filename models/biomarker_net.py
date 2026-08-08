"""
v3/models/biomarker_net.py
═══════════════════════════════════════════════════════════════════════════
BiomarkerNet — Path D of the NeuroFusion-AD architecture.

Encodes expert-designed domain features (spectral, complexity, connectivity)
into a learned embedding via a small fully-connected network.

Architecture:
  Input(n_features) → Linear(128) → BN → GELU → Dropout(0.3)
                    → Linear(64)  → BN → GELU → Dropout(0.2)
                    → Linear(48)  → embed

Rationale:
  Deep learning models (EEGNet, CViT, GNN) learn representations from raw
  data, but may miss clinically validated biomarkers (PAF, θ/α ratio,
  SampEn). This branch injects domain knowledge into the fusion, giving
  the model access to features that took decades of neuroscience research
  to identify. It's particularly useful for small cohorts where end-to-end
  learning has limited data to discover these patterns.
═══════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn


class BiomarkerNet(nn.Module):
    """
    Fully-connected network for domain-specific EEG biomarker features.

    Parameters
    ----------
    n_features : int
        Number of input biomarker features.
    embed_dim : int
        Output embedding dimension (must match other branches for fusion).
    dropout : float
        Dropout rate for regularization.
    """

    def __init__(self, n_features, embed_dim=48, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1: Reduce from high-dim feature space
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            # Layer 2: Intermediate compression
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.67),  # lighter dropout in deeper layers

            # Layer 3: Final embedding
            nn.Linear(64, embed_dim),
        )

        # Initialize weights with Kaiming for GELU
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (B, n_features)

        Returns
        -------
        embed : Tensor, shape (B, embed_dim)
        """
        return self.net(x)
