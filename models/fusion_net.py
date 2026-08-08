"""
v3/models/fusion_net.py
═══════════════════════════════════════════════════════════════════════════
NeuroFusion-AD — 4-Branch Multimodal Hybrid Model with Cross-Path Attention.

Architecture Overview:
  Path A: EEGNet        → embed_eegnet (temporal patterns from raw 1D)
  Path B: CViT          → embed_cvit   (time-frequency from CWT scalograms)
  Path C: GNN           → embed_gnn    (spatial topology from connectivity)
  Path D: BiomarkerNet  → embed_bio    (domain-knowledge features)

Fusion Strategy — Cross-Path Multi-Head Self-Attention:
  Instead of simple concatenation, we stack the 4 branch embeddings into
  a sequence of 4 tokens (each of dim embed_dim) and apply multi-head
  self-attention. This allows each branch to attend to and be modulated
  by every other branch.

  Cross-attention formula:
    Q, K, V = linear projections of stacked embeddings
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V

  After attention: mean-pool over the 4 tokens → classifier head.

  This is superior to concatenation because:
  1. It learns which branches are most informative per sample
  2. Allows cross-modal interactions (e.g., spectral slowing confirmed
     by reduced GNN connectivity = higher confidence)
  3. Scales naturally if more branches are added

References:
  - JMIR, Frontiers 2025: CViT for EEG
  - IEEE 2024-2025: GNN for brain connectivity
  - Frontiers 2025: Multi-path fusion with attention
═══════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn

from .eegnet import EEGNet

from .cvit import CViT
from .gnn import GNN
from .biomarker_net import BiomarkerNet


class CrossPathAttention(nn.Module):
    """
    Multi-head self-attention across branch embeddings.

    Takes 4 branch embeddings, treats them as a length-4 sequence,
    and applies transformer-style self-attention.

    Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
    """

    def __init__(self, embed_dim, n_heads=2, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        """
        Parameters
        ----------
        embeddings : Tensor, shape (B, n_branches, embed_dim)

        Returns
        -------
        fused : Tensor, shape (B, embed_dim)
            Mean-pooled attention output.
        attn_weights : Tensor, shape (B, n_branches, n_branches)
            Attention weights for XAI visualization.
        """
        # Self-attention across branches
        attn_out, attn_weights = self.attention(
            embeddings, embeddings, embeddings
        )

        # Residual + LayerNorm
        attn_out = self.norm(attn_out + embeddings)
        attn_out = self.dropout(attn_out)

        # Mean-pool across branches → single vector
        fused = attn_out.mean(dim=1)  # (B, embed_dim)

        return fused, attn_weights


class FusionNet(nn.Module):
    """
    NeuroFusion-AD: 4-branch multimodal hybrid model.

    Parameters
    ----------
    in_channels : int — number of EEG channels
    n_classes : int — number of output classes
    n_bio_features : int — number of biomarker features (from extractor_v3)
    n_freqs : int — frequency bins in scalogram
    n_times_scalo : int — time bins in scalogram
    n_times_1d : int — time samples in raw signal
    embed_dim : int — unified embedding dim for all branches
    dropout : float — classifier dropout
    """

    def __init__(self, in_channels=19, n_classes=2, n_bio_features=0,
                 dropout=0.4, n_freqs=22, n_times_scalo=256, n_times_1d=1024,
                 embed_dim=48, disabled_branches=None, use_gnn=False):
        super().__init__()

        self.has_biomarker = n_bio_features > 0
        self.embed_dim = embed_dim
        # Branches to zero-mask during forward (for ablation studies)
        # Valid names: 'eegnet', 'cvit', 'gnn', 'biomarker'
        self.disabled_branches = set(disabled_branches or [])

        # ── Path A: EEGNet (Temporal) ────────────────────────────────
        self.eegnet = EEGNet(nb_classes=n_classes, Chans=in_channels,
                             Samples=n_times_1d, dropoutRate=0.5)
        eegnet_out_dim = self.eegnet.flatten_size
        # Project EEGNet output to unified embed_dim
        self.eegnet_proj = nn.Linear(eegnet_out_dim, embed_dim)

        # ── Path B: CViT (Time-Frequency) ───────────────────────────
        self.cvit = CViT(in_channels=in_channels,
                         img_size=(n_freqs, n_times_scalo),
                         embed_dim=embed_dim, dropout=0.4)

        # ── Path C: GNN (Spatial/Connectivity) ──────────────────────
        self.use_gnn = use_gnn
        if self.use_gnn and 'gnn' not in self.disabled_branches:
            self.gnn = GNN(num_nodes=in_channels, embed_dim=32)
            self.gnn_proj = nn.Linear(32, embed_dim)

        # ── Path D: BiomarkerNet (Domain Features) ──────────────────
        if self.has_biomarker:
            self.biomarker_net = BiomarkerNet(
                n_features=n_bio_features,
                embed_dim=embed_dim, dropout=0.4
            )

        # ── Cross-Path Attention Fusion ──────────────────────────────
        self.cross_attention = CrossPathAttention(
            embed_dim=embed_dim, n_heads=2, dropout=0.1
        )

        # ── Classifier Head ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, n_classes)
        )

    def forward(self, x_1d, x_scalo, x_adj, x_bio=None):
        """
        Parameters
        ----------
        x_1d   : (B, 1, C, T) — raw EEG for EEGNet
        x_scalo: (B, C, F, T) — CWT scalograms for CViT
        x_adj  : (B, C, C)    — connectivity matrices for GNN
        x_bio  : (B, n_bio)   — biomarker features (optional)

        Returns
        -------
        logits : (B, n_classes)
        attn_weights : (B, n_branches, n_branches) — for XAI
        """
        # ── Path A: EEGNet ───────────────────────────────────────────
        x = self.eegnet.conv1(x_1d)
        x = self.eegnet.batchnorm1(x)
        x = self.eegnet.depthwise1(x)
        x = self.eegnet.batchnorm2(x)
        x = self.eegnet.activation1(x)
        x = self.eegnet.se_block(x)
        x = self.eegnet.avg_pool1(x)
        x = self.eegnet.dropout1(x)
        x = self.eegnet.separable1(x)
        x = self.eegnet.batchnorm3(x)
        x = self.eegnet.activation2(x)
        x = self.eegnet.avg_pool2(x)
        x = self.eegnet.dropout2(x)
        embed_eeg = self.eegnet_proj(x.reshape(x.size(0), -1))  # (B, embed_dim)

        # ── Path B: CViT ────────────────────────────────────────────
        embed_cvit = self.cvit(x_scalo)  # (B, embed_dim)

        # ── Path C: GNN ─────────────────────────────────────────────
        if self.use_gnn and 'gnn' not in self.disabled_branches:
            embed_gnn = self.gnn_proj(self.gnn(x_adj))  # (B, embed_dim)
        else:
            embed_gnn = None

        # ── Stack Branches (apply ablation masking) ───────────────────
        branches = []

        if 'eegnet' in self.disabled_branches:
            branches.append(torch.zeros_like(embed_eeg))
        else:
            branches.append(embed_eeg)

        if 'cvit' in self.disabled_branches:
            branches.append(torch.zeros_like(embed_cvit))
        else:
            branches.append(embed_cvit)

        if embed_gnn is not None:
            branches.append(embed_gnn)

        if self.has_biomarker and x_bio is not None:
            embed_bio = self.biomarker_net(x_bio)  # (B, embed_dim)
            if 'biomarker' in self.disabled_branches:
                branches.append(torch.zeros_like(embed_bio))
            else:
                branches.append(embed_bio)

        # (B, n_branches, embed_dim)
        stacked = torch.stack(branches, dim=1)

        # ── Cross-Path Attention ─────────────────────────────────────
        fused, attn_weights = self.cross_attention(stacked)

        # ── Classifier ───────────────────────────────────────────────
        logits = self.classifier(fused)

        return logits, attn_weights
