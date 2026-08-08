"""
bilstm.py
─────────────────────────────────────────────────────────────────────────────
Bidirectional LSTM (BiLSTM) architecture for end-to-end EEG sequence classification.
Extracts temporal dynamics across raw 1D multichannel EEG signals.
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM architecture for EEG temporal feature learning.

    Parameters
    ----------
    in_channels : int — number of EEG channels (default 16)
    hidden_size : int — hidden state dimension per direction
    num_layers : int — number of stacked LSTM layers
    n_classes : int — output classification categories
    dropout : float — dropout rate for regularization
    """
    def __init__(self, in_channels=16, hidden_size=64, num_layers=2, n_classes=2, dropout=0.5):
        super().__init__()
        self.in_channels = in_channels

        # 1D Conv feature extraction & temporal pooling to compress sequence length efficiently
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=dropout / 2)
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        """
        Forward pass.
        x: (B, 1, C, T) or (B, C, T)
        """
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, C, T)

        feat = self.feature_extractor(x)      # (B, 32, T_new)
        feat = feat.permute(0, 2, 1)          # (B, T_new, 32)

        out, _ = self.lstm(feat)              # (B, T_new, hidden_size * 2)
        out_pooled = torch.mean(out, dim=1)   # (B, hidden_size * 2)

        logits = self.classifier(out_pooled)
        return logits
