"""
v2/models/eegnet.py
─────────────────────────────────────────────────────────────────────────────
EEGNet architecture for end-to-end EEG classification.
Based on Lawhern et al. (2018).
"""

import torch
import torch.nn as nn

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EEGNet(nn.Module):
    def __init__(self, nb_classes=2, Chans=16, Samples=1024, 
                 dropoutRate=0.5, kernLength=128, F1=8, D=2, F2=16, extra_dim=0):
        super().__init__()
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        # Block 1
        self.conv1 = nn.Conv2d(1, self.F1, (1, self.kernLength), padding=(0, self.kernLength // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1, affine=False)
        
        self.depthwise1 = Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.Chans, 1), 
                                               groups=self.F1, bias=False, max_norm=1.0)
        self.batchnorm2 = nn.BatchNorm2d(self.F1 * self.D, affine=False)
        self.activation1 = nn.ELU()
        self.se_block = SqueezeExcitation(self.F1 * self.D, reduction=2) # Attention mechanism
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(p=self.dropoutRate)
        
        # Block 2
        self.separable1 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), padding=(0, 8), groups=self.F1 * self.D, bias=False),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False)
        )
        self.batchnorm3 = nn.BatchNorm2d(self.F2, affine=False)
        self.activation2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(p=self.dropoutRate)
        
        # Calculate the size of the features after pooling
        out_samples = self.Samples // 4 // 8
        self.flatten_size = self.F2 * out_samples
        self.extra_dim = extra_dim
        
        # Classification & Regression Heads (Hybrid support)
        combined_dim = self.flatten_size + self.extra_dim
        
        if self.extra_dim > 0:
            self.fusion = nn.Sequential(
                nn.Linear(combined_dim, self.flatten_size),
                nn.ELU(),
                nn.Dropout(p=self.dropoutRate)
            )
            classifier_in = self.flatten_size
        else:
            classifier_in = self.flatten_size
            
        self.classifier = nn.Linear(classifier_in, self.nb_classes, bias=True)
        self.regressor = nn.Linear(classifier_in, 1, bias=True)

    def forward(self, x, extra_features=None):
        # x shape: (batch, 1, Chans, Samples)
        
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise1(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.se_block(x) # Squeeze-and-Excitation Attention
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separable1(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Feature Fusion
        if self.extra_dim > 0 and extra_features is not None:
            x = torch.cat([x, extra_features], dim=1)
            x = self.fusion(x)
            
        # Heads
        logits = self.classifier(x)
        mmse = self.regressor(x)
        
        return logits, mmse
