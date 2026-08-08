import torch
import torch.nn as nn

class CViT(nn.Module):
    def __init__(self, in_channels, img_size=(22, 256), patch_size=(11, 16), embed_dim=48, depth=3, num_heads=4, dropout=0.3):
        """
        Convolutional Vision Transformer for EEG Scalograms.
        Balanced capacity: depth=3, embed_dim=48, dropout=0.3
        """
        super().__init__()
        self.patch_size = patch_size
        
        # CNN Backbone to extract patches
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Calculate number of patches
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 2, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        # x shape: (B, C, Freq, Time)
        B = x.shape[0]
        
        # Extract patches
        x = self.conv(x)  # (B, embed_dim, H_out, W_out)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Embedding + dropout
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # Return CLS token representation
        return x[:, 0]
