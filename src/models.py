import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_features=32):
        super().__init__()
        
        # Encoder (Local Feature Extraction)
        self.enc1 = DoubleConv(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck (Where the Transformer will go later)
        self.bottleneck = DoubleConv(base_features * 4, base_features * 8)
        
        # Decoder (Spatial Reconstruction)
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_features * 8, base_features * 4)
        
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_features * 4, base_features * 2)
        
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_features * 2, base_features)
        
        # Output layer (No sigmoid here, we will use BCEWithLogitsLoss)
        self.out_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([e3, d3], dim=1)  # Skip connection
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)  # Skip connection
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)  # Skip connection
        d1 = self.dec1(d1)
        
        out = self.out_conv(d1)
        return out

class TransformerBottleneck(nn.Module):
    """A lightweight self-attention block to learn global context."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten spatial dimensions into a sequence: [B, H*W, C]
        x_seq = x.view(B, C, H * W).permute(0, 2, 1) 
        
        # Transformer Self-Attention
        attn_out, _ = self.attn(self.norm1(x_seq), self.norm1(x_seq), self.norm1(x_seq))
        x_seq = x_seq + attn_out
        
        # Feed-forward Network
        x_seq = x_seq + self.mlp(self.norm2(x_seq))
        
        # Reshape back to image spatial dimensions: [B, C, H, W]
        out = x_seq.permute(0, 2, 1).view(B, C, H, W)
        return out


class HybridUNet(nn.Module):
    """Hybrid CNN-Transformer Architecture."""
    def __init__(self, in_channels=4, out_channels=1, base_features=32):
        super().__init__()
        
        # CNN Encoder (Local Feature Extraction)
        self.enc1 = DoubleConv(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Hybrid Bottleneck: CNN + Transformer
        self.bottleneck_conv = DoubleConv(base_features * 4, base_features * 8)
        self.transformer = TransformerBottleneck(dim=base_features * 8, num_heads=4)
        
        # CNN Decoder (Spatial Reconstruction)
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_features * 8, base_features * 4)
        
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_features * 4, base_features * 2)
        
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_features * 2, base_features)
        
        self.out_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck (Local -> Global)
        b = self.bottleneck_conv(self.pool3(e3))
        b = self.transformer(b)
        
        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([e3, d3], dim=1) 
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out_conv(d1)