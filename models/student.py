import torch
import torch.nn as nn


class ParallelConvBlock(nn.Module):
    """
    Parallel Convolutional Block: The first layer of the student model,
    capturing multi-scale features through parallel 1x1, 3x3, and 5x5 convolutions.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Initial embedding layer
        self.embed = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Three parallel convolutional branches
        self.p1 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.p2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.p3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        feat = self.embed(x)
        o1 = self.pool(self.p1(feat))
        o2 = self.pool(self.p2(feat))
        o3 = self.pool(self.p3(feat))
        # Concatenate features from three branches and return intermediate feature map for skip connection
        return torch.cat([o1, o2, o3], dim=1), feat


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LATUPStudent(nn.Module):
    """
    Lightweight Student Model Architecture.
    """

    def __init__(self, in_channels=4, out_channels=3, base_filters=16, deep_supervision=True):
        super().__init__()
        self.ds = deep_supervision

        # --- Encoder ---
        self.enc1_pc = ParallelConvBlock(in_channels, base_filters)  # Output channels: base_filters * 3

        self.enc2 = ConvBlock(base_filters * 3, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2)

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(base_filters * 4, base_filters * 8)

        # --- Decoder ---
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = ConvBlock(base_filters * 8 + base_filters * 4, base_filters * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = ConvBlock(base_filters * 4 + base_filters * 2, base_filters * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = ConvBlock(base_filters * 2 + base_filters, base_filters)

        # --- Output Heads ---
        self.out_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

        # Deep Supervision Outputs
        if self.ds:
            self.out_conv2 = nn.Conv3d(base_filters * 2, out_channels, kernel_size=1)
            self.out_conv3 = nn.Conv3d(base_filters * 4, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder phase
        x1, skip1 = self.enc1_pc(x)
        skip2 = self.enc2(x1);
        x2 = self.pool2(skip2)
        skip3 = self.enc3(x2);
        x3 = self.pool3(skip3)

        # Bottleneck phase
        x_bot = self.bottleneck(x3)

        # Decoder phase (with skip connections)
        d3 = self.dec3(torch.cat([self.up3(x_bot), skip3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), skip2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), skip1], dim=1))

        logits = self.out_conv(d1)

        # Returns outputs at multiple scales in training mode for auxiliary loss calculation
        if self.ds and self.training:
            return [logits, self.out_conv2(d2), self.out_conv3(d3)]
        return logits