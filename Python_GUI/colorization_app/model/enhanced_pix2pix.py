"""
Enhanced Pix2Pix Model Architecture
Extracted and adapted for PyQt5 application from Jupyter notebook
Includes attention mechanisms, SE blocks, and multi-scale discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SelfAttention(nn.Module):
    """Self-attention mechanism for improved feature learning"""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # Use 1x1 convolutions to reduce computation
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # Query, Key, Value projections
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        proj_value = self.value(x).view(batch_size, -1, width * height)

        # Attention map
        attention = torch.bmm(proj_query, proj_key)
        attention = self.softmax(attention)

        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EnhancedUNetDown(nn.Module):
    """Enhanced U-Net downsampling block with optional SE attention"""

    def __init__(
        self, in_channels, out_channels, normalize=True, dropout=0.0, use_se=False
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not normalize,
            )
        ]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

        # Add SE block if enabled
        self.se_block = SEBlock(out_channels) if use_se else None

    def forward(self, x):
        x = self.model(x)
        if self.se_block is not None:
            x = self.se_block(x)
        return x


class EnhancedUNetUp(nn.Module):
    """Enhanced U-Net upsampling block with optional SE attention"""

    def __init__(self, in_channels, out_channels, dropout=0.0, use_se=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

        # Add SE block if enabled
        self.se_block = SEBlock(out_channels) if use_se else None

    def forward(self, x, skip_input=None):
        x = self.model(x)

        if self.se_block is not None:
            x = self.se_block(x)

        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)

        return x


class EnhancedGeneratorUNet(nn.Module):
    """Enhanced U-Net Generator with attention and SE blocks"""

    def __init__(self, in_channels=1, out_channels=3, use_attention=True, use_se=True):
        super().__init__()

        # Encoder
        self.down1 = EnhancedUNetDown(in_channels, 64, normalize=False, use_se=use_se)
        self.down2 = EnhancedUNetDown(64, 128, use_se=use_se)
        self.down3 = EnhancedUNetDown(128, 256, use_se=use_se)
        self.down4 = EnhancedUNetDown(256, 512, use_se=use_se)
        self.down5 = EnhancedUNetDown(512, 512, use_se=use_se)
        self.down6 = EnhancedUNetDown(512, 512, use_se=use_se)
        self.down7 = EnhancedUNetDown(512, 512, use_se=use_se)
        self.down8 = EnhancedUNetDown(512, 512, normalize=False, use_se=use_se)

        # Self-attention at bottleneck
        self.attention = SelfAttention(512) if use_attention else nn.Identity()

        # Decoder
        self.up1 = EnhancedUNetUp(512, 512, dropout=0.5, use_se=use_se)
        self.up2 = EnhancedUNetUp(1024, 512, dropout=0.5, use_se=use_se)
        self.up3 = EnhancedUNetUp(1024, 512, dropout=0.5, use_se=use_se)
        self.up4 = EnhancedUNetUp(1024, 512, dropout=0.0, use_se=use_se)
        self.up5 = EnhancedUNetUp(1024, 256, dropout=0.0, use_se=use_se)
        self.up6 = EnhancedUNetUp(512, 128, dropout=0.0, use_se=use_se)
        self.up7 = EnhancedUNetUp(256, 64, dropout=0.0, use_se=use_se)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Apply attention at bottleneck
        d8 = self.attention(d8)

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class PatchDiscriminator(nn.Module):
    """Patch-based discriminator for adversarial training"""

    def __init__(self, in_channels=4):  # 1 (gray) + 3 (RGB)
        super().__init__()

        self.model = nn.Sequential(
            # C64
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # C128
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # C256
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # C512
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, gray, rgb):
        x = torch.cat([gray, rgb], dim=1)
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for improved training stability"""

    def __init__(self, in_channels=1, out_channels=3, num_scales=3):
        super().__init__()
        self.num_scales = num_scales

        # Create discriminators at different scales
        self.discriminators = nn.ModuleList(
            [PatchDiscriminator(in_channels + out_channels) for _ in range(num_scales)]
        )

        # Downsampling layer
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, gray, rgb):
        outputs = []

        # Run through each discriminator at different scales
        for i, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(gray, rgb))

            # Downsample for next scale (except for last scale)
            if i < self.num_scales - 1:
                gray = self.downsample(gray)
                rgb = self.downsample(rgb)

        return outputs


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features for better image quality"""

    def __init__(self):
        super().__init__()
        # Load pretrained VGG16
        vgg = models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1").features

        # Use features from multiple layers
        self.slice1 = nn.Sequential(*list(vgg[:4]))  # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:16]))  # relu3_3
        self.slice4 = nn.Sequential(*list(vgg[16:23]))  # relu4_3

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        # Normalization for ImageNet
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, x):
        # Denormalize from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Normalize for VGG
        return (x - self.mean) / self.std

    def forward(self, fake, real):
        # Normalize inputs
        fake = self.normalize(fake)
        real = self.normalize(real)

        # Extract features
        fake_features = []
        real_features = []

        x_fake = fake
        x_real = real

        for slice_layer in [self.slice1, self.slice2, self.slice3, self.slice4]:
            x_fake = slice_layer(x_fake)
            x_real = slice_layer(x_real)
            fake_features.append(x_fake)
            real_features.append(x_real)

        # Calculate loss across all layers
        loss = 0
        for fake_feat, real_feat in zip(fake_features, real_features):
            loss += F.l1_loss(fake_feat, real_feat)

        return loss / len(fake_features)


def weights_init_normal(m):
    """Initialize network weights with normal distribution"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def create_enhanced_generator(
    in_channels=1, out_channels=3, use_attention=True, use_se=True
):
    """Factory function to create enhanced generator"""
    generator = EnhancedGeneratorUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        use_attention=use_attention,
        use_se=use_se,
    )
    generator.apply(weights_init_normal)
    return generator


def create_multi_scale_discriminator(in_channels=1, out_channels=3, num_scales=3):
    """Factory function to create multi-scale discriminator"""
    discriminator = MultiScaleDiscriminator(
        in_channels=in_channels, out_channels=out_channels, num_scales=num_scales
    )
    discriminator.apply(weights_init_normal)
    return discriminator
