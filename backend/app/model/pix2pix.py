import torch
import torch.nn as nn
from torchvision import transforms

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, 
                     stride=2, padding=1, bias=not normalize)
        ]
        
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 
                              kernel_size=4, stride=2, 
                              padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip_input=None):
        x = self.model(x)
        
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)
            
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        
        self.down1 = UNetDown(in_channels, 64, normalize=False)  # 256x256 -> 128x128
        self.down2 = UNetDown(64, 128)                           # 128x128 -> 64x64
        self.down3 = UNetDown(128, 256)                          # 64x64 -> 32x32
        self.down4 = UNetDown(256, 512)                          # 32x32 -> 16x16
        self.down5 = UNetDown(512, 512)                          # 16x16 -> 8x8
        self.down6 = UNetDown(512, 512)                          # 8x8 -> 4x4
        self.down7 = UNetDown(512, 512)                          # 4x4 -> 2x2
        self.down8 = UNetDown(512, 512, normalize=False)         # 2x2 -> 1x1
        
        self.up1 = UNetUp(512, 512, dropout=0.5)        # 1x1 -> 2x2
        self.up2 = UNetUp(1024, 512, dropout=0.5)       # 2x2 -> 4x4
        self.up3 = UNetUp(1024, 512, dropout=0.5)       # 4x4 -> 8x8
        self.up4 = UNetUp(1024, 512, dropout=0.0)       # 8x8 -> 16x16
        self.up5 = UNetUp(1024, 256, dropout=0.0)       # 16x16 -> 32x32
        self.up6 = UNetUp(512, 128, dropout=0.0)        # 32x32 -> 64x64
        self.up7 = UNetUp(256, 64, dropout=0.0)         # 64x64 -> 128x128
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        
        total_input_channels = in_channels + out_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(total_input_channels, 64, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 1)
    
    def forward(self, gray_img, rgb_img):
        x = torch.cat((gray_img, rgb_img), 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x