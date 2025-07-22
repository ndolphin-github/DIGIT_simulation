import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 96)
        self.enc2 = ConvBlock(96, 192)
        self.enc3 = ConvBlock(192, 384)
        self.enc4 = ConvBlock(384, 768)
        self.middle = ConvBlock(768, 1536)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(1536, 768, 2, stride=2)
        self.dec4 = ConvBlock(1536, 768)
        self.up3 = nn.ConvTranspose2d(768, 384, 2, stride=2)
        self.dec3 = ConvBlock(768, 384)
        self.up2 = nn.ConvTranspose2d(384, 192, 2, stride=2)
        self.dec2 = ConvBlock(384, 192)
        self.up1 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.dec1 = ConvBlock(192, 96)

        self.out = nn.Conv2d(96, out_channels, kernel_size=1)

    def crop_or_pad(self, upsampled, target):
        _, _, h, w = upsampled.shape
        _, _, h2, w2 = target.shape
        dh = h2 - h
        dw = w2 - w
        pad = nn.functional.pad
        if dh > 0 or dw > 0:
            upsampled = pad(upsampled, [0, max(0, dw), 0, max(0, dh)])
        elif dh < 0 or dw < 0:
            upsampled = upsampled[:, :, :h2, :w2]
        return upsampled

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        m = self.middle(self.pool(e4))

        u4 = self.up4(m)
        u4 = self.crop_or_pad(u4, e4)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = self.up3(d4)
        u3 = self.crop_or_pad(u3, e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        u2 = self.crop_or_pad(u2, e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = self.crop_or_pad(u1, e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out(d1)
