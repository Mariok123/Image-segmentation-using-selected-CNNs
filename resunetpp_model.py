import torch
import torch.nn as nn
from modules import (
    SqueezeAndExcite,
    ASPP
)

class StemBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride, 0),
            nn.BatchNorm2d(out_c)
        )

        self.sae = SqueezeAndExcite(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.shortcut(inputs)
        y = self.sae(x + s)
        return y

class ResNetBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, 3, stride, 1)
        )

        self.c2 = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride, 0),
            nn.BatchNorm2d(out_c)
        )

        self.sae = SqueezeAndExcite(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        x = self.c2(x)
        s = self.shortcut(inputs)
        y = self.sae(x + s)
        return y

class AttentionBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(),
            nn.Conv2d(in_c[0], out_c, 3, 1, 1),
            nn.MaxPool2d((2, 2))
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(in_c[1], out_c, 3, 1, 1)
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a1 = AttentionBlock(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNetBlock(in_c[0]+in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d

class ResUNETpp(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = StemBlock(3, 16, stride=1)
        self.c2 = ResNetBlock(16, 32, stride=2)
        self.c3 = ResNetBlock(32, 64, stride=2)
        self.c4 = ResNetBlock(64, 128, stride=2)

        self.b1 = ASPP(128, 256)

        self.d1 = DecoderBlock([64, 256], 128)
        self.d2 = DecoderBlock([32, 128], 64)
        self.d3 = DecoderBlock([16, 64], 32)

        self.aspp = ASPP(32, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, inputs):
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)

        b1 = self.b1(c4)

        d1 = self.d1(c3, b1)
        d2 = self.d2(c2, d1)
        d3 = self.d3(c1, d2)

        output = self.aspp(d3)
        output = self.output(output)

        return output