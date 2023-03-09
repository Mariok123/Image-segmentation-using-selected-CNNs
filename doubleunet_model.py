import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from modules import (
    SqueezeAndExcite,
    ASPP
)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.sae = SqueezeAndExcite(out_c)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.sae(x)
        return x

class Encoder1(nn.Module):
    def __init__(self):
        super().__init__()

        network = vgg19(weights=VGG19_Weights.DEFAULT)
        # print(network)

        self.x1 = network.features[:4]
        self.x2 = network.features[4:9]
        self.x3 = network.features[9:18]
        self.x4 = network.features[18:27]
        self.x5 = network.features[27:36]

    def forward(self, x):
        x0 = x
        x1 = self.x1(x0)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)
        return x5, [x4, x3, x2, x1]

class Encoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))

        self.c1 = ConvBlock(3, 32)
        self.c2 = ConvBlock(32, 64)
        self.c3 = ConvBlock(64, 128)
        self.c4 = ConvBlock(128, 256)

    def forward(self, x):
        x0 = x

        x1 = self.c1(x0)
        p1 = self.pool(x1)

        x2 = self.c2(p1)
        p2 = self.pool(x2)

        x3 = self.c3(p2)
        p3 = self.pool(x3)

        x4 = self.c4(p3)
        p4 = self.pool(x4)

        return p4, [x4, x3, x2, x1]
    
class Decoder1(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = ConvBlock(64+512, 256)
        self.c2 = ConvBlock(512, 128)
        self.c3 = ConvBlock(256, 64)
        self.c4 = ConvBlock(128, 32)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip[0]], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, skip[1]], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, skip[2]], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, skip[3]], axis=1)
        x = self.c4(x)

        return x

class Decoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = ConvBlock(832, 256)
        self.c2 = ConvBlock(640, 128)
        self.c3 = ConvBlock(320, 64)
        self.c4 = ConvBlock(160, 32)

    def forward(self, x, skip1, skip2):
        x = self.up(x)
        x = torch.cat([x, skip1[0], skip2[0]], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, skip1[1], skip2[1]], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, skip1[2], skip2[2]], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, skip1[3], skip2[3]], axis=1)
        x = self.c4(x)

        return x

class DoubleUNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = Encoder1()
        self.a1 = ASPP(512, 64)
        self.d1 = Decoder1()
        self.y1 = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        self.e2 = Encoder2()
        self.a2 = ASPP(256, 64)
        self.d2 = Decoder2()
        self.y2 = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        x, skip1 = self.e1(x)
        x = self.a1(x)
        x = self.d1(x, skip1)
        y1 = self.y1(x)

        input_x = x0 * self.sigmoid(y1)
        x, skip2 = self.e2(input_x)
        x = self.a2(x)
        x = self.d2(x, skip1, skip2)
        y2 = self.y2(x)

        #output = torch.cat([y1, y2], axis=1)
        return y2

if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = DoubleUNET()
    y1, y2, output = model(x)
    print(y1.shape, y2.shape, output.shape)