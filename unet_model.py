import torch
import torch.nn as nn

# One step of the UNET architecture (2 blue arrows)
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

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return x

# The encoder part (left part) of the UNET model
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))

        self.c1 = ConvBlock(3, 64)
        self.c2 = ConvBlock(64, 128)
        self.c3 = ConvBlock(128, 256)
        self.c4 = ConvBlock(256, 512)

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

# The denoder part (right part) of the UNET model
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.c1 = ConvBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.c2 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c3 = ConvBlock(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c4 = ConvBlock(128, 64)


    def forward(self, x, skip):
        x = self.up1(x)
        x = torch.cat([x, skip[0]], axis=1)
        x = self.c1(x)

        x = self.up2(x)
        x = torch.cat([x, skip[1]], axis=1)
        x = self.c2(x)

        x = self.up3(x)
        x = torch.cat([x, skip[2]], axis=1)
        x = self.c3(x)

        x = self.up4(x)
        x = torch.cat([x, skip[3]], axis=1)
        x = self.c4(x)

        return x
    
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.e = Encoder()
        self.bottom = ConvBlock(512, 1024) # Bottom layer of UNET
        self.d = Decoder()
        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x, skip = self.e(x)
        x = self.bottom(x)
        x = self.d(x, skip)
        y = self.output(x)

        return y

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET()
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
