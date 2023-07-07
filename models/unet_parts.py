""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, activation='relu'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        #self.double_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if activation == 'softmax':
            self.activate = nn.Softmax(dim=1)
        else:
            self.activate = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x =  self.double_conv(x)
        return self.activate(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, activation='relu', inner_most=False):
        super().__init__()
        if not inner_most:
            in_channels_up = in_channels*2
        else:
            in_channels_up = in_channels

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels_up, out_channels, in_channels, activation=activation)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels_up, out_channels, activation=activation)

    def forward(self, x1, skip=None):
        x1 = self.up(x1)
        if skip is None:
            return self.conv(x1)

        # input is CHW
        diffY = skip.size()[2] - x1.size()[2]
        diffX = skip.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([skip, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()


    def forward(self, x):
        return self.tanh(self.conv(x))