# README:
# Implementation of FADE: A Task-Agnostic Upsampling Operator for Encoder-Decoder Architectures.
# Requirements:
# 1. PyTorch
# 2. mmcv


import torch
from torch import nn
import torch.nn.functional as F
from mmcv.ops.carafe import carafe
from mmcv.cnn import xavier_init


class GateGenerator(nn.Module):
    def __init__(self, in_channels):
        super(GateGenerator, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.weights_init_random()

    def forward(self, x):
        return torch.sigmoid(F.interpolate(self.conv(x), scale_factor=2))

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class Aligner(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Aligner, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.weights_init_random()

    def forward(self, x):
        return self.conv(x)

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class SemiShift(nn.Module):
    def __init__(self, in_channels_en, in_channels_de, out_channels, embedding_dim=64, kernel_size=3):
        super(SemiShift, self).__init__()
        self.compressor_en = nn.Conv2d(in_channels_en, embedding_dim, kernel_size=1)
        self.compressor_de = nn.Conv2d(in_channels_de, embedding_dim, kernel_size=1, bias=False)
        self.content_encoder = nn.Conv2d(embedding_dim, out_channels, kernel_size=kernel_size,
                                         padding=kernel_size // 2)
        self.weights_init_random()

    def forward(self, en, de):
        enc = self.compressor_en(en)
        dec = self.compressor_de(de)
        output = self.content_encoder(enc) + F.interpolate(self.content_encoder(dec), scale_factor=2)
        return output

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class SemiShiftDepthWise(nn.Module):
    def __init__(self, in_channels_en, in_channels_de, out_channels, kernel_size=3):
        super(SemiShiftDepthWise, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.compressor_en = nn.Conv2d(in_channels_en, out_channels, kernel_size=1)
        self.compressor_de = nn.Conv2d(in_channels_de, out_channels, kernel_size=1, bias=False)
        self.content_encoder = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                                         padding=kernel_size // 2, groups=out_channels)
        self.weights_init_random()

    def forward(self, en, de):
        enc = self.compressor_en(en)
        dec = self.compressor_de(de)
        output = self.content_encoder(enc) + F.interpolate(self.content_encoder(dec), scale_factor=2)
        return output

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class KernelGenerator(nn.Module):
    def __init__(self, in_channels_en, in_channels_de, conv, up_kernel_size=5):
        super(KernelGenerator, self).__init__()
        self.conv = conv(in_channels_en, in_channels_de, out_channels=up_kernel_size ** 2)

    def forward(self, en, de):
        return F.softmax(self.conv(en, de), dim=1)


class SemiShiftUp(nn.Module):
    def __init__(self, in_channels_en, in_channels_de=None, scale=2, up_kernel_size=5):
        super(SemiShiftUp, self).__init__()
        in_channels_de = in_channels_de if in_channels_de is not None else in_channels_en
        self.scale = scale
        self.up_kernel_size = up_kernel_size
        self.ker_generator = SemiShift(in_channels_en, in_channels_de, out_channels=up_kernel_size ** 2)

    def forward(self, en, de):
        kernels = F.softmax(self.ker_generator(en, de), dim=1)
        return carafe(de, kernels, self.up_kernel_size, 1, self.scale)


class FADE(nn.Module):
    def __init__(self, in_channels_en, in_channels_de=None, scale=2, up_kernel_size=5):
        super(FADE, self).__init__()
        in_channels_de = in_channels_de if in_channels_de is not None else in_channels_en
        self.scale = scale
        self.up_kernel_size = up_kernel_size
        self.gate_generator = GateGenerator(in_channels_de)
        # self.aligner = Aligner(in_channels_en, in_channels_de)
        self.ker_generator = SemiShift(in_channels_en, in_channels_de,
                                       out_channels=up_kernel_size ** 2)

    def forward(self, en, de):
        gate = self.gate_generator(de)
        kernels = F.softmax(self.ker_generator(en, de), dim=1)
        return gate * en + (1 - gate) * carafe(de, kernels, self.up_kernel_size, 1, self.scale)


class FADELite(nn.Module):
    def __init__(self, in_channels_en, in_channels_de=None, scale=2, up_kernel_size=5):
        super(FADELite, self).__init__()
        in_channels_de = in_channels_de if in_channels_de is not None else in_channels_en
        self.scale = scale
        self.up_kernel_size = up_kernel_size
        self.gate_generator = GateGenerator(in_channels_de)
        # self.aligner = Aligner(in_channels_en, in_channels_de)
        self.ker_generator = SemiShiftDepthWise(in_channels_en, in_channels_de,
                                                out_channels=up_kernel_size ** 2)

    def forward(self, en, de):
        gate = self.gate_generator(de)
        kernels = F.softmax(self.ker_generator(en, de), dim=1)
        return gate * en + (1 - gate) * carafe(de, kernels, self.up_kernel_size, 1, self.scale)


if __name__ == '__main__':
    x = torch.randn(2, 3, 4, 4).to('cuda')
    y = torch.randn(2, 3, 8, 8).to('cuda')
    fade = FADE(3).to('cuda')
    fade_lite = FADELite(3).to('cuda')
    print(fade(y, x).shape)
    print(fade_lite(y, x).shape)
