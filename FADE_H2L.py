# README:
# Implementation of FADE: Fusing the Assets of Decoder and Encoder for Task-Agnostic Upsampling.
# Requirements:
# 1. PyTorch
# 2. mmcv


import torch
from torch import nn
import torch.nn.functional as F
from mmcv.ops.carafe import carafe
from mmcv.cnn import xavier_init


def space_reassemble(x, scale=2):
    B, C, H, W = x.shape
    C = C // scale ** 2
    return x.permute(0, 2, 3, 1).contiguous().view(
        B, H, W, scale, scale, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(
        B, scale * W, scale * W, C).permute(0, 3, 1, 2).contiguous()


class GateGenerator(nn.Module):
    def __init__(self, in_channels):
        super(GateGenerator, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.weights_init_random()

    def forward(self, x):
        return torch.sigmoid(F.interpolate(self.conv(x), scale_factor=2))

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class SemiShift(nn.Module):
    def __init__(self, in_channels_en, in_channels_de, scale=2, embedding_dim=64, up_kernel_size=5):
        super(SemiShift, self).__init__()
        self.scale = scale
        self.embedding_dim = embedding_dim
        self.up_kernel_size = up_kernel_size
        self.conv1_en = nn.Conv2d(in_channels_en, embedding_dim, kernel_size=1)
        self.conv1_de = nn.Conv2d(in_channels_de, embedding_dim, kernel_size=1, bias=False)
        self.conv2_kernels = nn.Parameter(torch.empty((up_kernel_size ** 2, embedding_dim, 3, 3)))
        nn.init.xavier_normal_(self.conv2_kernels, gain=1)
        self.conv2_bias = nn.Parameter(torch.empty(up_kernel_size ** 2))
        nn.init.constant_(self.conv2_bias, val=0)
        self.weights_init_random()

    def forward(self, en, de):
        B, C, H, W = de.shape
        compressed_en = self.conv1_en(en)
        compressed_de = self.conv1_de(de)
        pad_en = []
        pad_en.append(F.pad(compressed_en, pad=[1, 0, 1, 0]))
        pad_en.append(F.pad(compressed_en, pad=[0, 1, 1, 0]))
        pad_en.append(F.pad(compressed_en, pad=[1, 0, 0, 1]))
        pad_en.append(F.pad(compressed_en, pad=[0, 1, 0, 1]))
        pad_en = torch.cat(pad_en, dim=1).view(B * self.scale ** 2,
                                               self.embedding_dim, self.scale * H + 1,
                                               self.scale * H + 1)
        kernels = F.conv2d(pad_en, self.conv2_kernels, self.conv2_bias, stride=2
                           ).view(B, self.scale ** 2, self.up_kernel_size ** 2, H, W) + \
                  F.conv2d(compressed_de, self.conv2_kernels, self.conv2_bias, stride=1,
                           padding=1).unsqueeze(1)
        kernels = space_reassemble(kernels.view(B, self.scale ** 2 * self.up_kernel_size ** 2, H, W))
        return kernels

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class FADE(nn.Module):
    def __init__(self, in_channels_en, in_channels_de=None, scale=2, up_kernel_size=5):
        super(FADE, self).__init__()
        in_channels_de = in_channels_de if in_channels_de is not None else in_channels_en
        self.scale = scale
        self.up_kernel_size = up_kernel_size
        self.gate_generator = GateGenerator(in_channels_de)
        self.kernel_generator = SemiShift(in_channels_en, in_channels_de,
                                              up_kernel_size=up_kernel_size, scale=scale)

    def forward(self, en, de):
        gate = self.gate_generator(de)
        kernels = F.softmax(self.kernel_generator(en, de), dim=1)
        return gate * en + (1 - gate) * carafe(de, kernels, self.up_kernel_size, 1, self.scale)


if __name__ == '__main__':
    x = torch.randn(2, 3, 4, 4).to('cuda')
    y = torch.randn(2, 3, 8, 8).to('cuda')
    fade = FADE(3).to('cuda')
    print(fade(y, x).shape)
