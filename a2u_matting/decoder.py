import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from hlconv import hlconv
from inplace_abn import ABN
from FADE_L2H import FADE, FADELite


class Decoder(nn.Module):
    def __init__(self, inp, oup, upsample='bilinear', conv_operator='std_conv', kernel_size=5, batch_norm=ABN):
        super(Decoder, self).__init__()
        hlConv2d = hlconv[conv_operator]
        BatchNorm2d = batch_norm

        # inp, oup, kernel_size, stride, batch_norm
        self.dconv = hlConv2d(inp, oup, kernel_size, 1, BatchNorm2d)
        self.dconv1 = hlConv2d(oup, oup, kernel_size, 1, BatchNorm2d)

        self.upsample = upsample
        if self.upsample == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        elif self.upsample == 'fade':
            self.up = FADE(inp)
        elif self.upsample == 'fade_lite':
            self.up = FADELite(inp)
        else:
            raise NotImplementedError

        self._init_weight()

    def forward(self, l_encode, l_low, f_en=None):
        if f_en is not None:
            if self.upsample == 'bilinear':
                l_encode = self.up(l_encode)
            else:
                l_encode = self.up(f_en, l_encode)
        l_cat = l_encode + l_low if l_low is not None else l_encode
        return self.dconv1(self.dconv(l_cat))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
