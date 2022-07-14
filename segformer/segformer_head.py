# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
# import attr

from IPython import embed
from ..builder import build_loss
from FADE_L2H import FADE, FADELite, SemiShiftUp

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=kwargs['norm_cfg']
        )
        self.upsample_stages = 6
        self.upsample_mode = kwargs['upsample_cfg']['upsample_mode']
        self.is_guided = kwargs['upsample_cfg']['is_guided']
        # print(self.upsample_mode)
        if self.upsample_mode != 'bilinear':
            self.upsamplers = []
            if self.upsample_mode == 'fade':
                for i in range(self.upsample_stages):
                    self.upsamplers.append(FADE(embedding_dim))
            elif self.upsample_mode == 'fade_lite':
                for i in range(self.upsample_stages):
                    self.upsamplers.append(FADELite(embedding_dim))
            elif self.upsample_mode == 'semishift':
                for i in range(self.upsample_stages):
                    self.upsamplers.append(SemiShiftUp(embedding_dim))
            else:
                raise NotImplementedError
            self.upsamplers = nn.ModuleList(self.upsamplers)

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]).contiguous()
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3]).contiguous()
        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3]).contiguous()
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3]).contiguous()

        if self.upsample_mode == 'bilinear':
            _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)
        elif self.is_guided:
            _c4 = self.upsamplers[0](_c3, _c4)
            _c4 = self.upsamplers[1](_c2, _c4)
            _c4 = self.upsamplers[2](_c1, _c4)
        else:
            _c4 = self.upsamplers[0](_c4)
            _c4 = self.upsamplers[1](_c4)
            _c4 = self.upsamplers[2](_c4)

        if self.upsample_mode == 'bilinear':
            _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)
        elif self.is_guided:
            _c3 = self.upsamplers[3](_c2, _c3)
            _c3 = self.upsamplers[4](_c1, _c3)
        else:
            _c3 = self.upsamplers[3](_c3)
            _c3 = self.upsamplers[4](_c3)

        if self.upsample_mode == 'bilinear':
            _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
        elif self.is_guided:
            _c2 = self.upsamplers[5](_c1, _c2)
        else:
            _c2 = self.upsamplers[5](_c2)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
