import torch
from torch import nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import einops


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
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

    def forward(self, en, de):
        B, _, H, W = de.shape
        compressed_en = self.conv1_en(en)
        compressed_de = self.conv1_de(de)
        pad_en = []
        pad_en.append(F.pad(compressed_en, pad=[1, 0, 1, 0]))
        pad_en.append(F.pad(compressed_en, pad=[0, 1, 1, 0]))
        pad_en.append(F.pad(compressed_en, pad=[1, 0, 0, 1]))
        pad_en.append(F.pad(compressed_en, pad=[0, 1, 0, 1]))
        pad_en = torch.cat(pad_en, dim=1)

        # h = H + 1, w = W + 1
        pad_en = einops.rearrange(pad_en, 'b (c scale_2) h w -> (b scale_2) c h w', scale_2=self.scale ** 2)
        kernels = F.conv2d(pad_en, self.conv2_kernels, self.conv2_bias, stride=2)
        # c = self.up_kernel_size ** 2
        kernels = einops.rearrange(kernels, '(b scale_2) c h w -> b scale_2 c h w', scale_2=self.scale ** 2)

        kernels = kernels + F.conv2d(compressed_de, self.conv2_kernels, self.conv2_bias, stride=1,
                           padding=1).unsqueeze(1)
        
        kernels = einops.rearrange(kernels, 'b (scale1 scale2) c h w -> b c (h scale1) (w scale2)',
                                   scale1=self.scale, scale2=self.scale)
        return kernels



class FADE(nn.Module):
    def __init__(self, in_channels_en, in_channels_de=None, scale=2, up_kernel_size=5):
        super(FADE, self).__init__()
        in_channels_de = in_channels_de if in_channels_de is not None else in_channels_en
        self.scale = scale
        self.up_kernel_size = up_kernel_size
        self.gate_generator = GateGenerator(in_channels_de)
        self.kernel_generator = SemiShift(in_channels_en, in_channels_de,
                                              up_kernel_size=up_kernel_size, scale=scale)
        self.carafe = CARAFE()

    def forward(self, en, de):
        gate = self.gate_generator(de)
        kernels = F.softmax(self.kernel_generator(en, de), dim=1)
        return gate * en + (1 - gate) * self.carafe(de, kernels, self.up_kernel_size, self.scale)


class CARAFE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, kernel, kernel_size=5, ratio=2):
        B, C, H, W = x.shape
        x = F.unfold(x, kernel_size=kernel_size, stride=1, padding=2)
        x = einops.rearrange(x, 'b (c k_up2) (h w) -> b k_up2 c h w',
                             k_up2=kernel_size**2, w=W)
        x = einops.repeat(x, 'b k c h w -> ratio_2 b k c h w', ratio_2=ratio**2)
        x = einops.rearrange(x, '(r1 r2) b k_up2 c h w -> b k_up2 c (h r1) (w r2)',
                             r1=ratio)
        x = torch.einsum('bkchw,bkhw->bchw',[x, kernel])
        return x
    


if __name__ == '__main__':
    x = torch.randn(2, 3, 12, 16).to('cuda')
    y = torch.randn(2, 3, 24, 32).to('cuda')
    fade = FADE(3).to('cuda')
    print(fade(y, x).shape)
