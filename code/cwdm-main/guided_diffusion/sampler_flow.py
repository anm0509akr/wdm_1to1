# guided_diffusion/wavelet_flow.py
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Haar3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        h = torch.tensor([1.0, 1.0]) / (2.0 ** 0.5)
        g = torch.tensor([1.0, -1.0]) / (2.0 ** 0.5)
        k = []
        for a in (h, g):
            for b in (h, g):
                for c in (h, g):
                    w = a[:, None, None] * b[None, :, None] * c[None, None, :]
                    k.append(w)
        k = torch.stack(k, dim=0)
        k = k.view(8, 1, 2, 2, 2)
        self.register_buffer('enc', k)
        self.register_buffer('dec', k)
        self.channels = channels

    def dwt(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        assert C == self.channels
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0
        weight = self.enc.repeat(C, 1, 1, 1, 1)
        y = F.conv3d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)
        return y

    def idwt(self, y: torch.Tensor) -> torch.Tensor:
        B, C8, D, H, W = y.shape
        C = self.channels
        assert C8 == 8 * C
        weight = self.dec.repeat(C, 1, 1, 1, 1)
        x = F.conv_transpose3d(y, weight=weight, bias=None, stride=2, padding=0, groups=C)
        return x


def dwt3d(x: torch.Tensor) -> torch.Tensor:
    m = Haar3D(x.size(1)).to(x.device)
    return m.dwt(x)


def idwt3d(y: torch.Tensor, channels: int) -> torch.Tensor:
    m = Haar3D(channels).to(y.device)
    return m.idwt(y)