# guided_diffusion/wavelet_flow.py
# Minimal 3D Haar wavelet transform (1-level) using grouped Conv3d.
# If you already have a validated DWT/IDWT, replace imports in scripts with your implementation.

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Haar3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        h = torch.tensor([1.0, 1.0]) / (2.0 ** 0.5)
        g = torch.tensor([1.0, -1.0]) / (2.0 ** 0.5)
        # 8 subbands
        k = []
        for a in (h, g):
            for b in (h, g):
                for c in (h, g):
                    w = a[:, None, None] * b[None, :, None] * c[None, None, :]
                    k.append(w)
        k = torch.stack(k, dim=0)  # [8,2,2,2]
        k = k.view(8, 1, 2, 2, 2)
        self.register_buffer('enc', k)  # analysis filters
        self.register_buffer('dec', k)  # synthesis filters (Haar is orthonormal)
        self.channels = channels

    def dwt(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,D,H,W] -> y: [B,8C,D/2,H/2,W/2]
        B, C, D, H, W = x.shape
        assert C == self.channels
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, 'Dims must be even for stride=2'
        weight = self.enc.repeat(C, 1, 1, 1, 1)  # [8C,1,2,2,2]
        y = F.conv3d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)
        return y

    def idwt(self, y: torch.Tensor) -> torch.Tensor:
        # y: [B,8C,D,H,W] -> x: [B,C,2D,2H,2W]
        B, C8, D, H, W = y.shape
        C = self.channels
        assert C8 == 8 * C
        weight = self.dec.repeat(C, 1, 1, 1, 1)  # [8C,1,2,2,2]
        x = F.conv_transpose3d(y, weight=weight, bias=None, stride=2, padding=0, groups=C)
        return x


def dwt3d(x: torch.Tensor) -> torch.Tensor:
    m = Haar3D(x.size(1)).to(x.device)
    return m.dwt(x)


def idwt3d(y: torch.Tensor, channels: int) -> torch.Tensor:
    m = Haar3D(channels).to(y.device)
    return m.idwt(y)