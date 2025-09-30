# guided_diffusion/unet_flow.py
# Minimal 3D U-Net that predicts velocity in wavelet coefficient space.
# Time embedding: sinusoidal + MLP, applied as FiLM (scale/shift) at each block.

from typing import Optional, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    # t: [B,] or [B,1,...] in [0,1]
    if t.dim() > 1:
        t = t.view(t.size(0))
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(0, half, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / (half - 1))
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FiLM(nn.Module):
    def __init__(self, in_ch: int, temb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(temb_dim, 2 * in_ch)
        )
    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.mlp(temb)[:, :, None, None, None]
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        return x * (1 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim, dropout=0.0):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.film = FiLM(out_ch, temb_dim)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, temb)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Down(nn.Module):
    """Pre-downsample residuals at ch_in, then stride-2 conv to ch_out."""
    def __init__(self, ch_in: int, ch_out: int, temb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.res1 = ResBlock(ch_in, ch_in, temb_dim, dropout)
        self.res2 = ResBlock(ch_in, ch_in, temb_dim, dropout)
        # ★ ch_in → ch_out にチャンネルアップしながらダウンサンプル
        self.down = nn.Conv3d(ch_in, ch_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x, temb):
        x = self.res1(x, temb)
        x = self.res2(x, temb)
        skip = x                    # skip は ch_in
        x = self.down(x)            # x は ch_out
        return x, skip


class Up(nn.Module):
    """Upsample, concat skip (ch_skip), then reduce to ch_out."""
    def __init__(self, ch_in: int, ch_skip: int, ch_out: int, temb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.res1 = ResBlock(ch_in + ch_skip, ch_out, temb_dim, dropout)
        self.res2 = ResBlock(ch_out, ch_out, temb_dim, dropout)

    def forward(self, x, skip, temb):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, temb)
        x = self.res2(x, temb)
        return x


class UNet3DFlow(nn.Module):
    """3D U-Net that outputs velocity u_theta(x,t,cond) with same channel size as x.
    cond is concatenated at input along channel dim.
    """
    def __init__(self, in_ch: int, cond_ch: int = 0, base_ch: int = 64, depth: int = 4,
                 temb_dim: int = 256, out_ch: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        assert depth >= 2, "depth >= 2 を推奨"
        self.in_ch = in_ch
        self.cond_ch = cond_ch
        self.out_ch = out_ch or in_ch
        self.temb_dim = temb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim * 4), nn.SiLU(), nn.Linear(temb_dim * 4, temb_dim)
        )

        # チャネルスケジュール（例: [64, 128, 256, 512]）
        chs: List[int] = [base_ch * (2 ** i) for i in range(depth)]

        # stem
        self.stem = nn.Conv3d(in_ch + cond_ch, chs[0], 3, padding=1)

        # encoder
        self.downs = nn.ModuleList([
            Down(chs[i], chs[i + 1], temb_dim, dropout=dropout) for i in range(depth - 1)
        ])

        # bottleneck
        self.mid1 = ResBlock(chs[-1], chs[-1], temb_dim, dropout)
        self.mid2 = ResBlock(chs[-1], chs[-1], temb_dim, dropout)

        # decoder（逆順で Up を積む）
        self.ups = nn.ModuleList([
            Up(ch_in=chs[i + 1], ch_skip=chs[i], ch_out=chs[i], temb_dim=temb_dim, dropout=dropout)
            for i in reversed(range(depth - 1))
        ])

        # head
        self.norm = nn.GroupNorm(8, chs[0])
        self.head = nn.Conv3d(chs[0], self.out_ch, 3, padding=1)

    def forward(self, x, t, cond: Optional[torch.Tensor] = None):
        # t: [B,] or [B,1,1,1,1] in [0,1]
        if t.dim() == 5:
            t = t.view(t.size(0))
        temb = self.time_mlp(timestep_embedding(t, self.temb_dim))

        h = x if cond is None else torch.cat([x, cond], dim=1)
        h = self.stem(h)

        skips = []
        for down in self.downs:
            h, s = down(h, temb)
            skips.append(s)

        h = self.mid1(h, temb)
        h = self.mid2(h, temb)

        for up in self.ups:
            h = up(h, skips.pop(), temb)

        h = F.silu(self.norm(h))
        return self.head(h)
