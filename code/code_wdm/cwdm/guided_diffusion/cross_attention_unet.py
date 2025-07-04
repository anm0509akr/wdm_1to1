import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv3d as Conv, ConvTranspose3d as Deconv

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """Create sinusoidal timestep embeddings like in the original DDPM implementation."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device= timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# -----------------------------------------------------------------------------
# Cross‑Attention block
# -----------------------------------------------------------------------------


class CrossAttention(nn.Module):
    """
    Multi-Head X-Attention with optional spatial down-sampling.
      down_factor = 1 → 従来通り
                  = 2 → D,H,W を 1/2 へ
                  = 4 → 1/4 へ …
    """
    def __init__(self, dim, heads=8, dim_head=64, down_factor=1):
        super().__init__()
        assert down_factor >= 1 and down_factor & (down_factor-1) == 0,\
            "down_factor must be power of 2"
        self.down_factor = down_factor
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head
        self.to_q = nn.Conv3d(dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv3d(dim, inner_dim, 1, bias=False)
        self.to_v = nn.Conv3d(dim, inner_dim, 1, bias=False)
        self.to_out = nn.Conv3d(inner_dim, dim, 1, bias=False)

        # ↓ ここで平均プーリング。stride=down_factor
        if down_factor > 1:
            self.pool = nn.AvgPool3d(kernel_size=down_factor,
                                     stride=down_factor,
                                     padding=0)
        else:
            self.pool = nn.Identity()

    def forward(self, x, context):
        """
        x       : [B, C, D, H, W] (query)
        context : [B, C, D, H, W] (key/value)
        """
        # 1) query はそのまま（プールしない）
    
        b, _, d1, h1, w1 = x.shape
        q = self.to_q(x) .reshape(b, self.heads, -1, d1*h1*w1).transpose(-2, -1)  # (B, heads, HW₁, dim_head)
        print(f"[DEBUG] q tokens = {d1*h1*w1}")

        # 2) key/value はプールしてトークン数を削減
        ctx = self.pool(context)
        b, _, d2, h2, w2 = ctx.shape
        k = self.to_k(ctx) \
                .reshape(b, self.heads, -1, d2*h2*w2)  # (B, heads, dim_head, HW₂)
        v = self.to_v(ctx) \
                .reshape(b, self.heads, -1, d2*h2*w2) \
                .transpose(-2, -1)    # (B, heads, HW₂, dim_head)
        print(f"[DEBUG] k/v tokens = {d2*h2*w2}")

        # 4) Attention
        scores = torch.matmul(q, k) * self.scale                                   # (B, heads, HW, HW')
        attn   = scores.softmax(dim=-1)
        out    = torch.matmul(attn, v)                                             # (B, heads, HW, d)
        out    = out.transpose(-2, -1).reshape(b, -1, d, h, w)
        return self.to_out(out)

# -----------------------------------------------------------------------------
# Building Blocks
# -----------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, with_cross: bool = False, down_factor: int = 1):
        super().__init__()
        self.with_cross = with_cross
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            SiLU(),
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            SiLU(),
            nn.Dropout(0.0),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
        )
        self.time_proj = nn.Sequential(SiLU(), nn.Linear(time_dim, out_ch))
        self.cross_attn = (CrossAttention(out_ch, down_factor=down_factor) if with_cross else None)
        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: Optional[torch.Tensor] = None):
        h = self.block1(x)
        # add timestep
        t = self.time_proj(t_emb)[:, :, None, None, None]
        h = h + t
        # optional cross‑attention
        if self.with_cross and context is not None:
            h = h + self.cross_attn(h, context)
        h = self.block2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


# -----------------------------------------------------------------------------
# Cross‑Attention UNet Model
# -----------------------------------------------------------------------------


class CrossAttentionUNet(nn.Module):
    """A UNet backbone where the noisy input features can attend to conditioning features via cross‑attention."""

    def __init__(self, in_ch: int = 8, cond_ch: int = 24, base_ch: int = 64, ch_mult: Tuple[int] = (1, 2, 4, 8), num_res_blocks: int = 2, ca_down_factor: int = 1, **kwargs):
        super().__init__()
        self.ca_down_factor = ca_down_factor
        self.base_ch = base_ch
        self.time_dim = base_ch * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(base_ch, self.time_dim), SiLU(), nn.Linear(self.time_dim, self.time_dim)
        )
        self.input_conv = nn.Conv3d(in_ch, base_ch, 3, padding=1)
        self.cond_conv = nn.Conv3d(cond_ch, base_ch, 3, padding=1)

        self.downs = nn.ModuleList()
        ch = base_ch
        cond_chs: List[int] = []
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(ch, out_ch, self.time_dim, with_cross=True, down_factor=self.ca_down_factor))
                cond_chs.append(out_ch)
                ch = out_ch
            if i != len(ch_mult) - 1:
                self.downs.append(Downsample(ch))
                cond_chs.append(ch)

        self.mid = nn.Sequential(
            ResBlock(ch, ch, self.time_dim, with_cross=True, down_factor=self.ca_down_factor),
            ResBlock(ch, ch, self.time_dim, with_cross=True, down_factor=self.ca_down_factor),
        )

        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):  # +1 to match skip connections
                self.ups.append(ResBlock(ch + cond_chs.pop(), out_ch, self.time_dim, with_cross=True, down_factor=self.ca_down_factor))
                ch = out_ch
            if i != 0:
                self.ups.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_act = SiLU()
        self.out_conv = nn.Conv3d(ch, in_ch, 3, padding=1)

    # ---------------------------------------------------------------------
    # Encoding of the conditioning images
    # ---------------------------------------------------------------------
    def encode_condition(self, c: torch.Tensor) -> List[torch.Tensor]:
        """Downsample conditioning feature maps so they match UNet resolutions."""
        feats = []
        h = self.cond_conv(c)
        feats.append(h)
        for module in self.downs:
            if isinstance(module, Downsample):
                h = module(h)
            else:
                h = module(h, torch.zeros(h.size(0), self.time_dim, device=h.device), None)
            feats.append(h)
        return feats

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None, **kwargs):
        t_emb = timestep_embedding(t, self.base_ch).to(x.device)
        t_emb = self.time_embedding(t_emb)
        # Encode conditioning path
        if cond is None:
            B, _, D, H, W =x.shape
            cond = torch.zeros(
                B,
                self.cond_conv.in_channels,
                D, H, W,
                device=x.device,
                dtype=x.dtype,
            )
        cond_feats = self.encode_condition(cond)

        h = self.input_conv(x)
        hs: List[torch.Tensor] = []
        feat_idx = 0
        # Down path with cross‑attention
        for module in self.downs:
            if isinstance(module, Downsample):
                h = module(h)
            else:
                h = module(h, t_emb, context=cond_feats[feat_idx])
                feat_idx += 1
            hs.append(h)

        # Mid
        h = self.mid[0](h, t_emb, context=cond_feats[-1])
        h = self.mid[1](h, t_emb, context=cond_feats[-1])

        # Up path
        for module in self.ups:
            if isinstance(module, Upsample):
                h = module(h)
            else:
                skip = hs.pop()
                cond_f = cond_feats.pop()
                h = torch.cat([h, skip], dim=1)
                h = module(h, t_emb, context=cond_f)

        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)
