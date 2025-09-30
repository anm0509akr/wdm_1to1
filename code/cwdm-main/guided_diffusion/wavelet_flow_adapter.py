from __future__ import annotations

from typing import Sequence, Tuple
import torch

# ---- Import your DWT/IDWT implementation (repo layout: DWT_IDWT/DWT_IDWT_layer.py)
try:  # primary (recommended)
    from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D  # type: ignore
except Exception as e:  # optional fallback: relative copy next to this file
    try:
        from .DWT_IDWT_layer import DWT_3D, IDWT_3D  # type: ignore
    except Exception as ee:
        raise ImportError(
            "Could not import DWT_3D/IDWT_3D. Place DWT_IDWT/DWT_IDWT_layer.py at the repo root, "
            "or copy DWT_IDWT_layer.py next to guided_diffusion/ and retry."
        ) from ee

# Channel order if DWT returns 8 tensors
SUBBAND_ORDER: Tuple[str, ...] = (
    "LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"
)

__all__ = ["dwt3d", "idwt3d", "SUBBAND_ORDER"]


def _assert_5d(x: torch.Tensor, name: str) -> None:
    if x.ndim != 5:
        raise ValueError(f"{name} must be 5D [B,C,D,H,W], got shape={tuple(x.shape)}")


def dwt3d(x: torch.Tensor, wavename: str = "haar") -> torch.Tensor:
    """Apply 3D DWT and pack 8 subbands along channel dim.

    Accepts both backends:
      1) returns 8 tensors: (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
      2) returns 2 tensors: (LL, Hstack) where Hstack is [B, 7*C, ...]

    Parameters
    ----------
    x : torch.Tensor
        Input volume [B, C, D, H, W] (even D/H/W). Range is arbitrary.
    wavename : str
        Wavelet name for your implementation (default: "haar").

    Returns
    -------
    torch.Tensor
        Packed subbands [B, 8*C, D/2, H/2, W/2].
    """
    _assert_5d(x, "x")
    dwt = DWT_3D(wavename)
    out = dwt(x)

    # Case A: 8 subbands returned
    if isinstance(out, (tuple, list)) and len(out) == 8:
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = out  # type: ignore[misc]
        return torch.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

    # Case B: (LL, Hstack) returned
    if isinstance(out, (tuple, list)) and len(out) == 2:
        LL, Hstack = out  # type: ignore[misc]
        C = x.shape[1]
        if Hstack.shape[1] != 7 * C:
            raise RuntimeError(
                f"Unexpected Hstack channels: {Hstack.shape[1]} vs expected {7*C}. "
                "Make sure your DWT packs exactly 7*C high-frequency channels."
            )
        return torch.cat([LL, Hstack], dim=1)

    # Otherwise unsupported
    raise RuntimeError(
        f"DWT_3D returned unsupported type/len: {type(out)} len="
        f"{getattr(out, '__len__', lambda: 'n/a')()}"
    )


def idwt3d(y: torch.Tensor, wavename: str = "haar") -> torch.Tensor:
    """Inverse 3D DWT.

    Accepts both backends (tries 8-arg IDWT first, falls back to 2-arg):
      - 8-arg: IDWT_3D(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
      - 2-arg: IDWT_3D(LL, Hstack)

    Parameters
    ----------
    y : torch.Tensor
        Packed subbands [B, 8*C, D/2, H/2, W/2]
    wavename : str
        Wavelet name.

    Returns
    -------
    torch.Tensor
        Reconstructed volume [B, C, D, H, W].
    """
    _assert_5d(y, "y")
    B, C8, D2, H2, W2 = y.shape
    if C8 % 8 != 0:
        raise ValueError(f"channels must be multiple of 8, got {C8}")
    C = C8 // 8

    LL = y[:, :C, ...]
    Hst = y[:, C:, ...]  # [B, 7*C, ...]

    # Split Hstack into 7 parts for 8-arg IDWT (common case)
    parts: Sequence[torch.Tensor] = list(torch.split(Hst, C, dim=1))
    if len(parts) != 7:
        raise RuntimeError(f"expected 7 high-frequency groups, got {len(parts)}")

    idwt = IDWT_3D(wavename)

    # Try 8-arg API first (many implementations use this)
    try:
        LLL, LLH, LHL, LHH, HLL, HLH, HHL = parts  # type: ignore[misc]
        # Heuristic ordering aligns with SUBBAND_ORDER
        return idwt(LL, LLL, LLH, LHL, LHH, HLL, HLH, HHL)  # type: ignore[call-arg]
    except TypeError:
        # Fallback to 2-arg API (LL, Hstack)
        return idwt(LL, Hst)  # type: ignore[call-arg]

def make_t_schedule(steps: int, rho: float = 1.5, device: torch.device | None = None) -> torch.Tensor:
    """
    t を 0→1 に単調増加で並べるスケジュールを作る（CFMの前向き積分用）。
    steps: 時間離散化の個数（例: 80, 100, 160）
    rho:   [0,1] を s^rho で歪ませる（大きいほど序盤を細かく）
    """
    if device is None:
        device = torch.device("cpu")
    if steps < 2:
        return torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)
    s = torch.linspace(0.0, 1.0, steps, dtype=torch.float32, device=device)
    t = s.pow(rho).clamp_(0.0, 1.0)
    t[0], t[-1] = 0.0, 1.0
    return t
