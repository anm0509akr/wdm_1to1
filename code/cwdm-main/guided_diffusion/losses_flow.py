# guided_diffusion/losses_flow.py
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from guided_diffusion.wavelet_flow_adapter import dwt3d, SUBBAND_ORDER

def make_t(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch_size, device=device)  # U[0,1], [B]

@torch.no_grad()
def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.sum(a * b)
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-12
    return float((num / den).item())

def _broadcast_t(t: torch.Tensor, x_like: torch.Tensor) -> torch.Tensor:
    while t.dim() < x_like.dim():
        t = t.view(t.size(0), *([1] * (x_like.dim() - 1)))
    return t

def weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                 sb_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if sb_weight is None:
        return F.mse_loss(pred, target)
    w = sb_weight
    while w.dim() < pred.dim():
        w = w.view(1, -1, *([1] * (pred.dim() - 2)))  # [C]→[1,C,1,1,1]
    return ((pred - target) ** 2 * w).mean()

def compute_cfm_loss(
    model,
    *,
    x1: torch.Tensor,                 # GT（空間ドメイン）[B,1,D,H,W]
    cond,                             # 条件（dictなら cond['cond_1'] を使用）
    sigma: float = 0.0,
    sb_weight: Optional[torch.Tensor] = None,   # [8] または [8*C]
) -> Dict[str, object]:
    """
    条件付き CFM を Wavelet 空間で計算（x0=cond, x1=GT）。サブバンド別ログも返す。
    """
    device = x1.device

    # --- cond 抽出（dict/Tensor 両対応） ---
    x0_sp = cond['cond_1'] if isinstance(cond, dict) else cond     # [B,1,D,H,W]

    # --- Wavelet に変換（8*C チャンネル） ---
    x0 = dwt3d(x0_sp)                      # [B, 8*C, D/2, H/2, W/2]
    x1 = dwt3d(x1)                         # 同形状
    B, C8, *_ = x0.shape
    assert C8 % 8 == 0, "wavelet チャンネルは 8*C である必要があります"
    C = C8 // 8

    # --- CFM 補間点 & 目標速度 ---
    t  = make_t(B, device)                 # [B]
    tB = _broadcast_t(t, x0)               # [B,1,1,1,1]
    xt  = (1.0 - tB) * x0 + tB * x1
    if sigma > 0.0:
        eps = torch.randn_like(x0)
        g = tB * (1.0 - tB)
        xt = xt + sigma * g * eps
    res = x1 - x0                           # target velocity

    # --- 予測速度（cond は x0 を渡す） ---
    vpred = model(xt, t, cond=x0)

    # --- 損失（サブバンド重み対応） ---
    if sb_weight is not None:
        w = sb_weight.to(device).float()
        if w.numel() == 8:
            w = w.repeat_interleave(C)      # [8]→[8*C]
        assert w.numel() == C8, f"sb_weight {w.numel()} != {C8}"
        loss = weighted_mse(vpred, res, w)
    else:
        loss = F.mse_loss(vpred, res)

    # --- サブバンド別ログ ---
    logs = {}
    for i, band in enumerate(SUBBAND_ORDER):  # ['LLL','LLH','LHL','LHH','HLL','HLH','HHL','HHH']
        sl = slice(i * C, (i + 1) * C)
        diff = (vpred[:, sl] - res[:, sl])
        logs[f"band/mse_{band}"] = float(torch.mean(diff ** 2).item())
        logs[f"band/ene_{band}"] = float(torch.mean(res[:, sl] ** 2).item())
        logs[f"band/cos_{band}"] = _cosine(vpred[:, sl], res[:, sl])

    return {"loss": loss, **logs}
