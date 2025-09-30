# guided_diffusion/losses_flow.py
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def make_t(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch_size, device=device)


def cfm_interpolant(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor,
                    sigma: float = 0.0,
                    eps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    while t.dim() < x0.dim():
        t = t.view(t.size(0), *[1]*(x0.dim()-1))
    xt = (1.0 - t) * x0 + t * x1
    v = x1 - x0
    if sigma > 0.0:
        if eps is None:
            eps = torch.randn_like(x0)
        g = t * (1.0 - t)
        xt = xt + sigma * g * eps
        v = v + sigma * (1.0 - 2.0 * t) * eps
    return xt, v


def weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                 sb_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if sb_weight is None:
        return F.mse_loss(pred, target)
    while sb_weight.dim() < pred.dim():
        sb_weight = sb_weight.view(1, -1, *[1]*(pred.dim()-2))
    return ((pred - target) ** 2 * sb_weight).mean()


def compute_cfm_loss(model, x1: torch.Tensor, cond: Optional[torch.Tensor],
                     sigma: float = 0.0, sb_weight: Optional[torch.Tensor] = None) -> dict:
    device = x1.device
    B = x1.size(0)
    x0 = torch.randn_like(x1)
    t = make_t(B, device)
    xt, vtar = cfm_interpolant(x0, x1, t, sigma)
    vpred = model(xt, t, cond)
    loss = weighted_mse(vpred, vtar, sb_weight)
    return {'loss': loss, 't': t, 'xt': xt.detach(), 'v_pred': vpred.detach(), 'v_tar': vtar.detach()}