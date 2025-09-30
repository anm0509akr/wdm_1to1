# scripts/utils_flow.py
import torch
from typing import Optional


def parse_sb_weight(arg: Optional[str], C8: int) -> torch.Tensor:
    if arg is None:
        return torch.ones(C8)
    vals = [float(x) for x in arg.split(',')]
    if len(vals) == 8:
        rep = C8 // 8
        w = torch.tensor(vals, dtype=torch.float32).repeat_interleave(rep)
    else:
        assert len(vals) == C8, 'Provide 8 weights or per-channel weights'
        w = torch.tensor(vals, dtype=torch.float32)
    return w