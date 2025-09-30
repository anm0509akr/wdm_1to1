# scripts/sample_flow.py
# Debug-only unconditional sampler (saves .pt). For NIfTI, use sample_flow_infer.py.
import os
import math
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
from torch import nn
from tqdm import tqdm

# our modules
from guided_diffusion.unet_flow import UNet3DFlow
from guided_diffusion.wavelet_flow_adapter import idwt3d  # optional: to save recon too


def load_ckpt(ckpt_path: str, model: nn.Module, map_location: str = "cuda"):
    obj = torch.load(ckpt_path, map_location=map_location)
    # try common layouts
    state = None
    for k in ("model_ema", "ema", "model", "state_dict"):
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            state = obj[k]
            break
    if state is None and isinstance(obj, dict):
        # flat dict?
        state = {k.replace("module.", ""): v for k, v in obj.items() if hasattr(v, "shape")}

    if state is None:
        raise RuntimeError(f"Unrecognized checkpoint format: keys={list(obj.keys()) if isinstance(obj, dict) else type(obj)}")
    # strip potential DistributedDataParallel prefixes
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def make_t_schedule(steps: int, rho: float = 1.5) -> torch.Tensor:
    """
    CFM用の単純降順スケジュール t_0=1 → t_{N-1}=0.
    rho で“前半/後半の密度”を調整（rho>1で後半が細かくなる）。
    """
    if steps < 2:
        return torch.tensor([1.0, 0.0], dtype=torch.float32)
    s = torch.linspace(0.0, 1.0, steps, dtype=torch.float32)  # 0..1
    t = (1.0 - s.pow(rho)).clamp(0.0, 1.0)                    # 1..0
    # 確実に端点を入れる
    t[0] = 1.0
    t[-1] = 0.0
    return t


@torch.no_grad()
def sample_unconditional(
    model: UNet3DFlow,
    shape: torch.Size,           # [B, 8*C, D2, H2, W2]
    steps: int = 40,
    rho: float = 1.5,
    device: torch.device = torch.device("cuda"),
    cond_ch: int = 0,
):
    """
    CFMの決定論的オイラー積分（x_{t-dt} = x_t - v_theta(x_t,t,cond)*dt）
    """
    B, C8, D2, H2, W2 = shape
    x = torch.randn(shape, device=device)

    # 条件なし（学習が条件付きでも、ゼロ条件で回すデバッグサンプリング）
    cond = torch.zeros(B, cond_ch, D2, H2, W2, device=device) if cond_ch > 0 else None

    t_sched = make_t_schedule(steps=steps, rho=rho).to(device)  # [steps]
    for i in tqdm(range(len(t_sched)-1), desc="Sampling", leave=False):
        t_now = t_sched[i].expand(B)
        t_next = t_sched[i+1].expand(B)
        dt = (t_next - t_now)  # negative
        v = model(x, t_now, cond)  # predict velocity in wavelet space
        x = x + v * dt[:, None, None, None, None]  # x_{t+dt}
    return x  # final at t=0 (predicted wavelet coefficients)


def main():
    parser = argparse.ArgumentParser("Wavelet Flow Debug Sampler (.pt)")
    parser.add_argument("--ckpt", required=True, help="path to checkpoint .pt")
    parser.add_argument("--out_dir", required=True, help="directory to save .pt samples")
    parser.add_argument("--num", type=int, default=4, help="number of samples")
    parser.add_argument("--steps", type=int, default=40, help="sampling steps")
    parser.add_argument("--rho", type=float, default=1.5, help="t-schedule rho (>1 denser near t~0)")
    parser.add_argument("--image_size", type=int, default=112, help="must match training (even)")
    parser.add_argument("--base_ch", type=int, default=64, help="fallback if ckpt lacks config")
    parser.add_argument("--depth", type=int, default=4, help="fallback if ckpt lacks config")
    parser.add_argument("--in_ch", type=int, default=8, help="model input channels (8*C). default 8 for C=1")
    parser.add_argument("--cond_ch", type=int, default=8, help="condition channels; 0 if unconditional model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_recon", action="store_true", help="also save inverse-DWT reconstruction (tensor)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデル構築（ckptにconfigが埋め込まれていれば、本来はそれを読むのがベスト）
    model = UNet3DFlow(
        in_ch=args.in_ch,
        cond_ch=args.cond_ch,
        base_ch=args.base_ch,
        depth=args.depth,
    ).to(device).eval()

    missing, unexpected = load_ckpt(args.ckpt, model, map_location=device.type)
    if missing or unexpected:
        print(f"[warn] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    # 出力ディレクトリ
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 係数空間の空間サイズ（DWTで半分）
    D2 = H2 = W2 = args.image_size // 2
    if args.image_size % 2 != 0:
        raise ValueError("--image_size must be even for DWT/IDWT")

    # 生成ループ
    meta: Dict[str, Any] = {
        "ckpt": os.path.abspath(args.ckpt),
        "steps": args.steps,
        "rho": args.rho,
        "seed": args.seed,
        "image_size": args.image_size,
        "in_ch": args.in_ch,
        "cond_ch": args.cond_ch,
        "base_ch": args.base_ch,
        "depth": args.depth,
        "save_recon": bool(args.save_recon),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    for i in range(args.num):
        y = sample_unconditional(
            model,
            shape=torch.Size([1, args.in_ch, D2, H2, W2]),
            steps=args.steps,
            rho=args.rho,
            device=device,
            cond_ch=args.cond_ch,
        )
        save_obj = {"y_wavelet": y.detach().cpu()}
        if args.save_recon:
            try:
                xr = idwt3d(y)  # -> [1, C, D, H, W], where C = in_ch//8
                save_obj["x_recon"] = xr.detach().cpu()
            except Exception as e:
                save_obj["recon_error"] = str(e)

        torch.save(save_obj, out_dir / f"sample_{i:03d}.pt")

    print(f"[done] saved {args.num} tensor samples to: {out_dir}")


if __name__ == "__main__":
    main()
