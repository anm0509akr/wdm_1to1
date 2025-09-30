# scripts/train_flow.py
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Dataset
try:
    from guided_diffusion.bratsloader_flow import BRATSVolumes as Dataset
except Exception:
    Dataset = None

from guided_diffusion.unet_flow import UNet3DFlow
from guided_diffusion.train_util_flow import TrainLoopFlow, TrainConfig
from scripts.utils_flow import parse_sb_weight


def build_loader(data_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    """
    BRATSVolumes から空間ドメインのテンソルをそのまま取り出して返す。
    ここでは DWT はしない（loss 側で DWT するため）。
    """
    if Dataset is None:
        raise ImportError("guided_diffusion.bratsloader_flow.BRATSVolumes が見つかりません。")
    ds = Dataset(data_dir, mode="train")

    def collate(samples):
        # samples: List[Tuple[target_tensor, cond_dict]]
        #   - target_tensor: [1, D, H, W] in [-1, 1]
        #   - cond_dict['cond_1']: [1, D, H, W] in [-1, 1]
        xs, ys = [], []
        for tgt, cond in samples:
            xs.append(tgt)  # 空間 [1,D,H,W]
            if isinstance(cond, dict):
                ys.append(cond["cond_1"])  # 空間 [1,D,H,W]
            elif torch.is_tensor(cond):
                ys.append(cond)
            else:
                raise TypeError(f"Unsupported cond type: {type(cond)}")
        return {
            "target": torch.stack(xs, 0),   # [B, 1, D, H, W]
            "cond":   torch.stack(ys, 0),   # [B, 1, D, H, W]
        }

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="runs_flow")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)

    # iteration-first（推奨）
    ap.add_argument("--max_steps", type=int, default=250000, help="総ステップ数（推奨）")
    ap.add_argument("--log_interval", type=int, default=100)
    ap.add_argument("--ckpt_interval", type=int, default=10000)
    ap.add_argument("--lr_anneal_steps", type=int, default=0)

    # 互換：max_steps 未指定時のみ使用
    ap.add_argument("--epochs", type=int, default=0)

    # モデル/学習
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--base_ch", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.0)

    # サブバンド重み（例: "4,1,1,1,1,1,1,1"）
    ap.add_argument("--subband_weight", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_loader(args.data_dir, args.batch_size, args.num_workers)

    # 1バッチだけ取り出してチャンネル数を確定（空間 ch → 波形 ch = 8 × 空間 ch）
    sample = next(iter(loader))
    x_sp = sample["target"]          # [B, C_sp, D, H, W]（通常 C_sp=1）
    c_sp = x_sp.size(1)              # 空間チャンネル数
    in_ch = c_sp * 8                 # DWT で 8 倍
    cond_sp = sample.get("cond")
    cond_c_sp = cond_sp.size(1) if cond_sp is not None else 0
    cond_ch = cond_c_sp * 8

    # モデルと最適化
    model = UNet3DFlow(in_ch=in_ch, cond_ch=cond_ch, base_ch=args.base_ch, depth=args.depth)
    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    # サブバンド重みをパース（長さ 8 or 8*C を許容）
    sbw = parse_sb_weight(args.subband_weight, in_ch)  # in_ch=8*C_sp と合うように

    loop = TrainLoopFlow(
        model,
        opt,
        loader,
        device,
        TrainConfig(
            save_dir=args.save_dir,
            lr=args.lr,
            max_steps=args.max_steps,
            log_interval=args.log_interval,
            ckpt_interval=args.ckpt_interval,
            lr_anneal_steps=args.lr_anneal_steps,
            epochs=args.epochs,
            sigma=args.sigma,
            subband_weight=sbw,
        ),
    )
    loop.run(args.save_dir)


if __name__ == "__main__":
    main()
