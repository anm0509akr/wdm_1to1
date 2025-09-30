# scripts/train_flow.py
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from guided_diffusion.bratsloader_flow import BRATSVolumes as Dataset
except Exception:
    Dataset = None

from guided_diffusion.unet_flow import UNet3DFlow
from guided_diffusion.train_util_flow import TrainLoopFlow, TrainConfig
from guided_diffusion.wavelet_flow_adapter import dwt3d, SUBBAND_ORDER
from scripts.utils_flow import parse_sb_weight


def build_loader(data_dir: str, batch_size: int, num_workers: int):
    if Dataset is None:
        raise ImportError('guided_diffusion.bratsloader_flow.BRATSVolumes が見つかりません。')
    # image_size は bratsloader_flow 側のデフォルト(112)を使用
    ds = Dataset(data_dir, mode='train')

    def collate(samples):
        # samples: List[Tuple[target_tensor, cond_dict]]
        xw_list, yw_list = [], []
        for tgt, cond in samples:
            # tgt, cond['cond_1']: [1, D, H, W] in [-1,1]
            xw = dwt3d(tgt.unsqueeze(0)).squeeze(0)              # [8*C, D/2, H/2, W/2]
            yw = dwt3d(cond['cond_1'].unsqueeze(0)).squeeze(0)   # 同上
            xw_list.append(xw)
            yw_list.append(yw)
        return {
            'target': torch.stack(xw_list, 0),
            'cond':   torch.stack(yw_list, 0),
        }

    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--save_dir', type=str, default='runs_flow')
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-4)
    # iteration-first
    ap.add_argument('--max_steps', type=int, default=250000, help='総ステップ数（推奨）')
    ap.add_argument('--log_interval', type=int, default=100)
    ap.add_argument('--ckpt_interval', type=int, default=10000)
    ap.add_argument('--lr_anneal_steps', type=int, default=0)
    # 互換用（未指定なら 0 のままで無視される）
    ap.add_argument('--epochs', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--base_ch', type=int, default=64)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--sigma', type=float, default=0.0)
    ap.add_argument('--subband_weight', type=str, default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = build_loader(args.data_dir, args.batch_size, args.num_workers)

    sample = next(iter(loader))
    xw = sample['target']
    C8 = xw.size(1)
    cond = sample.get('cond')
    cond_ch = 0 if cond is None else cond.size(1)

    model = UNet3DFlow(in_ch=C8, cond_ch=cond_ch, base_ch=args.base_ch, depth=args.depth)
    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    sbw = parse_sb_weight(args.subband_weight, C8)

    loop = TrainLoopFlow(
        model, opt, loader, device,
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
        )
    )
    loop.run(args.save_dir)


if __name__ == '__main__':
    main()