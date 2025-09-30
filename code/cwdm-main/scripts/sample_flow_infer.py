# scripts/sample_flow_infer.py  （置き換え版・ポイントは x 初期値と sign=auto）
import argparse, os, json, math, warnings
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import nibabel as nib

from guided_diffusion.bratsloader_flow import BRATSVolumes
from guided_diffusion.unet_flow import UNet3DFlow
from guided_diffusion.wavelet_flow_adapter import dwt3d, idwt3d, make_t_schedule

def heun_step(model, x, c, t1, t2, dt, apply_sign):
    with torch.no_grad():
        v1 = apply_sign(model(x, t1.expand(x.size(0)), cond=c))
        x_euler = x + dt * v1
        v2 = apply_sign(model(x_euler, t2.expand(x.size(0)), cond=c))
        x = x + dt * 0.5 * (v1 + v2)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--rho", type=float, default=1.5)
    ap.add_argument("--image_size", type=int, default=112)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--base_ch", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--sign", type=str, default="auto", choices=["plus","minus","auto"],
                    help="速度の符号。auto は t=0 の1回だけ誤差が小さい方を選ぶ")
    ap.add_argument(
    "--step_progress", action="store_true", default=False,
    help="各症例の中で、時間ステップごとの tqdm を表示する"
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ---- data
    ds = BRATSVolumes(args.data_dir, mode="eval", image_size=args.image_size)
    if len(ds) == 0:
        warnings.warn(f"No cases found under {args.data_dir} (mode=eval).")
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # ---- model & ckpt（EMA優先）
    model = UNet3DFlow(in_ch=8, cond_ch=8, base_ch=args.base_ch, depth=args.depth).to(device).eval()
    obj = torch.load(args.ckpt, map_location="cpu")
    sd  = obj.get("model_ema") or obj.get("model") or obj.get("state_dict") or obj
    sd  = {k.replace("module.",""): v for k,v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"[warn] missing={len(miss)} unexpected={len(unexp)}")
    meta = {
        "ckpt": args.ckpt,
        "step": int(obj.get("step") or -1),
        "base_ch": args.base_ch, "depth": args.depth,
        "steps": args.steps, "rho": args.rho, "image_size": args.image_size,
        "sign": args.sign,
    }
    with open(os.path.join(args.out_dir, "infer_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- time schedule (0 -> 1)
    t_sched = make_t_schedule(args.steps, args.rho, device=device)  # [S]
    # Heun: consecutive pairs (t1,t2)
    t_pairs = list(zip(t_sched[:-1], t_sched[1:]))

    # ---- total progress bar
    pbar = tqdm(total=len(ds), desc="Sampling (cases)", unit="case")

    for i, (tgt, cond) in enumerate(loader):
        # cond / tgt: [B,1,D,H,W] in [-1,1]
        affine = cond["affine"][0].numpy() if "affine" in cond else np.eye(4, dtype=np.float32)
        sid = cond.get("subject_id", [f"case_{i:05d}"])[0]

        cond_vol = cond["cond_1"].to(device)  # [B,1,D,H,W]
        B, _, D, H, W = cond_vol.shape
        cond_w = dwt3d(cond_vol)              # [B,8,D/2,H/2,W/2]

        # ---- 初期値は cond の小波
        x = cond_w.clone()

        # ---- 符号の適用関数を用意
        def apply_plus(v):  return v
        def apply_minus(v): return -v

        if args.sign == "plus":
            apply_sign = apply_plus
        elif args.sign == "minus":
            apply_sign = apply_minus
        else:
            # auto: t=0 で +v / -v を一度だけ比較して選ぶ（GTがあれば x1 で、なければ x 変化量の小さい方）
            t0 = torch.zeros(1, device=device)
            with torch.no_grad():
                v = model(x, t0.expand(B), cond=cond_w)
            if isinstance(tgt, torch.Tensor) and tgt.numel() > 0:
                x1 = dwt3d(tgt.to(device))
                err_plus  = (x + 0.01*v - x1).pow(2).mean()
                err_minus = (x - 0.01*v - x1).pow(2).mean()
                apply_sign = apply_plus if err_plus <= err_minus else apply_minus
            else:
                # GTが無いときは conservative に「変化量が小さい方」
                delta_plus  = (0.01*v).pow(2).mean()
                delta_minus = (-0.01*v).pow(2).mean()
                apply_sign = apply_plus if delta_plus <= delta_minus else apply_minus
        inner = tqdm(t_pairs, leave=False, desc=f"{sid}", unit="step") if args.step_progress else t_pairs
        # ---- Heun integration
        for (t1, t2) in inner:
            dt = (t2 - t1)
            x = heun_step(model, x, cond_w, t1, t2, dt, apply_sign)

        # ---- reconstruct & save
        pred_vol = idwt3d(x).clamp_(-1, 1)      # [B,1,D,H,W]
        pred = pred_vol[0,0].detach().cpu().numpy().astype(np.float32)

        # save pred / cond / gt（GT があれば）
        out_pred = os.path.join(args.out_dir, f"{sid}_pred.nii.gz")
        nib.save(nib.Nifti1Image(pred, affine), out_pred)

        out_cond = os.path.join(args.out_dir, f"{sid}_cond.nii.gz")
        nib.save(nib.Nifti1Image(cond_vol[0,0].detach().cpu().numpy().astype(np.float32), affine), out_cond)

        save_gt = bool(cond.get("has_gt", False))
        if save_gt:
            gt_np = tgt[0,0].detach().cpu().numpy().astype(np.float32)
            # 念のため all-zero を回避
            if np.any(gt_np != 0.0):
                out_gt = os.path.join(args.out_dir, f"{sid}_gt.nii.gz")
                nib.save(nib.Nifti1Image(gt_np, affine), out_gt)

        pbar.update(1)

    pbar.close()
    print(f"[done] saved NIfTI(s) to: {args.out_dir}")

if __name__ == "__main__":
    main()
