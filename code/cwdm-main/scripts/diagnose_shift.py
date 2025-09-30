# scripts/diagnose_shift.py
import os, glob, csv, math
import numpy as np
import nibabel as nib
import torch

from guided_diffusion.bratsloader_flow import clip_and_normalize
from guided_diffusion.wavelet_flow_adapter import dwt3d

def find_cases(root):
    # BraTS-XXX-YYY-ZZZ-t1n.nii.gz を探す
    paths = sorted(glob.glob(os.path.join(root, "**", "*-t1n.nii.gz"), recursive=True))
    cases = []
    for ptn in paths:
        subj = os.path.basename(os.path.dirname(ptn))
        ptc = ptn.replace("-t1n.nii.gz", "-t1c.nii.gz")
        has_t1c = os.path.exists(ptc)
        cases.append((subj, ptn, ptc if has_t1c else None))
    return cases

def raw_stats(vol):
    nz = (vol!=0)
    q = np.percentile(vol[nz] if nz.any() else vol, [0.1, 1, 50, 99, 99.9]).astype(np.float32)
    return dict(
        mean=float(vol.mean()), std=float(vol.std()),
        p001=float(q[0]), p1=float(q[1]), p50=float(q[2]), p99=float(q[3]), p999=float(q[4]),
        nonzero=float(nz.mean())
    )

def preprocess_to_tensor(pt, target_size=(112,112,112)):
    nii = nib.load(pt)
    aff = nii.affine
    vol = nii.get_fdata().astype(np.float32)  # [D,H,W] を想定（nibはZ,Y,X＝D,H,W）
    D0,H0,W0 = vol.shape
    Dz,Hy,Wx = target_size
    # ここでは学習と同じ ndimage.zoom を使わず、評価コスト削減のためスキップ可能
    # 正確に合わせたい場合は scipy.ndimage.zoom を使う（学習と同一に）
    try:
        from scipy.ndimage import zoom
        vol = zoom(vol, [Dz/D0, Hy/H0, Wx/W0], order=1, prefilter=False)
    except Exception:
        pass
    vol = clip_and_normalize(vol)  # [-1,1]
    ten = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)  # [B=1, C=1, D,H,W]
    return ten, aff

def wavelet_stats(t):
    # t: [1,1,D,H,W] in [-1,1]
    with torch.no_grad():
        w = dwt3d(t)  # [1, 8, D/2, H/2, W/2]
    bands = ["LLL","LLH","LHL","LHH","HLL","HLH","HHL","HHH"]
    out = {}
    total_energy = float((w**2).mean())
    for i,b in enumerate(bands):
        wi = w[:, i]
        out[f"{b}_mean"] = float(wi.mean())
        out[f"{b}_std"]  = float(wi.std())
        e = float((wi**2).mean())
        out[f"{b}_energy"] = e
        out[f"{b}_energy_frac"] = (e / (total_energy + 1e-12)) if total_energy>0 else 0.0
    out["wavelet_energy_total"] = total_energy
    return out

def one_split(root, split_name, writer):
    cases = find_cases(root)
    if not cases:
        print(f"[warn] no cases under {root}")
        return []

    rows = []
    for subj, ptn, ptc in cases:
        nii = nib.load(ptn)
        zooms = tuple(float(z) for z in nii.header.get_zooms()[:3])
        orient = "".join(nib.aff2axcodes(nii.affine))
        shape = tuple(int(s) for s in nii.shape[:3])

        # raw stats（前処理前）
        vol = nii.get_fdata().astype(np.float32)
        rs = raw_stats(vol)

        # preprocessed stats & wavelet stats（学習と同じ流れ）
        t, _ = preprocess_to_tensor(ptn)
        ps = dict(pp_mean=float(t.mean()), pp_std=float(t.std()),
                  pp_min=float(t.min()),  pp_max=float(t.max()))
        ws = wavelet_stats(t)

        row = {
            "split": split_name,
            "subject_id": subj,
            "path_t1n": ptn,
            "has_t1c": int(ptc is not None),
            "shape_D": shape[0], "shape_H": shape[1], "shape_W": shape[2],
            "zoom_D": zooms[0], "zoom_H": zooms[1], "zoom_W": zooms[2],
            "orient": orient,
            **rs, **ps, **ws
        }
        writer.writerow(row)
        rows.append(row)
    return rows

def summarize(rows, split):
    # 簡易集計（平均±標準偏差）
    import statistics as st
    def stat(keys):
        vals = [r[k] for r in rows if r["split"]==split]
        return (float(st.mean(vals)), float(st.pstdev(vals))) if vals else (math.nan, math.nan)

    keys = ["p001","p1","p50","p99","p999","pp_mean","pp_std",
            "LLL_energy_frac","LLH_energy_frac","LHL_energy_frac","LHH_energy_frac",
            "HLL_energy_frac","HLH_energy_frac","HHL_energy_frac","HHH_energy_frac"]
    print(f"\n[{split}] summary (mean±std)")
    for k in keys:
        m,s = stat(k)
        print(f"  {k:>16}: {m:.4f} ± {s:.4f}")

if __name__ == "__main__":
    import argparse, pathlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--test_root",  required=True)
    ap.add_argument("--out_csv", default="shift_summary.csv")
    args = ap.parse_args()

    outp = pathlib.Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split","subject_id","path_t1n","has_t1c","shape_D","shape_H","shape_W",
        "zoom_D","zoom_H","zoom_W","orient",
        "mean","std","p001","p1","p50","p99","p999","nonzero",
        "pp_mean","pp_std","pp_min","pp_max",
        "LLL_mean","LLL_std","LLL_energy","LLL_energy_frac",
        "LLH_mean","LLH_std","LLH_energy","LLH_energy_frac",
        "LHL_mean","LHL_std","LHL_energy","LHL_energy_frac",
        "LHH_mean","LHH_std","LHH_energy","LHH_energy_frac",
        "HLL_mean","HLL_std","HLL_energy","HLL_energy_frac",
        "HLH_mean","HLH_std","HLH_energy","HLH_energy_frac",
        "HHL_mean","HHL_std","HHL_energy","HHL_energy_frac",
        "HHH_mean","HHH_std","HHH_energy","HHH_energy_frac",
        "wavelet_energy_total"
    ]

    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        train_rows = one_split(args.train_root, "train", w)
        test_rows  = one_split(args.test_root,  "test",  w)

    # 簡易サマリを標準出力に
    all_rows = []
    import csv as _csv
    with open(outp, "r") as fr:
        r = _csv.DictReader(fr)
        for row in r:
            # 型キャスト
            for k in fieldnames:
                if k in ("split","subject_id","path_t1n","orient"): continue
                try:
                    row[k] = float(row[k])
                except Exception:
                    pass
            all_rows.append(row)
    summarize(all_rows, "train")
    summarize(all_rows, "test")
    print(f"\n[saved] {args.out_csv}")
