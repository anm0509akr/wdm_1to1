# scripts/eval_synthesis.py
# Evaluate conditional MRI synthesis in 3D (PSNR/SSIM/MAE/NRMSE) with YAML config
# and optional baseline evaluation of the conditioning image (cond vs gt).
#
# 優先順位: デフォルト → YAML(--config) → CLI(引数) の順で上書き（CLIが最優先）。
#
# YAML 例 (configs/eval.yaml):
# ---------------------------------------------------
# dir: /home/a_anami/work/data/flow_result_2509/samples_eval
# pattern_pred: "*_pred.nii.gz"
# pattern_gt:   "*_gt.nii.gz"
# pattern_cond: "*_cond.nii.gz"
# csv: /home/a_anami/work/data/flow_result_2509/samples_eval/metrics.csv
# axis: 0          # 0=axial, 1=coronal, 2=sagittal（SSIMのスライス方向）
# data_range: 2.0  # 入出力が[-1,1]なら2.0
# with_cond: true  # cond vs gt も併せて評価
# ---------------------------------------------------
#
# 実行例:
#   pip install pyyaml scikit-image nibabel tqdm
#   python scripts/eval_synthesis.py --config configs/eval.yaml
#   # 一部を CLI で上書き（CLIが優先）
#   python scripts/eval_synthesis.py --config configs/eval.yaml --axis 2 --with_cond

import os, re, argparse, csv, math, sys
from glob import glob
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Optional dependency: scikit-image (SSIM)
try:
    from skimage.metrics import structural_similarity as ssim2d
except Exception:
    ssim2d = None


# -----------------------
# Config helpers
# -----------------------
def _require_yaml():
    try:
        import yaml  # noqa
        return True
    except Exception:
        return False


def load_yaml(path: str) -> dict:
    if not _require_yaml():
        raise RuntimeError("PyYAML が見つかりません。`pip install pyyaml` を実行してください。")
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML のルートは dict である必要があります: {path}")
    return data


def coalesce(*vals, default=None, cast=None):
    for v in vals:
        if v is None:
            continue
        x = v
        if cast is not None and x is not None:
            try:
                x = cast(x)
            except Exception:
                raise ValueError(f"設定値を型変換できません: {v} -> {cast}")
        return x
    return default


# -----------------------
# Metrics
# -----------------------
def mse(a, b):
    d = (a - b).astype(np.float64)
    return float(np.mean(d * d))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def nrmse(a, b, mode="range"):
    rmse = math.sqrt(mse(a, b))
    if mode == "range":
        dr = float(np.max(b) - np.min(b))
        return float(rmse / (dr if dr > 0 else 1.0))
    elif mode == "std":
        s = float(np.std(b))
        return float(rmse / (s if s > 0 else 1.0))
    else:
        m = float(np.mean(np.abs(b)))
        return float(rmse / (m if m > 0 else 1.0))


def psnr(a, b, data_range=2.0):
    m = mse(a, b)
    if m <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(m)


def ssim_volume(axial_vol, axial_gt, data_range=2.0):
    """3D をスライス毎（axis前面）に SSIM 計測し平均。scikit-image が無ければ NaN。"""
    if ssim2d is None:
        return float("nan")
    D = axial_vol.shape[0]
    vals = []
    for z in range(D):
        v = axial_vol[z]
        g = axial_gt[z]
        try:
            s = ssim2d(v, g, data_range=data_range, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        except TypeError:
            s = ssim2d(v, g, data_range=data_range)
        vals.append(s)
    return float(np.mean(vals))


# -----------------------
# Utilities
# -----------------------
def id_from_name(p):
    """BraTS-GLI-01298-000_pred.nii.gz → BraTS-GLI-01298-000"""
    name = os.path.basename(p)
    name = re.sub(r'\.nii(\.gz)?$', '', name)
    name = re.sub(r'_(pred|gt|cond)$', '', name)
    return name


def transpose_for_axis(v, axis: int):
    """axis=0(axial),1(coronal),2(sagittal) に合わせて (D,H,W) にするための転置処理"""
    if axis == 0:
        return v
    elif axis == 1:
        return np.transpose(v, (1, 0, 2))
    else:
        return np.transpose(v, (2, 0, 1))


# -----------------------
# Argparse
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser("Evaluate 3D synthesis (PSNR/SSIM/MAE/NRMSE) with YAML & cond baseline")
    ap.add_argument("--config", type=str, default=None, help="YAML 設定ファイルへのパス")

    # YAML/CLI どちらでも設定可（CLIが最優先）
    ap.add_argument("--dir", type=str, default=None, help="pred/gt/cond が並ぶフォルダ")
    ap.add_argument("--pattern_pred", type=str, default=None, help="例: *_pred.nii.gz")
    ap.add_argument("--pattern_gt", type=str, default=None, help="例: *_gt.nii.gz")
    ap.add_argument("--pattern_cond", type=str, default=None, help="例: *_cond.nii.gz")
    ap.add_argument("--csv", type=str, default=None, help="CSV 出力パス（省略時 dir/metrics.csv）")
    ap.add_argument("--axis", type=int, default=None, help="SSIM のスライス軸（0=axial, 1=coronal, 2=sagittal）")
    ap.add_argument("--data_range", type=float, default=None, help="PSNR/SSIM の data_range（[-1,1]→2.0）")
    ap.add_argument("--with_cond", action="store_true", help="cond vs gt も併せて評価する")
    return ap.parse_args()


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()

    # 1) デフォルト
    cfg = {
        "dir": None,
        "pattern_pred": "*_pred.nii.gz",
        "pattern_gt": "*_gt.nii.gz",
        "pattern_cond": "*_cond.nii.gz",
        "csv": None,                # None -> dir/metrics.csv
        "axis": 0,
        "data_range": 2.0,          # [-1,1] を想定
        "with_cond": False,
    }

    # 2) YAML
    if args.config:
        ycfg = load_yaml(args.config)
        # 別名も許容
        alias = {
            "root": "dir",
            "out_dir": "dir",
            "pred_glob": "pattern_pred",
            "gt_glob": "pattern_gt",
            "cond_glob": "pattern_cond",
            "output_csv": "csv",
        }
        norm_ycfg = {}
        for k, v in (ycfg or {}).items():
            nk = alias.get(k, k)
            norm_ycfg[nk] = v
        for k, v in norm_ycfg.items():
            if v is not None:
                cfg[k] = v

    # 3) CLI（最優先）
    cfg["dir"]          = coalesce(args.dir, cfg.get("dir"), default=None, cast=str)
    cfg["pattern_pred"] = coalesce(args.pattern_pred, cfg.get("pattern_pred"), default="*_pred.nii.gz", cast=str)
    cfg["pattern_gt"]   = coalesce(args.pattern_gt, cfg.get("pattern_gt"), default="*_gt.nii.gz", cast=str)
    cfg["pattern_cond"] = coalesce(args.pattern_cond, cfg.get("pattern_cond"), default="*_cond.nii.gz", cast=str)
    cfg["csv"]          = coalesce(args.csv, cfg.get("csv"), default=None, cast=str)
    cfg["axis"]         = coalesce(args.axis, cfg.get("axis"), default=0, cast=int)
    cfg["data_range"]   = coalesce(args.data_range, cfg.get("data_range"), default=2.0, cast=float)
    # with_cond は bool 化（YAML bool or CLI flag）
    cfg["with_cond"]    = bool(args.with_cond or cfg.get("with_cond"))

    if not cfg["dir"]:
        print("[ERROR] データフォルダが未指定です。--dir か YAML の dir を指定してください。", file=sys.stderr)
        sys.exit(1)

    root = Path(cfg["dir"])
    if not root.exists():
        print(f"[ERROR] データフォルダが存在しません: {root}", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(cfg["csv"]) if cfg["csv"] else root / "metrics.csv"

    preds = sorted(glob(str(root / cfg["pattern_pred"])))
    gts   = sorted(glob(str(root / cfg["pattern_gt"])))
    if len(preds) == 0:
        print(f"[ERROR] pred が見つかりません: {cfg['pattern_pred']}", file=sys.stderr)
        sys.exit(1)
    if len(gts) == 0:
        print(f"[ERROR] gt が見つかりません: {cfg['pattern_gt']}（GT無し評価は未対応）", file=sys.stderr)
        sys.exit(1)

    pred_map = {id_from_name(p): p for p in preds}
    gt_map   = {id_from_name(p): p for p in gts}
    cond_map = {}
    if cfg["with_cond"]:
        conds = sorted(glob(str(root / cfg["pattern_cond"])))
        cond_map = {id_from_name(p): p for p in conds}

    common_ids = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if len(common_ids) == 0:
        print("[ERROR] pred と gt の共通IDがありません。命名規則を確認してください。", file=sys.stderr)
        sys.exit(1)

    print(f"[info] pairs found: {len(common_ids)}")
    rows = []

    for sid in tqdm(common_ids, desc="Eval", unit="case"):
        p_path = pred_map[sid]
        g_path = gt_map[sid]
        pv = nib.load(p_path).get_fdata().astype(np.float32)
        gv = nib.load(g_path).get_fdata().astype(np.float32)

        if pv.shape != gv.shape:
            print(f"[WARN] shape mismatch for {sid}: pred {pv.shape}, gt {gv.shape} → skip")
            continue

        # SSIM のために axis を前面に
        p3 = transpose_for_axis(pv, cfg["axis"])
        g3 = transpose_for_axis(gv, cfg["axis"])

        row = {
            "id": sid,
            "PSNR": psnr(pv, gv, data_range=cfg["data_range"]),
            "SSIM": ssim_volume(p3, g3, data_range=cfg["data_range"]),
            "MAE":  mae(pv, gv),
            "NRMSE(range)": nrmse(pv, gv, mode="range"),
            "MSE":  mse(pv, gv),
            "shape": str(pv.shape),
        }

        # cond のベースラインも評価（存在かつ形状一致のときのみ）
        if cfg["with_cond"] and sid in cond_map:
            cv = nib.load(cond_map[sid]).get_fdata().astype(np.float32)
            if cv.shape == gv.shape:
                c3 = transpose_for_axis(cv, cfg["axis"])
                row.update({
                    "PSNR_cond": psnr(cv, gv, data_range=cfg["data_range"]),
                    "SSIM_cond": ssim_volume(c3, g3, data_range=cfg["data_range"]),
                    "MAE_cond":  mae(cv, gv),
                    "NRMSE(range)_cond": nrmse(cv, gv, mode="range"),
                })
            else:
                row.update({
                    "PSNR_cond": "",
                    "SSIM_cond": "",
                    "MAE_cond": "",
                    "NRMSE(range)_cond": "",
                })

        rows.append(row)

    if len(rows) == 0:
        print("[ERROR] 評価可能なペアがありませんでした。", file=sys.stderr)
        sys.exit(1)

    # 要約（pred）
    def mean_std(rows, key):
        vals = [r.get(key) for r in rows]
        vals = [v for v in vals if v not in (None, "", float("inf")) and np.isfinite(v)]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    psnr_m, psnr_s = mean_std(rows, "PSNR")
    ssim_m, ssim_s = mean_std(rows, "SSIM")
    mae_m,  mae_s  = mean_std(rows, "MAE")
    nrm_m,  nrm_s  = mean_std(rows, "NRMSE(range)")

    # 要約（cond baseline; with_cond のときのみ）
    cond_summary = {}
    if cfg["with_cond"]:
        cond_summary["PSNR_cond_m"], cond_summary["PSNR_cond_s"] = mean_std(rows, "PSNR_cond")
        cond_summary["SSIM_cond_m"], cond_summary["SSIM_cond_s"] = mean_std(rows, "SSIM_cond")
        cond_summary["MAE_cond_m"],  cond_summary["MAE_cond_s"]  = mean_std(rows, "MAE_cond")
        cond_summary["NRMSE_cond_m"],cond_summary["NRMSE_cond_s"]= mean_std(rows, "NRMSE(range)_cond")

    # 表示
    print("\n========== Summary (PRED vs GT) ==========")
    print(f"PSNR (dB):         {psnr_m:.3f} ± {psnr_s:.3f}")
    print(f"SSIM:              {ssim_m:.4f} ± {ssim_s:.4f}")
    print(f"MAE:               {mae_m:.5f} ± {mae_s:.5f}")
    print(f"NRMSE (range):     {nrm_m:.5f} ± {nrm_s:.5f}")
    print("==========================================")
    if cfg["with_cond"]:
        print("---------- Baseline (COND vs GT) ---------")
        print(f"PSNR_cond (dB):    {cond_summary['PSNR_cond_m']:.3f} ± {cond_summary['PSNR_cond_s']:.3f}")
        print(f"SSIM_cond:         {cond_summary['SSIM_cond_m']:.4f} ± {cond_summary['SSIM_cond_s']:.4f}")
        print(f"MAE_cond:          {cond_summary['MAE_cond_m']:.5f} ± {cond_summary['MAE_cond_s']:.5f}")
        print(f"NRMSE_cond(range): {cond_summary['NRMSE_cond_m']:.5f} ± {cond_summary['NRMSE_cond_s']:.5f}")
        print("==========================================\n")
    else:
        print("")

    # CSV 保存（行ごとにキーが異なる可能性があるため union ヘッダを生成）
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = set()
    for r in rows:
        fieldnames.update(r.keys())
    fieldnames = list(sorted(fieldnames))

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        # Summary 行（pred）
        sum_row = {
            "id": "SUMMARY_PRED",
            "PSNR": psnr_m, "SSIM": ssim_m, "MAE": mae_m, "NRMSE(range)": nrm_m
        }
        # Summary 行（cond; 任意）
        if cfg["with_cond"]:
            sum_row.update({
                "PSNR_cond": cond_summary["PSNR_cond_m"],
                "SSIM_cond": cond_summary["SSIM_cond_m"],
                "MAE_cond":  cond_summary["MAE_cond_m"],
                "NRMSE(range)_cond": cond_summary["NRMSE_cond_m"],
            })
        # 補完してから書き出し
        for k in fieldnames:
            sum_row.setdefault(k, "")
        w.writerow(sum_row)

    print(f"[saved] {csv_path}")
    if ssim2d is None:
        print("[note] SSIM は scikit-image 未導入のため NaN です。`pip install scikit-image` で有効になります。")


if __name__ == "__main__":
    main()
