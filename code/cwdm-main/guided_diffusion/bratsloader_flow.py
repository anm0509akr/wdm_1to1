# file: guided_diffusion/bratsloader_flow.py
import os
import warnings
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.utils.data as tud
import nibabel as nib
from scipy.ndimage import zoom

__all__ = ["BRATSVolumes", "clip_and_normalize"]

def clip_and_normalize(img: np.ndarray) -> np.ndarray:
    """値をクリップし、[-1,1]に正規化（0.1–99.9%で外れ値抑制）。"""
    lo = float(np.quantile(img, 0.001))
    hi = float(np.quantile(img, 0.999))
    img = np.clip(img, lo, hi)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)  # -> [0,1]
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return (img * 2.0 - 1.0).astype(np.float32)  # -> [-1,1]


class BRATSVolumes(tud.Dataset):
    """
    BraTS 3D ボリュームローダ。
      - 入力: t1n（必須）
      - 目標: t1c（あれば使用）。無いときは zeros を返し、cond['has_gt']=False を付ける。
    返り値:
      target_tensor: [1, D, H, W] in [-1,1]
      cond: {
        'cond_1': [1,D,H,W] (t1n, [-1,1]),
        'subject_id': str,
        'affine': (4,4) np.ndarray,
        'has_gt': bool
      }
    """
    def __init__(self, directory: str, mode: str = "eval", image_size: int = 112):
        super().__init__()
        assert mode in {"train", "eval"}, f"mode must be 'train' or 'eval', got {mode}"
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.target_size = (image_size, image_size, image_size)

        self.seqtypes = ['t1n', 't1c']
        self.database = []  # List[Dict[str,str]]

        if not os.path.isdir(self.directory):
            warnings.warn(f"Directory not found: {self.directory}")
            return

        for root, dirs, files in os.walk(self.directory):
            if dirs or not files:
                continue
            files = sorted(files)

            dp: Dict[str, str] = {"subject_id": os.path.basename(root)}
            for f in files:
                # 例: BraTS-GLI-01298-000-t1n.nii.gz
                name = f.split('.')[0]
                seq = name.split('-')[-1]  # t1n / t1c
                if seq in self.seqtypes:
                    dp[seq] = os.path.join(root, f)

            if self.mode == "train":
                # train は t1n & t1c 必須
                if 't1n' in dp and 't1c' in dp:
                    self.database.append(dp)
            else:
                # eval は t1n 必須（t1c があれば GT で使う）
                if 't1n' in dp:
                    self.database.append(dp)

        if len(self.database) == 0:
            warnings.warn(f"No cases found under {self.directory} (mode={self.mode}).")

    def __len__(self) -> int:
        return len(self.database)

    def load_and_preprocess(self, file_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        NIfTI を読み込み、リサイズ＆正規化して [1, D, H, W] を返す。
        入力 NIfTI は [Z,Y,X] = [D,H,W] 想定。
        """
        nii = nib.load(file_path)
        affine = nii.affine.astype(np.float32)
        vol = nii.get_fdata().astype(np.float32)  # [D,H,W]

        D0, H0, W0 = vol.shape
        Dz, Hy, Wx = self.target_size
        zoom_factors = [Dz / D0, Hy / H0, Wx / W0]
        vol = zoom(vol, zoom_factors, order=1, prefilter=False)  # 線形補間

        vol = clip_and_normalize(vol)  # [-1,1]
        ten = torch.from_numpy(vol).float().unsqueeze(0)  # [1,D,H,W]
        return ten, affine

    def __getitem__(self, idx: int):
        rec = self.database[idx]

        # cond（t1n）は必須
        t1n_tensor, affine = self.load_and_preprocess(rec['t1n'])
        cond: Dict[str, Any] = {
            'cond_1': t1n_tensor,
            'subject_id': rec.get('subject_id', f"case_{idx:05d}"),
            'affine': affine,
        }

        # 目標（t1c）。eval でも存在すれば読む。無ければ zeros + has_gt=False
        if 't1c' in rec and os.path.exists(rec['t1c']):
            tgt_tensor, _ = self.load_and_preprocess(rec['t1c'])
            cond['has_gt'] = True
        else:
            tgt_tensor = torch.zeros_like(t1n_tensor)
            cond['has_gt'] = False

        return tgt_tensor, cond
