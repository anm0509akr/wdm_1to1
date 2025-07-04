import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# === 設定 ===
nii_path = "/home/a_anami/work/data/BraTS2023-GLI/validation/BraTS-GLI-00001-000/BraTS-GLI-00001-000-t1c.nii.gz"  # ここを出力画像のパスに変更
output_dir = "/home/a_anami/work/data/result_sample_0605_2D/0610_2/target_00001_000_coronal"  # スライス画像の保存先

# === データ読み込み ===
img_nii = nib.load(nii_path)
img = img_nii.get_fdata()  # shape: (H, W, D) ← 多くの医用画像でこの順

# === 正規化（0〜1にスケーリング）===
img = (img - np.min(img)) / (np.max(img) - np.min(img))

# === 出力フォルダ作成 ===
os.makedirs(output_dir, exist_ok=True)

for z in range(0, img.shape[2], 10):
    slice_img = img[:, :, z]
    #axial(横断)：img[:, :, z]
    #coronal(冠状)：img[:, y, :]
    #sagittal(矢状)：img[x, :, :]
    filename = f"{output_dir}/slice_axial_z{z}.png"
    plt.imsave(filename, slice_img, cmap="gray")
    print(f"保存: {filename}")
