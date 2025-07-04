import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# === 設定 ===
nii_path = "/home/a_anami/work/data/BraTS2023-GLI/validation/BraTS-GLI-01670-000/BraTS-GLI-01670-000-t1c.nii.gz"
output_base_dir = "/home/a_anami/work/data/result_sample_0605_2D/0610_2/target_01670_000"

# === データ読み込み ===
img_nii = nib.load(nii_path)
img = img_nii.get_fdata()  # shape: (H, W, D)

# === 正規化（0〜1にスケーリング）===
img = (img - np.min(img)) / (np.max(img) - np.min(img))

# === 保存関数 ===
def save_slices(img, direction, axis, step=10):
    output_dir = os.path.join(output_base_dir, direction)
    os.makedirs(output_dir, exist_ok=True)
    num_slices = img.shape[axis]
    
    for i in range(0, num_slices, step):
        if direction == "axial":
            slice_img = img[:, :, i]
        elif direction == "coronal":
            slice_img = img[:, i, :]
        elif direction == "sagittal":
            slice_img = img[i, :, :]
        else:
            continue
        
        filename = os.path.join(output_dir, f"slice_{direction}_{i}.png")
        plt.imsave(filename, slice_img.T, cmap="gray")  # 転置して正しい表示方向に
        print(f"保存: {filename}")

# === 各方向のスライス保存 ===
save_slices(img, direction="axial", axis=2)
save_slices(img, direction="coronal", axis=1)
save_slices(img, direction="sagittal", axis=0)
