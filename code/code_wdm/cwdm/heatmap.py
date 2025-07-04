import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === パス設定（例） ===
target_path = "/home/a_anami/work/data/result_sample_0605_2D/0610_2/target_00080_000/axial/slice_axial_120.png"
generated_path = "/home/a_anami/work/data/result_sample_0605_2D/0610/sample_00080_000/axial/slice_axial_120.png"
output_path = "/home/a_anami/work/data/result_sample_0605_2D/heatmap/00080_000_axial_120"

# === 画像読み込み ===
target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
generated_img = cv2.imread(generated_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

# サイズ確認
if target_img.shape != generated_img.shape:
    generated_img = cv2.resize(generated_img, (target_img.shape[1], target_img.shape[0]), interpolation=cv2.INTER_LINEAR)

# === 差分計算 ===
diff = np.abs(target_img - generated_img)

# === 正規化（0-255） ===
norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

# === カラーマップ適用 ===
heatmap = cv2.applyColorMap(norm_diff.astype(np.uint8), cv2.COLORMAP_JET)

# === 可視化（オーバーレイも可能） ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Target")
plt.imshow(target_img, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Generated")
plt.imshow(generated_img, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Difference Heatmap")
plt.imshow(heatmap)
plt.axis("off")

plt.tight_layout()
plt.savefig(output_path)
plt.show()
