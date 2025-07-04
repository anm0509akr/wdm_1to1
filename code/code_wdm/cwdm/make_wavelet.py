# 必要なライブラリをインストール
# pip install pywt pillow matplotlib numpy

import os
import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt

def dwt2_image(input_path: str, output_dir: str, wavelet: str = 'haar'):
    """
    画像を2次元DWTして、近似係数（cA）と水平・垂直・対角詳細係数（cH, cV, cD）を PNG で保存する。

    Parameters:
        input_path: 入力画像ファイルのパス
        output_dir: 出力ディレクトリ（存在しない場合は自動作成）
        wavelet: 使用するウェーブレット名（'haar', 'db1', 'sym2', ...）
    """
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # 画像読み込み（グレースケール変換）
    img = Image.open(input_path).convert('L')
    arr = np.array(img, dtype=np.float32)

    # 2D DWT 実行
    coeffs2 = pywt.dwt2(arr, wavelet)
    cA, (cH, cV, cD) = coeffs2

    # 各係数を正規化して 0–255 の uint8 に変換
    def norm_uint8(x):
        x_min, x_max = x.min(), x.max()
        return ((x - x_min) / (x_max - x_min) * 255).astype(np.uint8)

    cA_img = Image.fromarray(norm_uint8(cA))
    cH_img = Image.fromarray(norm_uint8(cH))
    cV_img = Image.fromarray(norm_uint8(cV))
    cD_img = Image.fromarray(norm_uint8(cD))

    # 保存
    base = os.path.splitext(os.path.basename(input_path))[0]
    cA_img.save(os.path.join(output_dir, f'{base}_cA.png'))
    cH_img.save(os.path.join(output_dir, f'{base}_cH.png'))
    cV_img.save(os.path.join(output_dir, f'{base}_cV.png'))
    cD_img.save(os.path.join(output_dir, f'{base}_cD.png'))

    print(f'DWT 結果を {output_dir} に保存しました：')
    print(f'  - 近似係数 cA: {base}_cA.png')
    print(f'  - 水平詳細 cH: {base}_cH.png')
    print(f'  - 垂直詳細 cV: {base}_cV.png')
    print(f'  - 対角詳細 cD: {base}_cD.png')

if __name__ == '__main__':
    # 例：input.jpg を DWT して output/ ディレクトリに保存
    output = '/home/a_anami/work/data/result_wavelet'
    input_dir = '/home/a_anami/work/data/result_sample_0605_2D/0610_2/target_00013_001/axial/slice_axial_100.png'
    dwt2_image(input_dir, output, wavelet='haar')
