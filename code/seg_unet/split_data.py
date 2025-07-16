# split_data.py
import os
import random
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def split_brats_data(data_path, output_path, split_ratio=0.8, seed=42):
    """
    BraTS2023の訓練データを訓練用と検証用に分割する関数

    Args:
        data_path (str or Path): BraTS2023の訓練データセットのパス
        output_path (str or Path): 分割後のデータを保存するディレクトリのパス
        split_ratio (float): 訓練データに割り当てる割合 (例: 0.8は80%)
        seed (int): 再現性のためのランダムシード
    """
    # パスをPathオブジェクトに変換
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # 出力ディレクトリの作成
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 患者（症例）ディレクトリのリストを取得
    patient_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not patient_dirs:
        print(f"エラー: {data_path} に患者ディレクトリが見つかりません。")
        return

    print(f"合計 {len(patient_dirs)} 人の患者データが見つかりました。")

    # 再現性のためにランダムシードを設定し、リストをシャッフル
    random.seed(seed)
    random.shuffle(patient_dirs)

    # 分割点を計算
    split_index = int(len(patient_dirs) * split_ratio)
    
    # 訓練用と検証用にリストを分割
    train_patients = patient_dirs[:split_index]
    val_patients = patient_dirs[split_index:]

    print(f"訓練用に {len(train_patients)} 人、検証用に {len(val_patients)} 人を割り当てます。")

    # 訓練データをコピー
    print("\n訓練データをコピー中...")
    for patient_path in tqdm(train_patients, desc="Copying train data"):
        shutil.copytree(patient_path, train_dir / patient_path.name)

    # 検証データをコピー
    print("\n検証データをコピー中...")
    for patient_path in tqdm(val_patients, desc="Copying validation data"):
        shutil.copytree(patient_path, val_dir / patient_path.name)

    print(f"\nデータ分割が完了しました。出力先: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split BraTS2023 training data into training and validation sets.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the original BraTS2023 training data directory.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the directory where split data will be saved.")
    parser.add_argument('--split_ratio', type=float, default=0.8, help="Ratio of data to be used for training (default: 0.8 for 80/20 split).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    
    split_brats_data(args.data_path, args.output_path, args.split_ratio, args.seed)