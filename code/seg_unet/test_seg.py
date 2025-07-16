import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import SimpleITK as sitk
import yaml

# 作成済みのスクリプトからインポート
from model import UNet3D
# ✨ ユーザーが作成したdataset.pyからBraTSDatasetをインポート
from dataset import BraTSDataset

# --- テスト用のデータセットクラス ---
# BraTSDatasetを継承し、テストに必要な情報だけを追加で返すように拡張
class TestBraTSDataset(BraTSDataset):
    def __init__(self, data_dir, patient_ids, missing_modalities=None):
        # 親クラスの初期化メソッドを呼び出す
        super().__init__(data_dir, patient_ids)
        self.missing_modalities = missing_modalities if missing_modalities else []
        if self.missing_modalities:
            print(f"以下のモダリティを欠損させます: {self.missing_modalities}")

    def __getitem__(self, idx):
        # 親クラスの__getitem__を呼び出して、前処理済みの画像とラベルを取得
        # ただし、親クラスの実装を少し変更して、テストに必要な情報を取得する
        patient_id = self.patient_ids[idx]
        patient_dir = os.path.join(self.data_dir, patient_id)

        images, original_nifti = [], None
        for i, modality in enumerate(self.modalities):
            nifti_img = nib.load(os.path.join(patient_dir, f"{patient_id}-{modality}.nii.gz"))
            if i == 0: original_nifti = nifti_img
            images.append(nifti_img.get_fdata(dtype=np.float32))
        
        image_stack = np.stack(images)
        original_shape = image_stack.shape[1:]

        # モダリティ欠損の処理
        for m in self.missing_modalities:
            if m in self.modalities:
                image_stack[self.modalities.index(m), ...] = 0

        # 親クラスの前処理ロジックをここに再現
        label = nib.load(os.path.join(patient_dir, f"{patient_id}-seg.nii.gz")).get_fdata(dtype=np.float32)
        label_stack_orig = np.stack([(label > 0), ((label == 1) | (label == 3)), (label == 3)]).astype(np.uint8)

        mask = image_stack.sum(0) > 0
        if mask.sum() > 0:
            for c in range(image_stack.shape[0]):
                channel = image_stack[c]
                mean, std = channel[mask].mean(), channel[mask].std()
                if std > 0: image_stack[c] = (channel - mean) / std
            
            true_points = np.argwhere(mask)
            min_idx = true_points.min(axis=0)
            max_idx = true_points.max(axis=0)
            image_cropped = image_stack[:, min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]
            shape_after_crop = image_cropped.shape[1:]
        else:
            # マスクが空の場合のフォールバック
            min_idx = np.zeros(3, dtype=int)
            shape_after_crop = (0,0,0)
            image_cropped = np.zeros((4,0,0,0))


        from dataset import pad_or_crop_to_size # dataset.pyから関数をインポート
        image_padded = pad_or_crop_to_size(image_cropped)
        image_tensor = torch.from_numpy(image_padded).float()

        return patient_id, image_tensor, label_stack_orig, original_nifti.affine, min_idx, shape_after_crop, original_shape

# --- 評価指標の計算 ---
def calculate_metrics(pred_binary, target_binary):
    pred_itk, target_itk = sitk.GetImageFromArray(pred_binary), sitk.GetImageFromArray(target_binary)
    dice_filter = sitk.LabelOverlapMeasuresImageFilter()
    try:
        dice_filter.Execute(pred_itk, target_itk)
        dice = dice_filter.GetDiceCoefficient()
    except RuntimeError: dice = 0.0
    
    hd_filter = sitk.HausdorffDistanceImageFilter()
    try:
        hd_filter.Execute(pred_itk, target_itk)
        hd95 = hd_filter.GetHausdorffDistance()
    except RuntimeError: hd95 = np.nan
    return dice, hd95

# --- メインのテスト関数 ---
def test(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet3D(in_channels=4, n_classes=3).to(device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()
    print(f"モデルを '{config['model_path']}' から読み込みました。")

    test_patient_ids = [d for d in os.listdir(config['data_dir']) if os.path.isdir(os.path.join(config['data_dir'], d))]
    missing_modalities = config.get('missing_modalities', None)
    test_dataset = TestBraTSDataset(config['data_dir'], test_patient_ids, missing_modalities)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    if os.path.dirname(config['csv_path']): os.makedirs(os.path.dirname(config['csv_path']), exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    results, class_names = [], ['Whole_Tumor', 'Tumor_Core', 'Enhancing_Tumor']
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for patient_id, image, label_orig, affine, crop_start_coords, shape_after_crop, original_shape in pbar:
            patient_id, image = patient_id[0], image.to(device)
            
            logits = model(image)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0]

            final_pred = np.zeros((3,) + tuple(s.item() for s in original_shape), dtype=np.uint8)
            ch, cw, cd = tuple(s.item() for s in shape_after_crop)
            pred_h_start = max(0, (128 - ch) // 2); pred_w_start = max(0, (128 - cw) // 2); pred_d_start = max(0, (128 - cd) // 2)
            crop_h, crop_w, crop_d = min(ch, 128), min(cw, 128), min(cd, 128)
            pred_valid_region = preds[:, pred_h_start:pred_h_start+crop_h, pred_w_start:pred_w_start+crop_w, pred_d_start:pred_d_start+crop_d]
            cs = crop_start_coords.numpy()[0]
            if pred_valid_region.shape[1:] == final_pred[:, cs[0]:cs[0]+crop_h, cs[1]:cs[1]+crop_w, cs[2]:cs[2]+crop_d].shape[1:]:
                final_pred[:, cs[0]:cs[0]+crop_h, cs[1]:cs[1]+crop_w, cs[2]:cs[2]+crop_d] = pred_valid_region

            patient_metrics = {'Patient_ID': patient_id}
            for i, name in enumerate(class_names):
                dice, hd95 = calculate_metrics(final_pred[i], label_orig[0][i].numpy())
                patient_metrics[f'{name}_Dice'], patient_metrics[f'{name}_HD95'] = dice, hd95
            results.append(patient_metrics)
            
            output_seg = (final_pred[0] * 1) + (final_pred[1] * 2) + (final_pred[2] * 4)
            output_seg[output_seg == 3] = 1; output_seg[output_seg > 3] = 3
            nib.save(nib.Nifti1Image(output_seg.astype(np.uint8), affine[0].numpy()), os.path.join(config['output_dir'], f'{patient_id}_seg_pred.nii.gz'))

    df = pd.DataFrame(results)
    mean_metrics = df.mean(numeric_only=True); mean_metrics['Patient_ID'] = 'Average'
    df = pd.concat([df, pd.DataFrame([mean_metrics])], ignore_index=True)
    df.to_csv(config['csv_path'], index=False)
    print(f"\n評価結果を '{config['csv_path']}' に保存しました。")
    print("\n--- 平均スコア ---")
    print(df.tail(1).to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D U-Net Segmentation - Test Script")
    parser.add_argument('--config', type=str, required=True, help="設定ファイル(.yaml)へのパス")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'gpu_id' in config and config['gpu_id'] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
        print(f"環境変数 CUDA_VISIBLE_DEVICES を '{config['gpu_id']}' に設定しました。")
    
    test(config)