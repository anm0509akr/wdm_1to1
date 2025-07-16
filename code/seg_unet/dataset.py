import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

def pad_or_crop_to_size(image, target_size=(128, 128, 128)):
    """
    画像を目標サイズにパディングまたはクロップする関数。
    
    Args:
        image (np.array): (C, H, W, D) のNumpy配列。
        target_size (tuple): 目標の (H, W, D) サイズ。
    
    Returns:
        np.array: サイズ変更後のNumpy配列。
    """
    c, h, w, d = image.shape
    th, tw, td = target_size

    # クロップ
    h_start = max(0, (h - th) // 2)
    w_start = max(0, (w - tw) // 2)
    d_start = max(0, (d - td) // 2)
    
    image = image[:, h_start:h_start+th, w_start:w_start+tw, d_start:d_start+td]
    
    # パディング
    cp, ch, cw, cd = image.shape
    
    pad_h = th - ch
    pad_w = tw - cw
    pad_d = td - cd
    
    pad_h_before = pad_h // 2
    pad_w_before = pad_w // 2
    pad_d_before = pad_d // 2
    
    padded_image = np.pad(
        image,
        (
            (0, 0), # チャンネル方向はパディングしない
            (pad_h_before, pad_h - pad_h_before),
            (pad_w_before, pad_w - pad_w_before),
            (pad_d_before, pad_d - pad_d_before)
        ),
        'constant',
        constant_values=0
    )
    
    return padded_image


class BraTSDataset(Dataset):
    def __init__(self, data_dir, patient_ids):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.modalities = ['t1n', 't1c', 't2f', 't2w']

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        # (前半の処理は変更なし)
        # ...
        patient_id = self.patient_ids[idx]
        patient_dir = os.path.join(self.data_dir, patient_id)

        images = []
        for modality in self.modalities:
            filename = f"{patient_id}-{modality}.nii.gz"
            file_path = os.path.join(patient_dir, filename)
            img = nib.load(file_path).get_fdata(dtype=np.float32)
            images.append(img)

        image_stack = np.stack(images)
        
        seg_filename = f"{patient_id}-seg.nii.gz"
        seg_path = os.path.join(patient_dir, seg_filename)
        label = nib.load(seg_path).get_fdata(dtype=np.float32)

        wt_label = (label > 0).astype(np.float32)
        tc_label = ((label == 1) | (label == 3)).astype(np.float32)
        et_label = (label == 3).astype(np.float32)
        
        label_stack = np.stack([wt_label, tc_label, et_label])
        
        mask = image_stack.sum(0) > 0
        
        for c in range(image_stack.shape[0]):
            channel = image_stack[c]
            mean = channel[mask].mean()
            std = channel[mask].std()
            if std > 0:
                image_stack[c] = (channel - mean) / std
        
        true_points = np.argwhere(mask)
        if true_points.shape[0] > 0:
            min_idx = true_points.min(axis=0)
            max_idx = true_points.max(axis=0)
            image_stack = image_stack[:, min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]
            label_stack = label_stack[:, min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]
        
        # --- ✨ここから修正・追加した部分✨ ---
        # 4. 固定サイズへのパディング/クロッピング
        # これにより、全ての画像のサイズが (128, 128, 128) に統一される
        target_size = (128, 128, 128)
        image_stack = pad_or_crop_to_size(image_stack, target_size)
        label_stack = pad_or_crop_to_size(label_stack, target_size)
        
        # PyTorchのテンソルに変換
        image_tensor = torch.from_numpy(image_stack).float()
        label_tensor = torch.from_numpy(label_stack).float()

        return image_tensor, label_tensor