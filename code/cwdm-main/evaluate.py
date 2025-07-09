import argparse
import os
import pathlib
import csv
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

# LPIPSライブラリのインポートを試みる
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# FID計算用のInceptionV3モデルをロード
class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception_v3(weights='Inception_V3_Weights.DEFAULT', transform_input=False)
        self.model.fc = nn.Identity()

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return self.model(x)

def calculate_activation_statistics(dataloader, model, device):
    model.eval()
    act_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Activations", leave=False):
            img = batch[0].to(device)
            pred = model(img)
            act_features.append(pred.cpu().numpy())
    
    act_features = np.concatenate(act_features, axis=0)
    mu = np.mean(act_features, axis=0)
    sigma = np.cov(act_features, rowvar=False)
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D MRI generation models and save results to CSV.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing subject subfolders.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the output CSV files.")
    # --- ファイル名を指定する引数を追加 ---
    parser.add_argument("--sample_fname", type=str, default="sample.nii.gz", help="Filename of the generated (sample) image.")
    parser.add_argument("--target_fname", type=str, default="target.nii.gz", help="Filename of the ground truth (target) image.")
    # ------------------------------------
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for calculations.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for FID calculation.")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    individual_results = []
    real_slices, generated_slices = [], []

    subject_dirs = sorted([d for d in pathlib.Path(args.results_dir).iterdir() if d.is_dir()])
    
    if LPIPS_AVAILABLE:
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    else:
        print("\n[Warning] LPIPS library not found. Skipping LPIPS calculation.")

    for subject_dir in tqdm(subject_dirs, desc="Processing Subjects"):
        # --- 引数で指定されたファイル名を使用するように変更 ---
        sample_path = subject_dir / args.sample_fname
        target_path = subject_dir / args.target_fname
        # -----------------------------------------------

        if not (sample_path.exists() and target_path.exists()):
            print(f"Skipping {subject_dir.name}: {args.sample_fname} or {args.target_fname} not found.")
            continue

        target_vol = nib.load(target_path).get_fdata().astype(np.float32)
        sample_vol = nib.load(sample_path).get_fdata().astype(np.float32)

        target_vol_norm = (target_vol - target_vol.min()) / (target_vol.max() - target_vol.min() + 1e-8)
        sample_vol_norm = (sample_vol - sample_vol.min()) / (sample_vol.max() - sample_vol.min() + 1e-8)
        
        mse_score = np.mean((target_vol_norm - sample_vol_norm) ** 2)
        psnr_score = peak_signal_noise_ratio(target_vol_norm, sample_vol_norm, data_range=1.0)
        
        ssim_slice_scores, lpips_slice_scores = [], []
        for i in range(target_vol_norm.shape[2]):
            target_slice = target_vol_norm[:, :, i]
            sample_slice = sample_vol_norm[:, :, i]
            ssim_slice_scores.append(structural_similarity(target_slice, sample_slice, data_range=1.0))
            if LPIPS_AVAILABLE:
                t_target = torch.from_numpy(target_slice).unsqueeze(0).unsqueeze(0).to(device) * 2 - 1
                t_sample = torch.from_numpy(sample_slice).unsqueeze(0).unsqueeze(0).to(device) * 2 - 1
                with torch.no_grad():
                    lpips_slice_scores.append(loss_fn_alex(t_target, t_sample).item())
            real_slices.append(torch.from_numpy(target_slice).unsqueeze(0))
            generated_slices.append(torch.from_numpy(sample_slice).unsqueeze(0))
        
        subject_scores = {
            'subject': subject_dir.name,
            'mse': mse_score,
            'psnr': psnr_score,
            'ssim': np.mean(ssim_slice_scores)
        }
        if LPIPS_AVAILABLE:
            subject_scores['lpips'] = np.mean(lpips_slice_scores) if lpips_slice_scores else 'N/A'
        
        individual_results.append(subject_scores)

    if not individual_results:
        print("\nNo valid subject data found. Please check your --results_dir path and filenames.")
        return

    # (CSVの書き出しと結果表示部分は変更なし)
    individual_csv_path = output_path / "individual_scores.csv"
    print(f"\nSaving individual scores to {individual_csv_path}...")
    try:
        fieldnames = list(individual_results[0].keys())
        with open(individual_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(individual_results)
        print("Done.")
    except Exception as e:
        print(f"Error writing individual scores CSV: {e}")

    print("\nCalculating FID score over all slices...")
    real_loader = DataLoader(TensorDataset(torch.stack(real_slices)), batch_size=args.batch_size)
    generated_loader = DataLoader(TensorDataset(torch.stack(generated_slices)), batch_size=args.batch_size)
    inception_model = InceptionV3().to(device)
    mu_real, sigma_real = calculate_activation_statistics(real_loader, inception_model, device)
    mu_generated, sigma_generated = calculate_activation_statistics(generated_loader, inception_model, device)
    fid_score = calculate_fid(mu_real, sigma_real, mu_generated, sigma_generated)
    
    mse_scores = [res['mse'] for res in individual_results]
    psnr_scores = [res['psnr'] for res in individual_results]
    ssim_scores = [res['ssim'] for res in individual_results]
    
    print("\n--- 総合評価結果 ---")
    print(f"評価対象の被験者数: {len(individual_results)}人")
    print("-" * 20)
    print(f"MSE  : {np.mean(mse_scores):.6f} ± {np.std(mse_scores):.6f}")
    print(f"PSNR : {np.mean(psnr_scores):.4f} ± {np.std(psnr_scores):.4f}")
    print(f"SSIM : {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    if LPIPS_AVAILABLE:
        lpips_scores = [res['lpips'] for res in individual_results if res.get('lpips') != 'N/A']
        if lpips_scores:
            print(f"LPIPS: {np.mean(lpips_scores):.4f} ± {np.std(lpips_scores):.4f}")
    print(f"FID  : {fid_score:.4f}  (calculated over all slices)")
    print("-" * 20)

    summary_csv_path = output_path / "summary_scores.csv"
    print(f"\nSaving summary scores to {summary_csv_path}...")
    summary_data = {
        'mse_mean': np.mean(mse_scores), 'mse_std': np.std(mse_scores),
        'psnr_mean': np.mean(psnr_scores), 'psnr_std': np.std(psnr_scores),
        'ssim_mean': np.mean(ssim_scores), 'ssim_std': np.std(ssim_scores),
        'fid': fid_score
    }
    if LPIPS_AVAILABLE and 'lpips_scores' in locals() and lpips_scores:
        summary_data['lpips_mean'] = np.mean(lpips_scores)
        summary_data['lpips_std'] = np.std(lpips_scores)

    try:
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=summary_data.keys())
            writer.writeheader()
            writer.writerow(summary_data)
        print("Done.")
    except Exception as e:
        print(f"Error writing summary scores CSV: {e}")

if __name__ == '__main__':
    main()