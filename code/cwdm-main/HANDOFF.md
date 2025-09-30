# Handoff (Wavelet Flow-Matching for MRI t1n→t1c)

## Dataset
- Root (train/val): BraTS2023_split
- Naming: `BraTS-XXXX-XXX-t1n.nii.gz`, `...-t1c.nii.gz`
- Preprocess: resize to 112^3, clip 0.1–99.9 pct, normalize to [-1,1]

## Code layout (key files)
- guided_diffusion/{unet_flow, losses_flow, wavelet_flow_adapter, train_util_flow, bratsloader_flow}.py
- DWT_IDWT/ (pywt-based)
- scripts/{train_flow, sample_flow_infer, eval_synthesis, diagnose_shift, split_dataset_by_patient}.py
- run_flow.sh

## Training
- Iteration-first: max_steps, log_interval, ckpt_interval
- Loss: compute_cfm_loss() 内で DWT/IDWT（**学習で二重DWT禁止**）
- Logs: SAVE_DIR/train_log.csv, SAVE_DIR/tb/

## Inference & Eval
- sample_flow_infer.py: outputs *_pred/_cond/_gt.nii.gz
- eval_synthesis.py: PSNR/SSIM/MAE/NRMSE (+cond baseline)

## Known issues / Next
- Velocity sign/scale: monitor cosine/alpha vs residual
- Try sigma>0, LLL-heavy weights
- Long-run training
