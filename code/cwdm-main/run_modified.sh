#!/bin/bash

# === 1. 設定項目 ===

# --- 一般設定 ---
GPU=0;                    # 使用するGPU
SEED=42;                  # ランダムシード
MODEL='unet';             # 使用するモデル ('unet')
DATASET='brats';          # データセット名 ('brats')

# --- 入出力モダリティの設定 ---
# ここで t1n から t1c への変換を指定します
INPUT_MODALITY='t1n'
OUTPUT_MODALITY='t1c'

# --- 学習済みモデルとデータパスの設定 ---
# ⚠️【重要】ご自身の環境に合わせてパスを確認・修正してください
MODEL_PATH="/home/a_anami/work/code/cwdm-main/runs/Jun27_08-55-49_5bfbe33afd57/checkpoints/brats_100000.pt"
DATA_DIR="/home/a_anami/work/data/BraTS2023-GLI/validation"
OUTPUT_DIR="/home/a_anami/work/data/results/${DATASET}_${MODEL}_${INPUT_MODALITY}_to_${OUTPUT_MODALITY}/"

# === 2. モデルパラメータ (通常は変更不要) ===

# --- U-Netモデル用の詳細設定 ---
# 1入力→1出力用にチャンネル数を設定済みです
CHANNEL_MULT="1,2,2,4,4"
IN_CHANNELS=16 # 8 (ノイズ) + 8 (条件画像)

# === 3. 実行コマンド (ここから下は変更不要) ===

echo "MODEL: WDM (U-Net)"
echo "MODE: sampling (image-to-image translation)"
echo "DATASET: BRATS"
echo "INPUT: ${INPUT_MODALITY}, OUTPUT: ${OUTPUT_MODALITY}"

# sample.py に渡す引数を組み立て
COMMON_ARGS="
--dataset=${DATASET}
--num_channels=64
--class_cond=False
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=1000
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=1
--num_groups=32
--in_channels=${IN_CHANNELS}
--out_channels=8
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=False
--use_freq=False
--predict_xstart=True
"

SAMPLE_ARGS="
--data_dir=${DATA_DIR}
--seed=${SEED}
--image_size=224
--use_fp16=False
--model_path=${MODEL_PATH}
--devices=${GPU}
--output_dir=${OUTPUT_DIR}
--num_samples=1000
--use_ddim=False
--sampling_steps=0
--clip_denoised=True
--input_modality=${INPUT_MODALITY}
--output_modality=${OUTPUT_MODALITY}
"

# Pythonスクリプトを実行
python scripts/sample.py $SAMPLE_ARGS $COMMON_ARGS