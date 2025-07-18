#!/bin/bash

# === 1. 設定項目 ===
GPU=1
SEED=42
# 💡 ここでステップ数を自由に設定できます
ODE_STEPS=100

# --- パス設定 ---
MODEL_PATH="/home/a_anami/work/code/cwdm-main/runs/Jul18_07-27-49_5bfbe33afd57/model_000000.pt"
DATA_DIR="/home/a_anami/work/data/BraTS2023_split/validation"
OUTPUT_DIR="/home/a_anami/work/data/results_flow_matching_0718/"

# === 2. 実行コマンド ===
echo "--- Starting Flow-Matching Sampling ---"
echo "Model: ${MODEL_PATH}"
echo "Steps: ${ODE_STEPS}" # ログに表示
echo "Output: ${OUTPUT_DIR}"

python scripts/sample.py \
    --model_path="${MODEL_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --data_dir="${DATA_DIR}" \
    --batch_size=1 \
    --devices=${GPU} \
    --seed=${SEED} \
    --ode_steps=${ODE_STEPS} # 💡 ここでステップ数を渡します

echo "--- Sampling Finished ---"