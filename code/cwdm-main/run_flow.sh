#!/usr/bin/env bash
set -euo pipefail

# ======================================================================================
# Wavelet Flow-Matching ランチャー (train / sample / infer)
# ======================================================================================

# ------------------------
# 基本設定（環境変数で上書き可）
# ------------------------
GPU=${GPU:-0}
SEED=${SEED:-42}
MODE=${MODE:-train}        # train | sample | infer

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ------------------------
# データパス
# ------------------------
TRAIN_DIR=${TRAIN_DIR:-/home/a_anami/work/data/BraTS2023_random/training_random}
EVAL_DIR=${EVAL_DIR:-/home/a_anami/work/data/BraTS2023_random/validation_random}

# ------------------------
# モデル / 学習ハイパラ
# ------------------------
BASE_CH=${BASE_CH:-64}
DEPTH=${DEPTH:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-1e-4}
NUM_WORKERS=${NUM_WORKERS:-12}
IMAGE_SIZE=${IMAGE_SIZE:-112}
SIGMA=${SIGMA:-0.0}
SB_WEIGHTS=${SB_WEIGHTS:-" 4,1,1,1,1,1,1,1"}

# iteration-first
MAX_STEPS=${MAX_STEPS:-250000}
LOG_INTERVAL=${LOG_INTERVAL:-1000}
CKPT_INTERVAL=${CKPT_INTERVAL:-50000}
LR_ANNEAL_STEPS=${LR_ANNEAL_STEPS:-250000}
EPOCHS=${EPOCHS:-}   # 空なら未使用

# ------------------------
# 出力 / サンプリング設定
# ------------------------
SAVE_DIR=${SAVE_DIR:-/home/a_anami/work/data/flow_result_2509_low_4}
CKPT=${CKPT:-"$SAVE_DIR/last.pt"}

# sample（.pt）
STEPS=${STEPS:-100}
RHO=${RHO:-1.5}
NUM=${NUM:-4}
OUT_DIR=${OUT_DIR:-"$SAVE_DIR/samples"}

# infer（.nii.gz）
INFER_OUT_DIR=${INFER_OUT_DIR:-"$SAVE_DIR/samples_eval"}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-1}
INFER_NUM_WORKERS=${INFER_NUM_WORKERS:-4}

# 進捗/保存トグル（必要なら環境変数で）
STEP_PROGRESS=${STEP_PROGRESS:-1}   # 1 なら per-step tqdm を出す
SAVE_SOURCE=${SAVE_SOURCE:-1}      # 0 で *_cond.nii.gz を保存しない
SAVE_GT=${SAVE_GT:-1}              # 0 で *_gt.nii.gz を保存しない

# 1 引数で MODE を上書き可: ./run_flow.sh infer
if [[ ${1:-} != "" ]]; then
  MODE="$1"
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONHASHSEED="$SEED"

# tqdm なければ簡易導入
if ! python -c "import tqdm" >/dev/null 2>&1; then
  echo "[setup] Installing tqdm..."
  python -m pip install -q tqdm
fi

printf "\n=========== Wavelet Flow Runner ===========\n"
printf "Mode        : %s\n" "$MODE"
printf "GPU         : %s\n" "$GPU"
printf "Seed        : %s\n" "$SEED"
printf "Train Dir   : %s\n" "$TRAIN_DIR"
printf "Eval Dir    : %s\n" "$EVAL_DIR"
printf "Save Dir    : %s\n" "$SAVE_DIR"
printf "Base Ch     : %s\n" "$BASE_CH"
printf "Depth       : %s\n" "$DEPTH"
printf "Batch Size  : %s\n" "$BATCH_SIZE"
printf "LR          : %s\n" "$LR"
printf "Image Size  : %s\n" "$IMAGE_SIZE"
printf "Sigma       : %s\n" "$SIGMA"
printf "SB Weights  : %s\n" "$SB_WEIGHTS"
[[ -n "${MAX_STEPS}" ]] && printf "Max Steps   : %s\n" "$MAX_STEPS"
[[ -n "${EPOCHS}" ]] && printf "Epochs      : %s\n" "$EPOCHS"
printf "===========================================\n\n"

mkdir -p "$SAVE_DIR"

if [[ "$MODE" == "train" ]]; then
  python scripts/train_flow.py \
    --data_dir "$TRAIN_DIR" \
    --save_dir "$SAVE_DIR" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    --base_ch "$BASE_CH" \
    --depth "$DEPTH" \
    --sigma "$SIGMA" \
    --subband_weight "$SB_WEIGHTS" \
    ${MAX_STEPS:+--max_steps "$MAX_STEPS"} \
    ${LOG_INTERVAL:+--log_interval "$LOG_INTERVAL"} \
    ${CKPT_INTERVAL:+--ckpt_interval "$CKPT_INTERVAL"} \
    ${LR_ANNEAL_STEPS:+--lr_anneal_steps "$LR_ANNEAL_STEPS"} \
    ${EPOCHS:+--epochs "$EPOCHS"}

elif [[ "$MODE" == "sample" ]]; then
  if [[ ! -f "$CKPT" ]]; then
    echo "[ERROR] CKPT not found: $CKPT" >&2
    exit 1
  fi
  mkdir -p "$OUT_DIR"
  python scripts/sample_flow.py \
    --ckpt "$CKPT" \
    --steps "$STEPS" \
    --rho "$RHO" \
    --num "$NUM" \
    --out_dir "$OUT_DIR"

elif [[ "$MODE" == "infer" ]]; then
  if [[ ! -f "$CKPT" ]]; then
    echo "[ERROR] CKPT not found: $CKPT" >&2
    exit 1
  fi
  # infer 依存（nibabel/scipy）チェック
  for mod in nibabel scipy; do
    if ! python -c "import $mod" >/dev/null 2>&1; then
      echo "[setup] Installing $mod..."
      python -m pip install -q "$mod"
    fi
  done

  mkdir -p "$INFER_OUT_DIR"

  # 追加フラグを組み立て
  INFER_FLAGS=""
  [[ -n "$STEP_PROGRESS" ]] && INFER_FLAGS+=" --step_progress"
  [[ "$SAVE_SOURCE" -eq 0 ]] && INFER_FLAGS+=" --no_save_source"
  [[ "$SAVE_GT" -eq 0 ]] && INFER_FLAGS+=" --no_save_gt"

  python scripts/sample_flow_infer.py \
    --data_dir "$EVAL_DIR" \
    --ckpt "$CKPT" \
    --out_dir "$INFER_OUT_DIR" \
    --steps "$STEPS" \
    --rho "$RHO" \
    --image_size "$IMAGE_SIZE" \
    --num_workers "$INFER_NUM_WORKERS" \
    --batch_size "$INFER_BATCH_SIZE" \
    $INFER_FLAGS

else
  echo "[ERROR] Unknown MODE: $MODE (use 'train' or 'sample' or 'infer')" >&2
  exit 1
fi
