# general settings
GPU=0;                      # gpu to use
SEED=42;                    # randomness seed
CHANNELS=64;                # number of model base channels
MODE='train';               # train, sample
DATASET='brats';            # brats
MODEL='unet';               # 'unet'
CONTR='t1c'                 # contrast to be generated

# settings for sampling/inference
ITERATIONS=1200;            # training iteration checkpoint
RUN_DIR="/home/a_anami/work/data"; # tensorboard dir for evaluation

# detailed settings (no need to change for reproducing)
if [[ $MODEL == 'unet' ]]; then
  echo "MODEL: Flow-Matching (U-Net)"; # Changed the name for clarity
  CHANNEL_MULT=1,2,2,4,4;
  ADDITIVE_SKIP=False;
  BATCH_SIZE=1;
  IMAGE_SIZE=224;
  IN_CHANNELS=16;
  # NOISE_SCHED='linear'; # No longer needed
else
  echo "MODEL TYPE NOT FOUND -> Check the supported configurations again";
fi

# Set data directories based on mode
if [[ $MODE == 'train' ]]; then
  echo "MODE: training";
  DATA_DIR=/home/a_anami/work/data/BraTS2023_split/training;

elif [[ $MODE == 'sample' ]]; then
  BATCH_SIZE=1;
  echo "MODE: sampling (image-to-image translation)";
  DATA_DIR=/home/a_anami/work/data/BraTS2023_split/validation;
fi

## --- ⬇️ MAJOR CHANGES HERE ⬇️ ---

# These are the arguments that are still relevant for the U-Net model itself
COMMON="
--dataset=${DATASET}
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=${IN_CHANNELS}
--out_channels=8
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=False
"

# Training arguments are mostly unchanged
TRAIN="
--data_dir=${DATA_DIR}
--resume_checkpoint=
--resume_step=0
--image_size=${IMAGE_SIZE}
--use_fp16=False
--lr=1e-5
--save_interval=50000
--num_workers=12
--devices=${GPU}
--lr_anneal_steps=200000
"

# Sampling arguments are simplified. Diffusion-specific ones are removed.
SAMPLE="
--data_dir=${DATA_DIR}
--seed=${SEED}
--image_size=${IMAGE_SIZE}
--use_fp16=False
--model_path=/home/a_anami/work/code/cwdm-main/runs/Jun27_08-55-49_5bfbe33afd57/checkpoints/brats_100000.pt
--devices=${GPU}
--output_dir=/home/a_anami/work/data/results/${DATASET}_${MODEL}_${ITERATIONS}000/
--num_samples=1000
"
# Note: You might want to add new arguments for Flow-Matching sampling,
# e.g., --ode_steps=100, if you implement that in your sample.py

# run the python scripts
if [[ $MODE == 'train' ]]; then
  python scripts/train.py $TRAIN $COMMON --contr=${CONTR}; # Added --contr here

elif [[ $MODE == 'sample' ]]; then
  python scripts/sample.py $SAMPLE $COMMON --contr=${CONTR}; # Added --contr here

else
  echo "MODE NOT FOUND -> Check the supported modes again";
fi