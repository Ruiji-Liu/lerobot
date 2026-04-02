#!/usr/bin/env bash
set -euo pipefail

mkdir -p log
ts="$(date +'%Y%m%d_%H%M%S')"
log_file="log/train_${ts}.log"

echo "[INFO] Logging to: ${log_file}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi | head -n 20
echo "------------------------------------------------------------"

# Line-buffered output so logs are readable in real-time
stdbuf -oL -eL lerobot-train \
  --dataset.repo_id= /home/ubuntu/LeForge/Data/Preprocess_lerobot/social_all_0324 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.vision_encoder_type="resnet" \
  --policy.vision_normalize_in_processor=true \
  --policy.vision_normalize_in_model=false \
  --policy.vision_freeze=false \
  --policy.push_to_hub=false \
  --policy.repo_id="" \
  --policy.type=act \
  --num_workers=8 \
  --batch_size=8 \
  --steps=2000000 \
  --output_dir=outputs/train_social/social_all_0324_aug \
  --job_name=social \
  --policy.device=cuda \
  --wandb.enable=false \
  --log_freq=20 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.random_order=true \
  --tolerance_s=0.001 \
  --resume=true \
  --config_path=outputs/train_social/social_0309/checkpoints/0480000/pretrained_model/train_config.json \
  2>&1 | tee -a "${log_file}"
