lerobot-train \
  --dataset.repo_id=/home/ubuntu/LeForge/Data/Preprocess_lerobot/pick_cubes_eef_rot6d_first \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.vision_encoder_type="resnet" \
  --policy.vision_normalize_in_processor=true \
  --policy.vision_normalize_in_model=false \
  --policy.vision_freeze=false \
  --policy.push_to_hub=false \
  --policy.repo_id="" \
  --policy.type=act \
  --num_workers=16 \
  --batch_size=32 \
  --steps=2000000 \
  --output_dir=outputs/train/rel0_pick_cubes_0401 \
  --job_name=pick_cubes \
  --policy.device=cuda \
  --wandb.enable=false \
  --log_freq=20 

## allowed_encoder_types = {"resnet", "siglip2", "dinov2", "dinov3"}
