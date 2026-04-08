lerobot-train \
  --dataset.repo_id=/home/ubuntu/LeForge/Data/Postprocess_lerobot/pick_cubes_relative_eef_obs_h5_action_30 \
  --dataset.use_relative_eef_chunk=true \
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
  --batch_size=16 \
  --steps=2000000 \
  --output_dir=outputs/train_relative/pick_cubes_obsh5_relative_30_0402 \
  --job_name=pick_cubes \
  --policy.device=cuda \
  --wandb.enable=false \
  --log_freq=20 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.random_order=true \
  --save_freq=20000
  #--resume=true \
  #--config_path=outputs/train_relative/pick_cubes_relative_30_0330/checkpoints/0520000/pretrained_model/train_config.json \

## allowed_encoder_types = {"resnet", "siglip2", "dinov2", "dinov3"}
