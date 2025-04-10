export CUDA_VISIBLE_DEVICES=0  # Adjust based on available GPUs

torchrun --nproc_per_node=1 -m training.train --train_data ./data/dog_erasure/metadata_final.jsonl \
  --ckpt ./checkpoints/anole_7b_hf \
  --output_dir ./checkpoints/testing \
  --epoch 100 \
  --bs 1 \
  --save_strategy steps \
  --save_steps 2000 \
  --warmup_ratio 0.1 \
  --name testing \
  --lr 1e-4 --grad_acc 1 --wandb --wandb_entity Medmax --wandb_disable_ssl --lora