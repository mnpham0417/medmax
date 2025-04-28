export CUDA_VISIBLE_DEVICES=0  # Adjust based on available GPUs

torchrun --nproc_per_node=1 -m training.train --train_data /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/maze_dataset/train_metadata_direct_500.jsonl \
  --ckpt ./checkpoints/anole_7b_hf \
  --output_dir ./checkpoints/maze_direct_500 \
  --epoch 1 \
  --bs 1 \
  --save_strategy steps \
  --save_steps 2000 \
  --warmup_ratio 0.1 \
  --name maze_direct_500 \
  --lr 1e-4 --grad_acc 4 --wandb --wandb_entity Medmax --wandb_disable_ssl --lora --bf16