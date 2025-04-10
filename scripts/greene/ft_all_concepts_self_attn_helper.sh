# #cat concept
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m training.train --train_data /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/cat/metadata_final.jsonl \
#                                           --ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf --ds training/ds_config.json \
#                                           --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/cat_self_attn \
#                                           --epoch 200 \
#                                           --bs 1 \
#                                           --save_strategy steps \
#                                           --save_steps 2500 \
#                                           --warmup_ratio 0.1 \
#                                           --name cat_self_attn \
#                                           --trainable_layers self_attn \
#                                           --lr 1e-4 --grad_acc 1 --wandb --wandb_entity Medmax --wandb_disable_ssl --bf16

# #church concept
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m training.train --train_data /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/church/metadata_final.jsonl \
#   --ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf --ds training/ds_config.json \
#   --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/church_self_attn \
#   --epoch 200 \
#   --bs 1 \
#   --save_strategy steps \
#   --save_steps 2500 \
#   --warmup_ratio 0.1 \
#   --name church_self_attn \
#   --trainable_layers self_attn \
#   --lr 1e-4 --grad_acc 1 --wandb --wandb_entity Medmax --wandb_disable_ssl --bf16

# #dog concept
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m training.train --train_data /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/dog/metadata_final.jsonl \
#   --ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf --ds training/ds_config.json \
#   --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_self_attn \
#   --epoch 200 \
#   --bs 1 \
#   --save_strategy steps \
#   --save_steps 2500 \
#   --warmup_ratio 0.1 \
#   --name dog_self_attn \
#   --trainable_layers self_attn \
#   --lr 1e-4 --grad_acc 1 --wandb --wandb_entity Medmax --wandb_disable_ssl --bf16

# #elon musk concept
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m training.train --train_data /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/elon_musk/metadata_final.jsonl \
#   --ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf --ds training/ds_config.json \
#   --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/elon_musk_self_attn \
#   --epoch 200 \
#   --bs 1 \
#   --save_strategy steps \
#   --save_steps 2500 \
#   --warmup_ratio 0.1 \
#   --name elon_musk_self_attn \
#   --trainable_layers self_attn \
#   --lr 1e-4 --grad_acc 1 --wandb --wandb_entity Medmax --wandb_disable_ssl --bf16

# #obama concept
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m training.train --train_data /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/obama/metadata_final.jsonl \
#   --ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf --ds training/ds_config.json \
#   --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/obama_self_attn \
#   --epoch 200 \
#   --bs 1 \
#   --save_strategy steps \
#   --save_steps 2500 \
#   --warmup_ratio 0.1 \
#   --name obama_self_attn \
#   --trainable_layers self_attn \
#   --lr 1e-4 --grad_acc 1 --wandb --wandb_entity Medmax --wandb_disable_ssl --bf16

# #trump concept
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m training.train --train_data /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/trump/metadata_final.jsonl \
#   --ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf --ds training/ds_config.json \
#   --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/trump_self_attn \
#   --epoch 200 \
#   --bs 1 \
#   --save_strategy steps \
#   --save_steps 2500 \
#   --warmup_ratio 0.1 \
#   --name trump_self_attn \
#   --trainable_layers self_attn \
#   --lr 1e-4 --grad_acc 1 --wandb --wandb_entity Medmax --wandb_disable_ssl --bf16


# #picasso concept
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m training.train --train_data /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/picasso/metadata_final.jsonl \
#   --ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf --ds training/ds_config.json \
#   --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/picasso_self_attn \
#   --epoch 200 \
#   --bs 1 \
#   --save_strategy steps \
#   --save_steps 2500 \
#   --warmup_ratio 0.1 \
#   --name picasso_self_attn \
#   --trainable_layers self_attn \
#   --lr 1e-4 --grad_acc 1 --wandb --wandb_entity Medmax --wandb_disable_ssl --bf16

# #rembrandt concept
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m training.train --train_data /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/rembrandt/metadata_final.jsonl \
#   --ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf --ds training/ds_config.json \
#   --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/rembrandt_self_attn \
#   --epoch 200 \
#   --bs 1 \
#   --save_strategy steps \
#   --save_steps 2500 \
#   --warmup_ratio 0.1 \
#   --name rembrandt_self_attn \
#   --trainable_layers self_attn \
#   --lr 1e-4 --grad_acc 1 --wandb --wandb_entity Medmax --wandb_disable_ssl --bf16

# #van gogh concept
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m training.train --train_data /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/van_gogh/metadata_final.jsonl \
#   --ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf --ds training/ds_config.json \
#   --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/van_gogh_self_attn \
#   --epoch 200 \
#   --bs 1 \
#   --save_strategy steps \
#   --save_steps 2500 \
#   --warmup_ratio 0.1 \
#   --name van_gogh_self_attn \
#   --trainable_layers self_attn \
#   --lr 1e-4 --grad_acc 1 --wandb --wandb_entity Medmax --wandb_disable_ssl --bf16

#picasso concept whole model
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m training.train --train_data /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/picasso/metadata_final.jsonl \
  --ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf \
  --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/picasso_whole_model \
  --epoch 200 \
  --bs 1 \
  --save_strategy steps \
  --save_steps 2500 \
  --warmup_ratio 0.1 \
  --name picasso_whole_model \
  --trainable_layers whole_model \
  --lr 1e-4 --grad_acc 1 --wandb --wandb_entity Medmax --wandb_disable_ssl --bf16