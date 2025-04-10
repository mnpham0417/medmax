#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:a100:1
#SBATCH --account=pr_95_tandon_priority
#SBATCH --time=23:59:00
#SBATCH --mem=64GB
#SBATCH --job-name=dog_lm_head
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mp5847@nyu.edu
#SBATCH --output=train_dog_lm_head_%j.out

module purge
module load nccl/cuda11.6/2.12.12
module load anaconda3/2024.02

singularity exec --nv \
    --overlay /scratch/work/public/imagenet/imagenet-train.sqf:ro \
	--overlay /scratch/work/public/imagenet/imagenet-val.sqf:ro \
	--overlay /scratch/mp5847/singularity_containers/overlay-50G-10M.ext3:ro \
	/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
	/bin/bash -c "source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh; conda activate /scratch/mp5847/conda_environments/conda_pkgs/medmax;
					 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m training.train --train_data /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/dog_erasure/metadata_final.jsonl \
                        --ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf \
                        --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_whole_model_image_only_no_boi_eoi \
                        --epoch 50 \
                        --bs 1 \
                        --save_strategy steps \
                        --save_steps 500 \
                        --warmup_ratio 0.1 \
                        --name dog_whole_model_image_only_no_boi_eoi \
                        --trainable_layers whole_model \
                        --lr 1e-4 --grad_acc 1 --wandb --wandb_entity Medmax --wandb_disable_ssl --bf16" 

