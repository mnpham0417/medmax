#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:a100:1
#SBATCH --account=pr_95_tandon_priority
#SBATCH --time=23:59:00
#SBATCH --mem=64GB
#SBATCH --job-name=church_lm_head
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mp5847@nyu.edu
#SBATCH --output=train_church_lm_head_%j.out

module purge
module load nccl/cuda11.6/2.12.12
module load anaconda3/2024.02

singularity exec --nv \
    --overlay /scratch/work/public/imagenet/imagenet-train.sqf:ro \
	--overlay /scratch/work/public/imagenet/imagenet-val.sqf:ro \
	--overlay /scratch/mp5847/singularity_containers/overlay-50G-10M.ext3:ro \
	/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
	/bin/bash -c "source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh; conda activate /scratch/mp5847/conda_environments/conda_pkgs/medmax;
					 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m training.train --train_data /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/church/metadata_final.jsonl \
                                          --ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf --ds training/ds_config.json \
                                          --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/church_lm_head \
                                          --epoch 5000 \
                                          --bs 16 \
                                          --save_strategy steps \
                                          --save_steps 5000 \
                                          --warmup_ratio 0.1 \
                                          --name church_lm_head \
                                          --trainable_layers lm_head \
                                          --lr 1e-3 --grad_acc 1 --wandb --wandb_entity Medmax --name church_lm_head --wandb_disable_ssl --bf16" 

