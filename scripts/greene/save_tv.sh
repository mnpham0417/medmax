#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --account=pr_95_tandon_priority
#SBATCH --time=23:59:00
#SBATCH --mem=64GB
#SBATCH --job-name=save_tv
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mp5847@nyu.edu
#SBATCH --output=save_tv_%j.out

module purge
module load nccl/cuda11.6/2.12.12
module load anaconda3/2024.02

singularity exec --nv \
    --overlay /scratch/work/public/imagenet/imagenet-train.sqf:ro \
	--overlay /scratch/work/public/imagenet/imagenet-val.sqf:ro \
	--overlay /scratch/mp5847/singularity_containers/overlay-50G-10M.ext3:ro \
	/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
	/bin/bash -c "source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh; conda activate /scratch/mp5847/conda_environments/conda_pkgs/medmax;
					 sh save_tv.sh" 