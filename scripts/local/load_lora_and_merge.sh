CUDA_VISIBLE_DEVICES=0 python -m src.load_lora_and_merge --ckpt_path /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/maze_pathfinding_2000/checkpoint-40000 \
            --output_dir ./checkpoints/maze_pathfinding_2000_hf \
            --base_path ./checkpoints/anole_7b_hf