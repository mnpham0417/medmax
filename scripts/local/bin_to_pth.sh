CUDA_VISIBLE_DEVICES=0 python -m src.bin_to_pth --trained_ckpt ./checkpoints/maze_direct_2000-iter-20000_hf \
            --original_ckpt ./checkpoints/Anole-7b-v0.1 \
            --new_ckpt ./checkpoints/maze_direct_2000-iter-20000_hf_pth