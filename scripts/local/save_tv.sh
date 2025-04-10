#/scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_lora_merged_image_only_with_boi_eoi

#do save tv for 0.1 to 1.0 with interval 0.05
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "Saving TV for alpha: ${alpha}"
    python3 -m src.save_tv --model_pretrained /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf \
        --model_finetuned /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_lora_merged_image_only_with_boi_eoi \
        --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_lora_merged_image_only_with_boi_eoi_tv-${alpha} \
        --tv_edit_alpha ${alpha}

    #bin to pth
    python3 -m src.bin_to_pth --trained_ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_lora_merged_image_only_with_boi_eoi_tv-${alpha}  \
        --original_ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/anole/checkpoints/Anole-7b-v0.1 \
        --new_ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_lora_merged_image_only_with_boi_eoi_tv-${alpha}_pth
    echo "Done saving TV for alpha: ${alpha}"
done

# /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_lora_merged_image_only_no_boi_eoi

#do save tv for 0.1 to 1.0 with interval 0.05
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "Saving TV for alpha: ${alpha}"
    python3 -m src.save_tv --model_pretrained /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf \
        --model_finetuned /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_lora_merged_image_only_no_boi_eoi \
        --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_lora_merged_image_only_no_boi_eoi_tv-${alpha} \
        --tv_edit_alpha ${alpha}

    #bin to pth
    python3 -m src.bin_to_pth --trained_ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_lora_merged_image_only_no_boi_eoi_tv-${alpha}  \
        --original_ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/anole/checkpoints/Anole-7b-v0.1 \
        --new_ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_lora_merged_image_only_no_boi_eoi_tv-${alpha}_pth
    echo "Done saving TV for alpha: ${alpha}"
done

# /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_whole_model_image_only_with_boi_eoi/checkpoint-5000

#do save tv for 0.1 to 1.0 with interval 0.05
for alpha in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
    echo "Saving TV for alpha: ${alpha}"
    python3 -m src.save_tv --model_pretrained /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf \
        --model_finetuned /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_whole_model_image_only_with_boi_eoi/checkpoint-5000 \
        --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_whole_model_image_only_with_boi_eoi_tv-${alpha} \
        --tv_edit_alpha ${alpha}

    #bin to pth
    python3 -m src.bin_to_pth --trained_ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_whole_model_image_only_with_boi_eoi_tv-${alpha}  \
        --original_ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/anole/checkpoints/Anole-7b-v0.1 \
        --new_ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_whole_model_image_only_with_boi_eoi_tv-${alpha}_pth
    echo "Done saving TV for alpha: ${alpha}"
done

#/scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_whole_model_image_only_no_boi_eoi/checkpoint-5000

#do save tv for 0.1 to 1.0 with interval 0.05
for alpha in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
    echo "Saving TV for alpha: ${alpha}"
    python3 -m src.save_tv --model_pretrained /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf \
        --model_finetuned /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_whole_model_image_only_no_boi_eoi/checkpoint-5000 \
        --output_dir /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_whole_model_image_only_no_boi_eoi_tv-${alpha} \
        --tv_edit_alpha ${alpha}

    #bin to pth
    python3 -m src.bin_to_pth --trained_ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_whole_model_image_only_no_boi_eoi_tv-${alpha}  \
        --original_ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/anole/checkpoints/Anole-7b-v0.1 \
        --new_ckpt /scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/dog_whole_model_image_only_no_boi_eoi_tv-${alpha}_pth
    echo "Done saving TV for alpha: ${alpha}"
done