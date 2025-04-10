CUDA_VISIBLE_DEVICES=0 torchrun -m --nproc-per-node=1 --master_port=29501 src.image_tokenization --input ./data/dog_erasure/image_data.csv \
 --output ./data/dog_erasure/image_data_tokenized \
 --ckpt ./checkpoints/medmax_7b