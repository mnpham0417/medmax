## Running this code

For image generation with extended vocab size:

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m training.train --train_data /localhome/hbansal/multi-x-dev/anole/facilitating_image_generation/new_dataset_tokenized.jsonl --ckpt /localhome/data/ckpts/Anole-7b-v0.1-hf/ --ds /localhome/hbansal/multi-x-dev/anole/facilitating_image_generation/ds_config.json --output_dir /localhome/data/ckpts/hbansal/anole_extended/ --epoch 1 --bs 1 --save_strategy epoch --warmup_ratio 0.1 --wandb --name image_gen_w_extend_vocab_8192 --mode train-image --extend_vocab --extend_vocab_size 8192 --path_to_tokenizer /localhome/data/ckpts/Anole-7b-v0.1-hf/tokenizer.json 
```