# python src/tokenization.py --input_file ./data/dog/metadata_final.jsonl \
#     --tokenizer_file ./checkpoints/text_tokenizer_modified.json \
#     --output_file ./data/dog/metadata_final.jsonl

python src/tokenization_image_only.py --input_file ./data/dog_erasure/metadata_final.jsonl \
    --tokenizer_file ./checkpoints/text_tokenizer_modified.json \
    --output_file ./data/dog_erasure/metadata_final.jsonl