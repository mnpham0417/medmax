# medmax
MedMax: Mixed-Modal Instruction Tuning for Training Biomedical Assistants


[[Webpage]()] [[Paper]()] [[Train Dataset 🤗]()] [[Eval Dataset 🤗]()] [[Model 🤗](https://huggingface.co/mint-medmax/medmax_7b)] 

<p align="center">
    <img src="static/logo.png" width="30%"> <br>.
</p>


## Abstract

Recent advancements in mixed-modal generative models have unlocked seamless multimodal information integration, opening transformative possibilities for biomedical AI in image analysis, diagnosis, and dataset creation. However, existing resources are limited data availability, narrow domain coverage, and restricted origins (e.g., medical papers). To address these gaps, we present MedMax, the first large-scale multimodal biomedical instruction-tuning dataset for mixed-modal foundation models. With 1.47 million instances, MedMax encompasses a diverse range of tasks, including multimodal content generation (interleaved image-text data), biomedical image captioning and generation, visual chatting, and report understanding. These tasks span diverse medical domains such as radiology and histopathology. Subsequently, we fine-tune a mixed-modal foundation model on the MedMax dataset, achieving significant performance improvements: a 26% gain over the base Chameleon model and an 18.3% improvement over GPT-4o across 12 downstream biomedical visual question-answering tasks. Additionally, we introduce a unified evaluation suite for biomedical tasks, providing a robust framework to guide the development of next-generation mixed-modal biomedical AI assistants.

<p align="center">
    <img src="static/main_result.png" width="80%"> <br>Main results.
</p>

## Installation

```
conda create -n medmax python=3.10
conda activate medmax
pip install -r requirements.txt
```

- The above command will install transformers too. However, we will use our custom transformers present in this repo. Hence, `pip uninstall transformers`.

## Evaluation

[Daniel has to finish this]


## Data setup

1. We provide the data on the huggingface datasets at [TBD].
2. The dataset is divided into two parts: (a) credential = 'YES' and (b) credential = 'NO'. In particular, we do not provide the `image_path` and `tokens` for `credential=YES` split.
3. If you want to train a model on the `credential=NO` split, you can directly skip to the finetuning section.
4. Use our instructions on the HF datasets' README to get the remaining image data in the `credential=YES` split. 

Specifically, we provide the instructions to get the `tokens` column once you access to the `image_path` for the remaining datasets.

1. Create a csv file with the absolute image paths in the `img_path` column.
2. Download the MedMax [checkpoint](https://huggingface.co/mint-medmax/medmax_7b) that comes with the VQGAN checkpoint. 
3. Run the following command:
```
    CUDA_VISIBLE_DEVICES=0,1 torchrun -m --nproc-per-node=2 src.image_tokenization --input <path to a csv file> --output <path to save folder> --ckpt <path to medmax folder>
```
4. The code will generate `parquet` files with an additional column of `img_tokens`. 
5. Specifically, the elements of the `img_tokens` should range from `0-8191` (8192 tokens). 


## Tokenizing Multimodal Sequence

1. Please add an offset of 4 to each element of the image tokens to map them to the corresponding vocab ID in the MedMax tokenizer. For example, a VQGAN token ID of 5678 corresponds to a MedMax token ID of 5682. Consequently, the updated tokens will reflect this mapping.

## Finetuning

**Note:** During finetuning, there might be an error due to versioning of the transformers and deepspeed. To fix this, we comment this line that was throwing that error because it was not crucial. 
```
        # if compare_versions("transformers", "<", "4.33"):
            # from transformers.deepspeed import HfDeepSpeedConfig, unset_hf_deepspeed_config
        # else:
        from transformers.integrations import HfDeepSpeedConfig, unset_hf_deepspeed_config
```
in `"/opt/conda/envs/medmax/lib/python3.10/site-packages/accelerate/utils/dataclasses.py", line 1295`.
