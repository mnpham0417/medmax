### Finetuning VQGAN Model

### 1. Install Requirements

```
conda env create -f environment.yaml
conda activate taming
git clone https://github.com/CompVis/taming-transformers.git
git checkout 3ba01b241669f5ade541ce990f7650a3b8f65318 
cd taming-transformers
pip install -e . --no-deps
sed -i '11c\string_classes = str'  taming/data/utils.py
cd ../

```

### 2. Prepare Data
Put all images in a single folder, then change the following lines in `config.yaml` to your folder path

```
      params:
        root: 
          - <path here>
```

### 3. Launch Training

Finally, run the following command 

```
python main.py \
    --base config.yaml \
    --gpus 0,1,2, -t \
    -ro  /path/to/vqgan/vqgan.ckpt 
```