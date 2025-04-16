from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import torch
from PIL import Image
import requests

processor = ChameleonProcessor.from_pretrained("/scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf")
model = ChameleonForConditionalGeneration.from_pretrained("/scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/checkpoints/anole_7b_hf", torch_dtype=torch.bfloat16, device_map="cuda")

# prepare image and text prompt
img_path = '/scratch/mp5847/workspace/mixed-modal-erasure/src/medmax/data/dog_erasure/train/img_dog_erasure_0.png'
image = Image.open(img_path)
prompt = "What do you see in this image?<image>"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

print(inputs)

# autoregressively complete prompt
# output = model.generate(**inputs, max_new_tokens=50)
# print(processor.decode(output[0], skip_special_tokens=True))