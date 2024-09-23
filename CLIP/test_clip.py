import numpy as np
import torch
import clip
from PIL import Image

device = 'mps'
model,preprocess = clip.load('ViT-B/32',device=device)

image_path = '/Users/zhenxing/Documents/workspace/dataset/3dgs/mip-NERF-360/360_v2/bonsai/images_2/DSCF5566.JPG'

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = clip.tokenize(['desk', 'fake flowers', 'vehicle', 'bikes','walls']).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image,text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
print(probs)