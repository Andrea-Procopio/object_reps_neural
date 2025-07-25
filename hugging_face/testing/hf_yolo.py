from datasets import load_dataset
from accelerate.test_utils.testing import get_backend
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load ADE20K dataset
ds = load_dataset("scene_parse_150", split="train[:50]")
ds = ds.train_test_split(test_size=0.2)
test_ds = ds["test"]
image = test_ds[0]["image"]

# Device
device, _, _ = get_backend()

# Preprocess image
encoding = image_processor(image, return_tensors="pt")
pixel_values = encoding.pixel_values.to(device)

# Run model
with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
logits = outputs.logits.cpu()

# Resize to original image size
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],  # (width, height) → (H, W)
    mode="bilinear",
    align_corners=False,
)

# Get predicted segmentation
pred_seg = upsampled_logits.argmax(dim=1)[0]

# ADE20K color palette
def ade_palette():
    return np.array([
        [0, 0, 0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
        [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230],
        [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70],
        [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7],
        [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
        [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92],
        [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71],
        [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6],
        [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
        [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140],
        [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0],
        [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255],
        [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
        [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0],
        [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41],
        [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0],
        [255, 0, 255], [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
        [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204],
        [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255],
        [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10],
        [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
        [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31],
        [0, 184, 255], [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255],
        [112, 224, 255], [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0],
        [255, 0, 163], [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
        [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212],
        [214, 255, 0], [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255],
        [0, 41, 255], [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255],
        [0, 245, 255], [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
        [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0],
        [92, 0, 255],
    ])

# Convert predicted mask to RGB color map
color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
palette = np.array(ade_palette())
for label, color in enumerate(palette):
    color_seg[pred_seg == label, :] = color
color_seg = color_seg[..., ::-1]  # RGB → BGR

# Blend image + segmentation overlay
img = np.array(image)
img = img * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)

# Display
plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.axis("off")
plt.show()
