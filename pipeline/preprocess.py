import cv2
import torch
import numpy as np

def preprocess(image_path, target_size=512):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (target_size, target_size))
    
    # Converter para float [0-255] (o encoder faz a normalização internamente)
    img_float = img_resized.astype(np.float32)
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, img_resized
