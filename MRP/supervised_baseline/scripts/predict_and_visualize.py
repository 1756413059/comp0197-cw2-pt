import os
import sys
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

# === Add project root for imports ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, LIST_FILE, CHECKPOINT_DIR, PRED_DIR
from utils.model import get_unet
from utils.dataset import GTMaskDataset

# === Setup
model_path = os.path.join(CHECKPOINT_DIR, 'fully_supervised_resnet18.pth')
mask_dir = os.path.join(os.path.dirname(CHECKPOINT_DIR), 'gt_masks')
os.makedirs(PRED_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_unet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Load dataset (same transform as training)
dataset = GTMaskDataset(IMAGE_DIR, mask_dir, LIST_FILE)

# === Predict and save
print("Generating predictions for first 10 samples...")
with torch.no_grad():
    for idx in tqdm(range(10)):
        image, _, image_name = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)

        output = model(image_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        filename = image_name.replace('.jpg', '_pred.png')
        save_path = os.path.join(PRED_DIR, filename)
        Image.fromarray(pred_mask, mode='L').save(save_path)

print(f"Saved predicted masks to: {PRED_DIR}")
