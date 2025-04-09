import os
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add project root for config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, LIST_FILE, CHECKPOINT_DIR
GT_MASK_DIR = os.path.join(os.path.dirname(IMAGE_DIR), 'annotations', 'trimaps')
OUTPUT_DIR = os.path.join(os.path.dirname(CHECKPOINT_DIR), 'gt_masks')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read image names from list file
image_names = []
with open(LIST_FILE, 'r') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) == 4 and int(parts[3]) == 1:  # train split only
            image_names.append(parts[0])

print(f"Processing {len(image_names)} trimaps...")

for name in tqdm(image_names):
    mask_path = os.path.join(GT_MASK_DIR, f"{name}.png")
    save_path = os.path.join(OUTPUT_DIR, f"{name}_mask.png")

    # Load trimap
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    # Convert: pet class 2 → 1, others → 0
    binary_mask = (mask_np == 2).astype(np.uint8) * 255
    Image.fromarray(binary_mask).save(save_path)

print(f"Saved binary masks to {OUTPUT_DIR}")
