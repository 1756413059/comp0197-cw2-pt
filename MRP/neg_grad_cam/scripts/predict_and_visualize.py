import os
import sys
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# === Add project root for importing config ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, MASK_DIR, LIST_FILE, CHECKPOINT_DIR, PRED_DIR
from utils.model import get_unet
from utils.dataset import PetSegmentationDataset

# === Load model ===
model_path = os.path.join(CHECKPOINT_DIR, 'unet_seg.pth')
os.makedirs(PRED_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_unet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = PetSegmentationDataset(IMAGE_DIR, MASK_DIR, LIST_FILE, transform=transform)

# === Predict and save masks
with torch.no_grad():
    for idx in range(10):  # Predict first 10 images
        image, _ = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)

        output = model(image_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        filename = dataset.samples[idx][0].replace('.jpg', '_pred.png')
        save_path = os.path.join(PRED_DIR, filename)
        Image.fromarray(pred_mask, mode='L').save(save_path)

print("✅ Saved prediction masks to outputs/preds/")
