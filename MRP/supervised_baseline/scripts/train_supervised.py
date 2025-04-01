import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, LIST_FILE, CHECKPOINT_DIR
from utils.dataset import PetSegmentationDataset
from utils.model import get_unet

# Use real ground truth masks from annotations/trimaps
GT_MASK_DIR = os.path.join(os.path.dirname(IMAGE_DIR), 'annotations', 'trimaps')
SAVE_PATH = os.path.join(CHECKPOINT_DIR, 'unet_seg_supervised.pth')

class GTMaskDataset(PetSegmentationDataset):
    def __getitem__(self, idx):
        image_name, _ = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Ground-truth mask: no "_mask" in filename
        gt_mask_name = image_name.replace('.jpg', '.png')
        gt_mask_path = os.path.join(self.mask_dir, gt_mask_name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(gt_mask_path).convert('L')

        image = TF.resize(image, (224, 224))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        mask = TF.resize(mask, (224, 224))
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask
    
# === Hyperparameters ===
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

# === Load dataset ===
dataset = GTMaskDataset(IMAGE_DIR, GT_MASK_DIR, LIST_FILE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Init model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_unet().to(device)

# === Loss & Optimizer ===
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Training loop ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# === Save model ===
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved supervised UNet model to {SAVE_PATH}")
