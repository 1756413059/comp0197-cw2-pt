import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import numpy as np
import segmentation_models_pytorch as smp

# === Set up paths ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.config import IMAGE_DIR, LIST_FILE, CHECKPOINT_DIR
from utils.dataset import GTMaskDataset
from utils.model import get_unet
from scripts.evaluate_supevised import evaluate_model

# === Paths ===
MASK_DIR = os.path.join(os.path.dirname(CHECKPOINT_DIR), 'gt_masks')
SAVE_PATH = os.path.join(CHECKPOINT_DIR, 'fully_supervised_resnet18.pth')

# === Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# === Hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4 

# === Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Dataset
dataset = GTMaskDataset(IMAGE_DIR, MASK_DIR, LIST_FILE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# Model: U-Net
model = get_unet().to(device)

# === Loss and Optimizer
def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# === Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for batch_idx, (images, masks, _) in enumerate(tqdm(loader, desc="Training", unit="batch")):
        images, masks = images.to(device), masks.to(device)

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        preds = model(images)
        bce = criterion(preds, masks)
        dice = dice_loss(preds, masks)
        loss = bce + dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    scheduler.step()

    avg_loss = total_loss / len(dataset)
    print(f"Avg Loss: {avg_loss:.4f}")

# === Save model
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved model to {SAVE_PATH}")

# === Evaluation
print("\nRunning evaluation on training set...")
evaluate_model(model, dataset, batch_size=BATCH_SIZE, device=device)
