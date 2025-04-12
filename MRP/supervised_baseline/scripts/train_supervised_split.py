import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import numpy as np

# === Set up paths ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.config import IMAGE_DIR, LIST_FILE, CHECKPOINT_DIR
from utils.dataset import GTMaskDataset
from scripts.evaluate_supevised import evaluate_model

# === Paths ===
MASK_DIR = os.path.join(os.path.dirname(CHECKPOINT_DIR), 'gt_masks')
SAVE_PATH = os.path.join(CHECKPOINT_DIR, 'fully_supervised_resnet18_split.pth')

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

# === Dataset and Split
dataset = GTMaskDataset(IMAGE_DIR, MASK_DIR, LIST_FILE)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# === Model: U-Net with ResNet-18 encoder ===
import segmentation_models_pytorch as smp
model = smp.Unet(
    encoder_name="resnet18",        # Use ResNet-18
    encoder_weights="imagenet", 
    in_channels=3,
    classes=1,
).to(device)

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
    for batch_idx, (images, masks, _) in enumerate(tqdm(train_loader, desc="Training", unit="batch")):
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

    avg_loss = total_loss / len(train_dataset)
    print(f"Avg Loss: {avg_loss:.4f}")

# === Save model
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved model to {SAVE_PATH}")

# === Evaluation
print("\nRunning evaluation on validation set...")
evaluate_model(model, val_dataset, batch_size=BATCH_SIZE, device=device)