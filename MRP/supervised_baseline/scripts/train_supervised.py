import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np

# === Set up paths ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.config import IMAGE_DIR, CHECKPOINT_DIR, TRAINVAL_LIST_FILE, TEST_LIST_FILE
from utils.dataset import GTMaskDataset
from utils.model import get_unet
from scripts.evaluate_supervised import evaluate_model

# === Paths ===
MASK_DIR = os.path.join(os.path.dirname(CHECKPOINT_DIR), 'gt_masks')
SAVE_PATH = os.path.join(CHECKPOINT_DIR, 'fully_supervised.pth')

# === Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# === Hyperparameters
BATCH_SIZE = 16
EPOCHS = 25
LR = 5e-4 

# === Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Dataset
train_dataset = GTMaskDataset(IMAGE_DIR, MASK_DIR, TRAINVAL_LIST_FILE, split=1)
val_dataset = GTMaskDataset(IMAGE_DIR, MASK_DIR, TRAINVAL_LIST_FILE, split=2)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# === Model
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

# === Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for i, (images, masks, _) in enumerate(train_loader):
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

        if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
            print(f"Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_dataset)
    print(f"Avg Loss: {avg_loss:.4f}")

# === Save model
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")

# === Evaluation
print("\n Evaluating on validation set...")
evaluate_model(model, val_dataset, batch_size=BATCH_SIZE, device=device)

# Optional: Test set evaluation
if os.path.exists(TEST_LIST_FILE):
    print("\n Evaluating on test set...")
    test_dataset = GTMaskDataset(IMAGE_DIR, MASK_DIR, TEST_LIST_FILE)
    evaluate_model(model, test_dataset, batch_size=BATCH_SIZE, device=device)
