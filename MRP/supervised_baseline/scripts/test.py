import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

from utils.model import get_unet
from utils.dataset import GTMaskDataset  # <- use the updated dataset class
from scripts.config import IMAGE_DIR, LIST_FILE, CHECKPOINT_DIR
from scripts.evaluate_supevised import evaluate_model

# === Updated path to pre-generated masks
MASK_DIR = os.path.join(os.path.dirname(CHECKPOINT_DIR), 'gt_masks')

# === Load binary mask dataset (just 1 sample)
dataset = GTMaskDataset(IMAGE_DIR, MASK_DIR, LIST_FILE)
subset = Subset(dataset, [0])  # Train on 1 image
loader = DataLoader(subset, batch_size=1, shuffle=True)

# === Setup model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_unet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === Train
model.train()
for epoch in range(10):
    for img, mask, _ in loader:
        img, mask = img.to(device), mask.to(device)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        pred = model(img)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# === Save
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'debug_overfit.pth'))

# === Evaluate on the same 1-sample dataset
print("\nRunning evaluation on single training sample...")
evaluate_model(model, subset, batch_size=1, device=device)
