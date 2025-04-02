import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Add project root to sys.path to support import from scripts.config
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, MASK_DIR, LIST_FILE, CHECKPOINT_DIR, TRAIN_LIST_FILE, TEST_LIST_FILE
from utils.dataset import PetSegmentationDataset
from utils.model import get_unet

# === Config ===
save_path = os.path.join(CHECKPOINT_DIR, 'unet_seg.pth')
batch_size = 8
epochs = 10
lr = 1e-4

# === Load dataset ===
dataset = PetSegmentationDataset(IMAGE_DIR, MASK_DIR, TRAIN_LIST_FILE)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==== Auto-select device: CUDA > MPS > CPU ====
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = get_unet().to(device)

# === Loss & Optimizer ===
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# === Training loop ===
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# === Save model ===
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"✅ Saved UNet model to {save_path}")
