import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import sys
from scripts.config import IMAGE_DIR, MASK_DIR, TRAIN_LIST_FILE, CHECKPOINT_DIR
from utils.dataset import PetSegmentationDataset
from utils.model import get_segmentor, freeze_backbone  # ⬅️ 替代 get_unet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

batch_size = 32
epochs = 15
lr = 1e-4
model_name = 'deeplabv3'   
# model_name = 'unet'   # 'unet' or 'deeplabv3'
save_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_seg_epoch_{epochs}.pth")


dataset = PetSegmentationDataset(IMAGE_DIR, MASK_DIR, TRAIN_LIST_FILE)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


model = get_segmentor(model_name=model_name, num_classes=1).to(device)

if model_name == 'deeplabv3':
    print("Freezing backbone except layer4, training classifier head...")
    freeze_backbone(model, unfreeze_layers=('layer3', 'layer4'))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        if model_name == 'deeplabv3':
            outputs = model(images)['out']  # DeepLabV3
        else:
            outputs = model(images)         # UNet

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"✅ Saved {model_name} model to {save_path}")
