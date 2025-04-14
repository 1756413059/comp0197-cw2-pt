import os

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from scripts.config import IMAGE_DIR, MASK_DIR, TRAIN_LIST_FILE, CHECKPOINT_DIR
from utils.dataset import PetSegmentationDataset
from utils.model import get_segmentor, freeze_backbone

def train_segmentor(
    model_name='unet',
    num_classes=1,
    batch_size=32,
    epochs=15,
    lr=1e-4,
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    list_file=TRAIN_LIST_FILE,
    save_path=None,
):
    print(f"Training {model_name} for {epochs} epochs")

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    dataset = PetSegmentationDataset(image_dir, mask_dir, list_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = get_segmentor(model_name=model_name, num_classes=num_classes).to(device)

    if model_name == 'deeplabv3':
        print("Freezing backbone except layer3/layer4")
        freeze_backbone(model, unfreeze_layers=('layer3', 'layer4'))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)['out'] if model_name == 'deeplabv3' else model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    if save_path is None:
        save_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_seg_epoch_{epochs}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

if __name__ == '__main__':
    model_name = 'unet'
    epochs = 15
    res = 50
    cam = 'otsu'

    train_segmentor(
        model_name=model_name,
        epochs=epochs,
        lr=1e-4,
        batch_size=32,
        save_path=os.path.join(CHECKPOINT_DIR, f"{model_name}_seg_resnet{res}_cam_{cam}_epoch_{epochs}.pth")
    )
