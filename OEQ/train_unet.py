# train_unet.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.Unet import SegmentationDataset, UNet, CombinedLoss

def run_train_unet():
    print("Start training UNet...")
    images_dir = "output_vit/images"
    pseudo_train_dir = os.path.join("output_vit", "masks")
    model_name = "vit"

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = SegmentationDataset(images_dir, pseudo_train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=1e-4)

    num_epochs = 15
    for epoch in range(num_epochs):
        unet_model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            preds = unet_model(images)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

    os.makedirs("saved_models", exist_ok=True)
    filename = f"unet_{model_name}.pth"
    torch.save(unet_model.state_dict(), os.path.join("saved_models", filename))
    print(f"Model successfully trained for {model_name} and saved as {filename}.")
# vit: IoU 0.5998, Dice 0.4419