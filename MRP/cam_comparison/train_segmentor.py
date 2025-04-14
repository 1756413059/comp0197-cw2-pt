import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.Unet import SegmentationDataset, UNet, CombinedLoss

if __name__ == '__main__':
    # Set paths for the original images and the pseudo labels
    images_dir = "oxford-iiit-pet/images"
    models = [
        "grad_cam_pp",
        "grad_cam",
        "cam",
    ]
    for model_name in models:
        pseudo_train_dir = os.path.join("output", "raw", model_name, "train")
        pseudo_test_dir = os.path.join("output", "raw", model_name, "test")

        # Define a transform for the images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # Create dataset and dataloaders for training and testing.
        train_dataset = SegmentationDataset(images_dir, pseudo_train_dir, transform=transform)
        test_dataset = SegmentationDataset(images_dir, pseudo_test_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

        # Set up device, model, loss, and optimizer.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(in_channels=3, out_channels=1).to(device)
        criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        num_epochs = 15
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

            # Validation step on test data.
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for images, masks in test_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    preds = model(images)
                    loss = criterion(preds, masks)
                    test_loss += loss.item() * images.size(0)
            test_loss = test_loss / len(test_loader.dataset)
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {test_loss:.4f}\n")

        # Save the trained model
        filename = f"unet_{model_name}.pth"
        torch.save(model.state_dict(), filename)
        print(f"Model successfully trained for {model_name} and saved as {filename}.")