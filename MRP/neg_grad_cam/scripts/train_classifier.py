import os
import sys
from torchvision.transforms.v2 import MixUp


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18, get_mobilenet_v3_small



from scripts.config import IMAGE_DIR, TRAIN_FILE, CHECKPOINT_DIR, TEST_FILE



def train_classifier(data_root, train_file, model_name="resnet",val_file=None, ckpt_dir=None,
                     num_classes=37, batch_size=32, epochs=10, lr=1e-4, early_stop=True,mixup=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # image processing: Resize → ToTensor → Normalize（ImageNet standard）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # load training dataset
    dataset = PetClassificationDataset(data_root, train_file, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # load training dataset
    dataset = PetClassificationDataset(data_root, train_file, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    if val_file is not None:
        val_dataset = PetClassificationDataset(data_root, val_file, transform)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    print("len(dataset): ", len(dataset))

    if model_name == "resnet":
        model = get_resnet18(num_classes=num_classes).to(device)
    elif model_name == "mobilenet":
        model = get_mobilenet_v3_small().to(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    
    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


    # train model
    last_val_loss = 1e10
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for images, labels, _ in dataloader:
            if mixup:
                images, labels = MixUp(num_classes=37)(images, labels,)
                
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            if not mixup:
                correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct / total * 100
        # print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.2f}%")

        # validation
        if val_file is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                for images, labels, _ in val_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)

                val_loss /= len(val_dataset)
                val_acc = val_correct / val_total * 100
                print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.2f}%  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

                if early_stop:
                    if val_loss < last_val_loss:
                        last_val_loss = val_loss
                    else:
                        print("Early stopping at epoch ", epoch)
                        break

    if ckpt_dir is not None:
        mixup_str = "_mixup" if mixup else ""
        save_path = os.path.join(ckpt_dir, f'{model_name}_epoch{epoch}{mixup_str}.pth')
        # save model weights
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


import os

if __name__ == '__main__':

    train_classifier(
        data_root=IMAGE_DIR,
        train_file=TRAIN_FILE,
        val_file=TEST_FILE,
        ckpt_dir=CHECKPOINT_DIR,
        num_classes=37,
        batch_size=64,
        epochs=20,
        lr=1e-4,
        mixup=False,
        early_stop=True,
        model_name="mobilenet"
    )