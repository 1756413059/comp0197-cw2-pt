import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18

from scripts.config import IMAGE_DIR, LIST_FILE, CHECKPOINT_DIR



def train_classifier(data_root, list_file, save_path,
                     num_classes=37, batch_size=32, epochs=10, lr=1e-4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # image processing: Resize → ToTensor → Normalize（ImageNet standard）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # load training dataset
    dataset = PetClassificationDataset(data_root, list_file, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # load model (ResNet18) and modify output layer
    model = get_resnet18(num_classes=num_classes).to(device)
    
    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train model
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.2f}%")

    # save model weights
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


import os

if __name__ == '__main__':
    save_path = os.path.join(CHECKPOINT_DIR, 'resnet18_cls_epoch_5.pth')

    train_classifier(
        data_root=IMAGE_DIR,
        list_file=LIST_FILE,
        save_path=save_path,
        num_classes=37,
        batch_size=32,
        epochs=5,
        lr=1e-4
    )