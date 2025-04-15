import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18
from scripts.config import IMAGE_DIR, TRAIN_LIST_FILE, TEST_LIST_FILE, CHECKPOINT_DIR


def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total * 100
    return avg_loss, acc


def train_classifier(data_root, train_list, val_list, save_path,
                     num_classes=37, batch_size=32, epochs=10, lr=1e-4, weight_decay=1e-4):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = PetClassificationDataset(data_root, train_list, train_transform)
    val_dataset = PetClassificationDataset(data_root, val_list, val_transform)

    use_cuda = device.type == 'cuda'
    use_mps = device.type == 'mps'

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0 if use_mps else 4, pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0 if use_mps else 4, pin_memory=use_cuda
    )

    model = get_resnet18(num_classes=num_classes).to(device)

    print("Unfreezing layer3, layer4 and fc (freezing earlier layers):")
    for name, param in model.named_parameters():
        if name.startswith("layer4") or name.startswith("layer3") or name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = False
            print(f" - Frozen: {name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
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

        train_loss = running_loss / len(train_dataset)
        train_acc = correct / total * 100
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    epochs = 10
    model_filename = f"resnet18_cls.pth"
    save_path = os.path.join(CHECKPOINT_DIR, model_filename)

    train_classifier(
        data_root=IMAGE_DIR,
        train_list=TRAIN_LIST_FILE,
        val_list=TEST_LIST_FILE,
        save_path=save_path,
        num_classes=37,
        batch_size=32,
        epochs=epochs,
        lr=1e-4,
        weight_decay=5e-4
    )
