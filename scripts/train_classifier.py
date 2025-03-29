import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18


def train_classifier(data_root, list_file, save_path,
                     num_classes=37, batch_size=32, epochs=10, lr=1e-4):
    # 设置设备：使用GPU（如果有）否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 图像预处理：Resize → ToTensor → Normalize（ImageNet 标准）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 加载训练数据集
    dataset = PetClassificationDataset(data_root, list_file, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 加载模型（ResNet18）并修改输出层
    model = get_resnet18(num_classes=num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
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

    # 保存模型权重
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    train_classifier(
        data_root='data/images',
        list_file='data/annotations/list.txt',
        save_path='outputs/checkpoints/resnet18_cls.pth',
        num_classes=37,
        batch_size=32,
        epochs=10,
        lr=1e-4
    )