import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from utils.dataset import PetDataset

# Data augmentation and normalization for training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
data_root = 'oxford-iiit-pet'
train_dataset = PetDataset(data_root, split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = PetDataset(data_root, split='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Initialize model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 37)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -----------------------------
# PHASE 1: Freeze all layers except layer4 and fc
# -----------------------------
for name, param in model.named_parameters():
    if not (name.startswith("fc") or name.startswith("layer4")):
        param.requires_grad = False

# Setup optimizer with differential learning rates
optimizer = optim.SGD([
    {'params': model.fc.parameters(), 'lr': 1e-3},
    {'params': model.layer4.parameters(), 'lr': 1e-4}
], momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Train for Phase 1
num_phase1_epochs = 5
model.train()
for epoch in range(num_phase1_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total * 100

    # Test
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()
            test_total += labels.size(0)

        test_loss /= len(test_dataset)
        test_acc = test_correct / test_total * 100
        print(
            f"Phase 1 Epoch {epoch + 1}/{num_phase1_epochs}  Train Loss: {epoch_loss:.4f}  "
            f"Train Acc: {epoch_acc:.2f}%  Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%")

# -----------------------------
# PHASE 2: Unfreeze layer3 for further fine-tuning
# -----------------------------
for name, param in model.named_parameters():
    if name.startswith("layer3"):
        param.requires_grad = True

# Update optimizer
optimizer = optim.SGD([
    {'params': model.fc.parameters(), 'lr': 1e-3},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-4}
], momentum=0.9, weight_decay=1e-4)

# Train for Phase 2
num_phase2_epochs = 15
for epoch in range(num_phase2_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total * 100

    # Test
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()
            test_total += labels.size(0)

        test_loss /= len(test_dataset)
        test_acc = test_correct / test_total * 100
        print(
            f"Phase 2 Epoch {epoch + 1}/{num_phase2_epochs}  Train Loss: {epoch_loss:.4f}  "
            f"Train Acc: {epoch_acc:.2f}%  Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%")

torch.save(model.state_dict(), "Resnet50_FT.pth")
print(f"Model successfully trained and saved as Resnet50_FT.pth")