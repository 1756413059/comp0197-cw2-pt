import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18, get_resnet50
from scripts.config import IMAGE_DIR, TRAIN_LIST_FILE, TEST_LIST_FILE, CHECKPOINT_DIR


def evaluate(model, dataloader, device, criterion):
    """
    Evaluate a classification model on a given dataset.

    Performs a full forward pass on the validation/test set and computes:
        - Average loss (CrossEntropy)
        - Overall classification accuracy (%)

    Args:
        model (torch.nn.Module): Trained classification model (e.g., ResNet).
        dataloader (torch.utils.data.DataLoader): DataLoader for validation or test set.
        device (torch.device): Device to run inference on (e.g., 'cpu', 'cuda', 'mps').
        criterion (torch.nn.Module): Loss function used for evaluation (typically nn.CrossEntropyLoss).

    Returns:
        tuple:
            - avg_loss (float): Mean loss over the entire dataset.
            - acc (float): Overall top-1 accuracy (%).

    Notes:
        - Model is set to `eval()` mode during evaluation.
        - `torch.no_grad()` is used to disable gradient computation for efficiency.
        - Assumes output logits and integer ground-truth labels.

    Example:
        >>> val_loss, val_acc = evaluate(model, val_loader, device, nn.CrossEntropyLoss())
        >>> print(f"Validation Accuracy: {val_acc:.2f}%")
    """
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
                     num_classes=37, batch_size=32, epochs=10, lr=1e-4, weight_decay=1e-4, model='resnet18'):
    """
    Train an image classifier (ResNet18 or ResNet50) on the Oxford-IIIT Pet dataset.

    This function performs supervised image classification training and validation.
    It also supports selective layer freezing and saves model checkpoints.

    Args:
        data_root (str): Path to the image folder.
        train_list (str): Path to list file for training set.
        val_list (str): Path to list file for validation set.
        save_path (str): Path to save the trained model (.pth).
        num_classes (int): Number of output classes (default 37).
        batch_size (int): Batch size for training and validation.
        epochs (int): Total number of training epochs.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for regularization.
        model (str): Model architecture to use: 'resnet18' or 'resnet50'.

    Returns:
        torch.nn.Module: The trained model (or loaded model if checkpoint already exists).

    Notes:
        - If the checkpoint already exists, training will be skipped and the model will be loaded.
        - Freezes early layers (conv1, bn1, layer1, layer2) to allow fast fine-tuning on small datasets.
        - Logs training and validation loss and accuracy every epoch.
        - Input images are augmented via random crop and horizontal flip (training only).

    Example:
        >>> train_classifier(
                data_root='data/images',
                train_list='data/annotations/train.txt',
                val_list='data/annotations/test.txt',
                save_path='outputs/checkpoints/resnet50_cls_epoch_10.pth',
                model='resnet50'
            )
    """
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
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

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

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

    if model == 'resnet18':
        print("Using ResNet18")
        model = get_resnet18(num_classes=num_classes).to(device)
    elif model == 'resnet50':
        print("Using ResNet50")
        model = get_resnet50(num_classes=num_classes).to(device)

    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}. Loading and skipping training.")
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        return model

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
    model_filename = f"resnet50_cls_epoch_{epochs}.pth"
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
        weight_decay=5e-4,
        model = 'resnet18'  # or 'resnet18'
    )
