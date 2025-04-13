import os
import torch
from torchvision import transforms
from utils.Unet import UNet, generate_and_save_segmentation
from utils.dataset import PetDataset
from utils.metrics import compute_metrics_for_split

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
data_root = 'oxford-iiit-pet'
test_dataset = PetDataset(data_root, split='test', transform=transform)

# Evaluate for all models
models = [
    'cam',
    'gcam',
    'gcampp',
]

for model_name in models:
    output_dir = os.path.join("output", "segmentation", model_name)
    # Load the trained U-Net model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1).to(device)
    filename = f"unet_{model_name}.pth"
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    generate_and_save_segmentation(test_dataset, model, "test", output_dir)
    dice, iou = compute_metrics_for_split("test", pred_dir=output_dir)
    print(f"{model_name}: IoU {iou:.4f}, Dice {dice:.4f}")

# vit: IoU 0.5998, Dice 0.4419