# evaluate_unet.py
import os
import torch
from torchvision import transforms
from utils.Unet import UNet, generate_and_save_segmentation
from utils.dataset import PetDataset
from utils.metrics import compute_metrics_for_split

def run_evaluate():
    print("Start evaluating the model...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    data_root = 'oxford-iiit-pet'
    test_dataset = PetDataset(data_root, split='test', transform=transform)

    model_name = "vit"
    output_dir = os.path.join("output", "segmentation", model_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = UNet(in_channels=3, out_channels=1).to(device)
    model_path = os.path.join("saved_models", f"unet_{model_name}.pth")
    unet_model.load_state_dict(torch.load(model_path, map_location=device))
    unet_model.eval()

    generate_and_save_segmentation(test_dataset, unet_model, "test", output_dir)
    dice, iou = compute_metrics_for_split("test", pred_dir=output_dir)
    print(f"{model_name}: IoU {iou:.4f}, Dice {dice:.4f}")
# vit: IoU 0.5998, Dice 0.4419
