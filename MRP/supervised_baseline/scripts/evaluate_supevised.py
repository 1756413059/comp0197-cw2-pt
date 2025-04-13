import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score, f1_score

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, LIST_FILE, CHECKPOINT_DIR
from utils.model import get_unet
from utils.dataset import GTMaskDataset

def compute_iou(pred, target):
    return jaccard_score(target.flatten(), pred.flatten())

def compute_dice(pred, target):
    return f1_score(target.flatten(), pred.flatten())

def evaluate_model(model, dataset, batch_size=8, device='cpu', verbose=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    n_samples = 0

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).cpu().numpy().astype(np.uint8)
            masks = masks.cpu().numpy().astype(np.uint8)
            
            for p, g in zip(preds, masks):
                p = np.squeeze(p)
                g = np.squeeze(g)

                iou = compute_iou(p, g)
                dice = compute_dice(p, g)

                total_iou += iou
                total_dice += dice
                n_samples += 1

                if verbose and n_samples <= 5:
                    print(f"[{n_samples}] IoU: {iou:.3f} | Dice: {dice:.3f}")

    avg_iou = total_iou / n_samples
    avg_dice = total_dice / n_samples

    if verbose:
        print("\nEvaluation Summary")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average Dice: {avg_dice:.4f}")
        print(f"Evaluated on {n_samples} samples.")

    return avg_iou, avg_dice

# === CLI Entrypoint ===
if __name__ == "__main__":
    model_path = os.path.join(CHECKPOINT_DIR, 'fully_supervised_20.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_unet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Use pre-generated GT mask
    MASK_DIR = os.path.join(os.path.dirname(CHECKPOINT_DIR), 'gt_masks')
    dataset = GTMaskDataset(IMAGE_DIR, MASK_DIR, LIST_FILE)

    evaluate_model(model, dataset, batch_size=8, device=device)
