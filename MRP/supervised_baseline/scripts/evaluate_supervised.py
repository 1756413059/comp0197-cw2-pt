import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score, f1_score

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, CHECKPOINT_DIR, TRAINVAL_LIST_FILE, TEST_LIST_FILE
from utils.model import get_unet
from utils.dataset import GTMaskDataset

def compute_iou(pred, target):
    return jaccard_score(target.flatten(), pred.flatten())

def compute_dice(pred, target):
    return f1_score(target.flatten(), pred.flatten())

def evaluate_model(model, dataset, batch_size=8, device='cpu', verbose=True, max_verbose=5):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    n_samples = 0

    print(f"Evaluating {len(dataset)} samples...")

    with torch.no_grad():
        for batch_idx, (images, masks, names) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).cpu().numpy().astype(np.uint8)
            masks = masks.cpu().numpy().astype(np.uint8)

            for p, g in zip(preds, masks):
                p = np.squeeze(p)
                g = np.squeeze(g)

                try:
                    iou = compute_iou(p, g)
                    dice = compute_dice(p, g)
                except ValueError as e:
                    print(f"Skipping one sample due to shape issue: {e}")
                    continue

                total_iou += iou
                total_dice += dice
                n_samples += 1

                if verbose and n_samples <= max_verbose:
                    print(f"[{n_samples}] IoU: {iou:.3f} | Dice: {dice:.3f}")

            # Optional: show progress every few batches
            if not verbose and (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(loader)} batches...")

    if n_samples == 0:
        print("No valid samples evaluated. Check for missing masks or bad predictions.")
        return 0.0, 0.0

    avg_iou = total_iou / n_samples
    avg_dice = total_dice / n_samples

    print("\n Evaluation Summary")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Evaluated on {n_samples} samples.")

    return avg_iou, avg_dice

# === CLI Entrypoint ===
if __name__ == "__main__":
    model_path = os.path.join(CHECKPOINT_DIR, 'fully_supervised.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_unet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    MASK_DIR = os.path.join(os.path.dirname(CHECKPOINT_DIR), 'gt_masks')

    # === Evaluate on validation set
    print("\n Evaluating on validation set...")
    val_dataset = GTMaskDataset(IMAGE_DIR, MASK_DIR, TRAINVAL_LIST_FILE, split=2)
    evaluate_model(model, val_dataset, batch_size=8, device=device, verbose=True)

    # === Evaluate on test set
    print("\n Evaluating on test set...")
    test_dataset = GTMaskDataset(IMAGE_DIR, MASK_DIR, TEST_LIST_FILE)
    evaluate_model(model, test_dataset, batch_size=8, device=device, verbose=False)