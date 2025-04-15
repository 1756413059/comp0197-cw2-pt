import os
import numpy as np
from PIL import Image

def compute_iou(pred, gt, ignore_val=255):
    """
    Compute IoU for a single image pair.
    pred: numpy array [H, W], values in {0, 1}
    gt:   numpy array [H, W], values in {0, 1, 255}, where 255 = ignore
    """
    valid = (gt != ignore_val)
    pred = pred.astype(bool) & valid
    gt = gt.astype(bool) & valid

    intersection = (pred & gt).sum()
    union = (pred | gt).sum()

    return intersection / union if union != 0 else float('nan')


# === Path settings (update as needed)
PRED_DIR = 'outputs/resnet_cam_unet/preds/'
GT_DIR = 'data/annotations/trimaps/'

ious = []

for filename in os.listdir(PRED_DIR):
    if not filename.endswith('.png'):
        continue

    pred_path = os.path.join(PRED_DIR, filename)
    gt_name = filename.replace('_pred.png', '.png')
    gt_path = os.path.join(GT_DIR, gt_name)

    if not os.path.exists(gt_path):
        print(f"⚠️ GT not found for {filename}, skipping.")
        continue

    # === Load predicted mask (0/255) → binarize
    pred = np.array(Image.open(pred_path).convert('L'))
    pred = (pred > 127).astype(np.uint8)

    # === Load ground truth mask (1=FG, 2=BG, 3=Boundary)
    gt = np.array(Image.open(gt_path).convert('L'))

    # === Resize pred to match GT size
    gt_size = gt.shape[::-1]  # (W, H)
    pred = np.array(Image.fromarray(pred).resize(gt_size, resample=Image.NEAREST))

    # === Convert GT: 1=FG, 2=BG, 3=ignore → {1,0,255}
    gt[gt == 1] = 1
    gt[gt == 2] = 0
    gt[gt == 3] = 255

    # === Compute IoU
    iou = compute_iou(pred, gt)
    ious.append(iou)
    print(f"{filename}: IoU = {iou:.4f}")

# === Final result
mean_iou = np.nanmean(ious)
print(f"\nMean IoU over {len(ious)} images: {mean_iou:.4f}")
