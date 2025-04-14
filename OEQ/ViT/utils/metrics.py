import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import PRED_DIR, GT_DIR



def otsu_threshold(image_array):
    pixel_counts = np.bincount(image_array.flatten(), minlength=256)
    total = image_array.size
    sum_total = np.dot(np.arange(256), pixel_counts)
    sum_bg, weight_bg, max_var, threshold = 0.0, 0.0, 0.0, 0
    for t in range(256):
        weight_bg += pixel_counts[t]
        if weight_bg == 0: continue
        weight_fg = total - weight_bg
        if weight_fg == 0: break
        sum_bg += t * pixel_counts[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def compute_metrics(pred_mask, gt_mask):
    pred_mask = np.array(pred_mask)
    pred = (pred_mask > otsu_threshold(pred_mask)).astype(np.uint8)
    gt_np = np.array(gt_mask)
    gt = (gt_np == 1).astype(np.uint8)
    pred = pred.flatten()
    gt = gt.flatten()
    dice = f1_score(gt, pred)
    iou = jaccard_score(gt, pred)

    return dice, iou


def compute_metrics_for_split(split, pred_dir):
    files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith('.png')])

    iou_scores = []
    dice_scores = []
    for filename in tqdm(files, desc=f"Processing {split}"):
        pred_path = os.path.join(pred_dir, filename)
        pred_img = Image.open(pred_path).convert("L")

        gt_path = os.path.join(GT_DIR, f"{filename.replace('_pred.png', '.png')}")
        if not os.path.exists(gt_path):
            print(f"Ground truth not found for {filename}. Skipping.")
            continue
        gt_img = Image.open(gt_path).convert("L")

        dice, iou = compute_metrics(pred_img, gt_img)
        iou_scores.append(iou)
        dice_scores.append(dice)

    iou_mean = np.mean(iou_scores)
    dice_mean = np.mean(dice_scores)
    print(f"\nAverage Dice score: {dice_mean:.4f}")
    print(f"Average IoU score:  {iou_mean:.4f}")

    return iou_mean, dice_mean