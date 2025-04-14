import os
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, jaccard_score
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import PRED_DIR, GT_DIR



def otsu_threshold(image_array):
    """
    Compute an optimal binarization threshold using Otsu's method.

    Args:
        image_array (np.ndarray): 2D grayscale image with pixel values in [0, 255].

    Returns:
        int: Threshold value in range [0, 255] that maximizes between-class variance.

    Notes:
        - Assumes input is already in uint8 format or castable to it.
        - Often used for adaptive thresholding of predicted CAM or mask maps.
    """
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
    """
    Compute Dice and IoU scores between a predicted mask and ground truth.

    Args:
        pred_mask (PIL.Image or np.ndarray): Grayscale predicted mask (values 0-255).
        gt_mask (PIL.Image or np.ndarray): Ground truth trimap where:
            - 1 = foreground
            - 2 = background
            - 3 = boundary/ignore (ignored during evaluation)

    Returns:
        tuple: (dice, iou), both in range [0, 1]

    Notes:
        - Applies Otsu thresholding to binarize the predicted mask.
        - Ground truth pixels with value != 1 are treated as background (no explicit ignore).
        - Returns flattened binary arrays for metric computation.
    """
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
    """
    Evaluate segmentation predictions using Dice and IoU.

    Args:
        split (str): Dataset split name (e.g., 'train' or 'test') for display only.
        pred_dir (str): Directory containing predicted mask .png files.

    Returns:
        tuple: (mean_iou, mean_dice) across all evaluated images.

    Process:
        - Loads each predicted mask from pred_dir
        - Matches it to the corresponding GT trimap from GT_DIR
        - Binarizes the predicted mask using Otsu threshold
        - Computes Dice and IoU scores and reports the average

    Expected filenames:
        - Prediction: xxx_pred.png
        - Ground truth: xxx.png
    """
 
    files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith('.png')])

    iou_scores = []
    dice_scores = []
    for filename in files:        
        pred_path = os.path.join(pred_dir, filename)
        pred_img = Image.open(pred_path).convert("L")

        # Load the corresponding ground truth trimap
        gt_path = os.path.join(GT_DIR, f"{filename.replace('_pred.png', '.png')}")
        if not os.path.exists(gt_path):
            print(f"Ground truth not found for {filename}. Skipping.")
            continue
        gt_img = Image.open(gt_path).convert("L")

        # Compute IoU for foreground and background
        dice, iou = compute_metrics(pred_img, gt_img)
        iou_scores.append(iou)
        dice_scores.append(dice)

    iou_mean = np.mean(iou_scores)
    dice_mean = np.mean(dice_scores)
    print(f"\nAverage Dice score: {dice_mean:.4f}")
    print(f"Average IoU score:  {iou_mean:.4f}")

    return iou_mean, dice_mean