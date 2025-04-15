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


def compute_metrics(pred_mask, gt_mask, threshold=127):
    pred = (np.array(pred_mask) >= threshold).astype(np.uint8)

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
    record = []  # 保存(iou, filename)

    for filename in tqdm(files, desc=f"Processing {split}"):
        pred_path = os.path.join(pred_dir, filename)
        pred_img = Image.open(pred_path).convert("L")

        gt_path = os.path.join(GT_DIR, filename.replace('_pred.png', '.png'))
        if not os.path.exists(gt_path):
            print(f"Ground truth not found for {filename}. Skipping.")
            continue
        gt_img = Image.open(gt_path).convert("L")

        dice, iou = compute_metrics(pred_img, gt_img)

        iou_scores.append(iou)
        dice_scores.append(dice)
        record.append((iou, filename))  

    iou_mean = np.mean(iou_scores)
    dice_mean = np.mean(dice_scores)
    print(f"\n✅ Average Dice score: {dice_mean:.4f}")
    print(f"✅ Average IoU score:  {iou_mean:.4f}")

    record.sort(reverse=True)
    topk = 20
    top_dir_mask = os.path.join(PRED_DIR, 'top20_mask')
    top_dir_crop = os.path.join(PRED_DIR, 'top20_crop')
    os.makedirs(top_dir_mask, exist_ok=True)
    os.makedirs(top_dir_crop, exist_ok=True)

    for iou, filename in record[:topk]:
        src_mask = os.path.join(PRED_DIR, filename)
        dst_mask = os.path.join(top_dir_mask, filename)
        os.system(f'cp {src_mask} {dst_mask}')

        src_crop = os.path.join(PRED_DIR.replace('preds', 'crops'), filename.replace('_pred.png', '_crop.png'))
        dst_crop = os.path.join(top_dir_crop, filename.replace('_pred.png', '_crop.png'))
        if os.path.exists(src_crop):
            os.system(f'cp {src_crop} {dst_crop}')

    print(f'🎯 Top{topk} results saved to {top_dir_mask} and {top_dir_crop}')

    return iou_mean, dice_mean