import os
import numpy as np
from PIL import Image

PRED_DIR = 'outputs/resnet_cam_unet/preds/'
GT_DIR = 'data/annotations/trimaps/'
CAM_DIR = 'outputs/resnet_cam_unet/cams/'
CROP_DIR = 'outputs/resnet_cam_unet/crops/'
SAVE_DIR = 'outputs/top20_visual/'
SINGLE_SAVE_DIR = 'outputs/top20_visual_single/'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SINGLE_SAVE_DIR, exist_ok=True)

def compute_iou(pred, gt, ignore_val=255):
    valid = (gt != ignore_val)
    pred = pred.astype(bool) & valid
    gt = gt.astype(bool) & valid
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return intersection / union if union != 0 else float('nan')

def compute_dice(pred, gt, ignore_val=255):
    valid = (gt != ignore_val)
    pred = pred.astype(bool) & valid
    gt = gt.astype(bool) & valid
    intersection = (pred & gt).sum()
    return 2 * intersection / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) != 0 else float('nan')


ious_pred, dices_pred = [], []
results = []

for filename in os.listdir(PRED_DIR):
    if not filename.endswith('.png'):
        continue

    pred_path = os.path.join(PRED_DIR, filename)
    gt_name = filename.replace('_pred.png', '.png')
    gt_path = os.path.join(GT_DIR, gt_name)

    if not os.path.exists(gt_path):
        print(f"⚠️ Missing GT for {filename}, skipping.")
        continue

    pred = np.array(Image.open(pred_path).convert('L'))
    gt = np.array(Image.open(gt_path).convert('L'))

    gt_size = gt.shape[::-1]  # (W, H)
    pred = np.array(Image.fromarray((pred > 127).astype(np.uint8)).resize(gt_size, resample=Image.NEAREST))

    gt[gt == 1] = 1
    gt[gt == 2] = 0
    gt[gt == 3] = 255

    iou_pred = compute_iou(pred, gt)
    dice_pred = compute_dice(pred, gt)

    ious_pred.append(iou_pred)
    dices_pred.append(dice_pred)

    results.append((filename, iou_pred, dice_pred))

    print(f"{filename}: IoU = {iou_pred:.4f} | Dice = {dice_pred:.4f}")

mean_pred_iou = np.nanmean(ious_pred)
mean_pred_dice = np.nanmean(dices_pred)

print(f"\nMean IoU: {mean_pred_iou:.4f}")
print(f"Mean Dice: {mean_pred_dice:.4f}")

# Top 20
results.sort(key=lambda x: x[1], reverse=True)
top20 = results[:20]

for filename, iou, dice in top20:
    img_name = filename.replace('_pred.png', '.jpg')

    cam_path = os.path.join(CAM_DIR, img_name.replace('.jpg', '_cam_overlay.jpg'))
    pred_path = os.path.join(PRED_DIR, filename)
    crop_path = os.path.join(CROP_DIR, img_name.replace('.jpg', '_crop.png'))

    cam = Image.open(cam_path).convert('RGB')
    pred = Image.open(pred_path).convert('RGB')
    crop = Image.open(crop_path).convert('RGB')

    h = max(cam.height, pred.height, crop.height)
    cam = cam.resize((h, h))
    pred = pred.resize((h, h))
    crop = crop.resize((h, h))

    new_img = Image.new('RGB', (h * 3, h))
    new_img.paste(cam, (0, 0))
    new_img.paste(pred, (h, 0))
    new_img.paste(crop, (h * 2, 0))

    save_path = os.path.join(SAVE_DIR, filename.replace('_pred.png', f'_iou_{iou:.4f}_dice_{dice:.4f}.jpg'))
    new_img.save(save_path)

    cam.save(os.path.join(SINGLE_SAVE_DIR, filename.replace('_pred.png', f'_cam_iou_{iou:.4f}_dice_{dice:.4f}.jpg')))
    pred.save(os.path.join(SINGLE_SAVE_DIR, filename.replace('_pred.png', f'_pred_iou_{iou:.4f}_dice_{dice:.4f}.jpg')))
    crop.save(os.path.join(SINGLE_SAVE_DIR, filename.replace('_pred.png', f'_crop_iou_{iou:.4f}_dice_{dice:.4f}.jpg')))

print(f'Top20 Concat Saved in {SAVE_DIR}')
print(f'Top20 Single Saved in {SINGLE_SAVE_DIR}')

