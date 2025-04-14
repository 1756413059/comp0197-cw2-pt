import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms

def otsu_threshold(cam_uint8):
    hist, _ = np.histogram(cam_uint8.flatten(), 256, [0, 256])
    threshold = 0.55
    total = cam_uint8.size
    sumB, wB, maximum, sum1 = 0.0, 0.0, 0.0, np.dot(np.arange(256), hist)
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > maximum:
            threshold = i
            maximum = between
    return threshold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.config import IMAGE_DIR, LIST_FILE, CHECKPOINT_DIR, MASK_DIR, TRAIN_LIST_FILE
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18
from utils.cam_utils import generate_cam

os.makedirs(MASK_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = PetClassificationDataset(IMAGE_DIR, TRAIN_LIST_FILE, transform=transform)

model = get_resnet18(num_classes=37)
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'resnet18_cls.pth'), map_location='mps', weights_only=True))

for image_tensor, label, image_name in dataset:
    cam = generate_cam(model, image_tensor, target_class=label)
    cam_uint8 = np.uint8(cam * 255)

    otsu_val = otsu_threshold(cam_uint8)
    threshold = otsu_val * 0.7  
    mask_np = (cam_uint8 >= threshold).astype(np.uint8) * 255

    # Detect front and backgroud ratio for justify threshold
    foreground_ratio = np.sum(mask_np) / mask_np.size

    if foreground_ratio < 0.1:  
        threshold = otsu_val * 0.6
        mask_np = (cam_uint8 >= threshold).astype(np.uint8) * 255

    if foreground_ratio < 0.02:  
        threshold = otsu_val * 0.5
        mask_np = (cam_uint8 >= threshold).astype(np.uint8) * 255

    original_path = os.path.join(IMAGE_DIR, image_name)
    original_image = np.array(Image.open(original_path).convert('RGB'))
    mask_np = cv2.resize(mask_np, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    grabcut_mask = np.where(mask_np == 255, cv2.GC_PR_FGD, cv2.GC_BGD).astype('uint8')

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(original_image, grabcut_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    final_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Filling holes in the mask
    contours, _ = cv2.findContours(final_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cv2.drawContours(final_mask, contours, i, 255, -1) 


    filename = os.path.splitext(image_name)[0]
    mask_img = Image.fromarray(final_mask, mode='L')
    mask_img.save(os.path.join(MASK_DIR, f'{filename}_mask.png'))

print("All pseudo masks generated with GrabCut refinement (without dilation).")
