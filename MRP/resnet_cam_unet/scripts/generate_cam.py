import os
import sys
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Add project root for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, LIST_FILE, CHECKPOINT_DIR, CAM_DIR
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18
from utils.cam_utils import generate_cam

def gray_to_colormap(cam_array):
    # cam_array: 0~1 float numpy array
    cam_uint8 = np.uint8(cam_array * 255)
    h, w = cam_uint8.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)

    # Manual colormap: Red-Green-Blue
    color_map[..., 0] = cam_uint8                     # Red
    color_map[..., 1] = np.maximum(0, cam_uint8 - 100)  # Green
    color_map[..., 2] = np.maximum(0, 255 - cam_uint8)  # Blue

    return Image.fromarray(color_map, mode='RGB')

# === Create output dir
os.makedirs(CAM_DIR, exist_ok=True)

# === Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load 1 image from dataset
dataset = PetClassificationDataset(IMAGE_DIR, LIST_FILE, transform=transform, train_only=True)
image, label, image_name = dataset[0]

# === Load classifier
model = get_resnet18(num_classes=37)
state_dict = torch.load(os.path.join(CHECKPOINT_DIR, 'resnet18_cls.pth'), map_location='cpu')
model.load_state_dict(state_dict)

# === Generate CAM
cam = generate_cam(model, image, target_class=label)

# === Save CAM visualization
cam_vis = gray_to_colormap(cam)
save_path = os.path.join(CAM_DIR, f'{image_name}_cam.jpg')
cam_vis.save(save_path)
print(f"✅ Saved CAM image to: {save_path}")
