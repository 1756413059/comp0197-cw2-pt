import os
import sys
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from matplotlib import cm

# Add project root for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, LIST_FILE, CHECKPOINT_DIR, CAM_DIR
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18
from utils.cam_utils import generate_cam

# === Number of images to visualize ===
NUM_SAMPLES = 100  # Change this number as needed

# === Create output directory ===
os.makedirs(CAM_DIR, exist_ok=True)

# === Preprocessing (same as training) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Utility: Overlay CAM heatmap on original image ===
def overlay_heatmap(cam_array, image_pil, alpha=0.5):
    cam_uint8 = np.uint8(cam_array * 255)
    heatmap = cm.jet(cam_uint8)[:, :, :3] * 255  # RGB heatmap (drop alpha)
    heatmap = heatmap.astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).resize(image_pil.size)

    blended = Image.blend(image_pil.convert("RGB"), heatmap_img, alpha=alpha)
    return blended

# === Load dataset ===
dataset = PetClassificationDataset(IMAGE_DIR, LIST_FILE, transform=transform)

# === Load classifier ===
model = get_resnet18(num_classes=37)
state_dict = torch.load(
    os.path.join(CHECKPOINT_DIR, 'resnet18_cls_epoch_10.pth'),
    map_location='cpu',
    weights_only=True
)
model.load_state_dict(state_dict)
model.eval()

# === Generate and save CAM overlay for first N images ===
for idx in range(min(NUM_SAMPLES, len(dataset))):
    image, label, image_name = dataset[idx]

    # Load original image
    original_path = os.path.join(IMAGE_DIR, image_name)
    original_image = Image.open(original_path).convert('RGB')

    # Generate CAM
    cam = generate_cam(model, image, target_class=label)

    # Overlay and save
    cam_overlay = overlay_heatmap(cam, original_image)
    save_path = os.path.join(CAM_DIR, image_name.replace('.jpg', '_cam_overlay.jpg'))
    cam_overlay.save(save_path)
    print(f"[{idx+1}/{NUM_SAMPLES}] ✅ Saved to: {save_path}")
