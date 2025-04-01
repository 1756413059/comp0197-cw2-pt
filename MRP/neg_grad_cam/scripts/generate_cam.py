import os
import sys
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add project root for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, LIST_FILE, CHECKPOINT_DIR, CAM_DIR
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18
from utils.grad_cam_utils import generate_grad_cam
# from utils.cam_utils import generate_cam

def denormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = tensor * std + mean
    image = image.clamp(0, 1)
    image = image.mul(255).byte().numpy()
    image = np.transpose(image, (1, 2, 0))  # CHW → HWC
    return Image.fromarray(image)

def cam_to_heatmap(cam_array):
    # Normalize cam to 0–1 if needed
    cam_array = cam_array - cam_array.min()
    cam_array = cam_array / (cam_array.max() + 1e-8)

    colormap = cm.get_cmap('jet')
    heatmap = colormap(cam_array)[:, :, :3]  # Drop alpha
    heatmap = np.uint8(heatmap * 255)
    return Image.fromarray(heatmap)

def overlay_image_and_heatmap(image, heatmap, alpha=0.5):
    heatmap = heatmap.resize(image.size)
    return Image.blend(image.convert("RGB"), heatmap.convert("RGB"), alpha)

# === Create output dir
os.makedirs(CAM_DIR, exist_ok=True)

# === Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load one image
dataset = PetClassificationDataset(IMAGE_DIR, LIST_FILE, transform=transform, train_only=True)
print("len(dataset): ", len(dataset))
image_tensor, label, image_name = dataset[99]
# print("label: ", label)

# === Load model
model = get_resnet18(num_classes=37)
state_dict = torch.load(os.path.join(CHECKPOINT_DIR, 'resnet18_cls.pth'), map_location='cpu')
model.load_state_dict(state_dict)


# excluded_classes = [i for i in range(0,37)]
# include_classes = [2,33,36]
# excluded_classes = list(set(excluded_classes) - set(include_classes))
excluded_classes = [label]


# === Generate CAM
grad_cam_list = [
    generate_grad_cam(model, image_tensor, target_class=c, negative=True)
    for c in range(0, 37) if c not in excluded_classes
]
grad_cam_list.append(generate_grad_cam(model, image_tensor, label, negative=False))
grad_cam_maxed = np.maximum.reduce(grad_cam_list)


# === Convert everything
original_image = denormalize_image(image_tensor)
heatmap = cam_to_heatmap(grad_cam_maxed)
overlay = overlay_image_and_heatmap(original_image, heatmap, alpha=0.5)

# === Save overlay
save_path = os.path.join(CAM_DIR, f'{image_name}_negradcam_overlay.jpg')
overlay.save(save_path)
print(f"✅ neg-grad-CAM overlay saved to: {save_path}")
