import os
import sys
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# Add project root for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, LIST_FILE, CHECKPOINT_DIR, MASK_DIR
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18
from utils.cam_utils import generate_cam
from utils.mask_utils import cam_to_mask

# Create output folder if needed
os.makedirs(MASK_DIR, exist_ok=True)

# Transform (same as classifier training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset (training only)
dataset = PetClassificationDataset(IMAGE_DIR, LIST_FILE, transform=transform, train_only=True)

# Load trained classifier
model = get_resnet18(num_classes=37)
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'resnet18_cls.pth'), map_location='cpu'))

# Generate pseudo masks
for image_tensor, label, image_name in dataset:
    cam = generate_cam(model, image_tensor, target_class=label)
    mask = cam_to_mask(cam, threshold=0.25)

    filename = os.path.splitext(image_name)[0]
    mask_img = Image.fromarray(mask, mode='L')
    mask_img.save(os.path.join(MASK_DIR, f'{filename}_mask.png'))

print("✅ All pseudo masks generated.")
