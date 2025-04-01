import os
import sys
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# Add project root for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, TRAIN_FILE, CHECKPOINT_DIR, MASK_DIR, TRIMAP_DIR
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18, get_mobilenet_v3_small
from utils.grad_cam_utils import generate_grad_cam, generate_grad_cam_mobilenet
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
dataset = PetClassificationDataset(IMAGE_DIR, TRAIN_FILE, transform=transform)


model_name = "resnet"
model_name = "mobilenet"
if model_name == "resnet":
    generate_cam_fn = generate_grad_cam
    # Load trained classifier
    model = get_resnet18(num_classes=37)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'resnet18_cls.pth'), map_location='cpu'))
elif model_name == "mobilenet":
    generate_cam_fn = generate_grad_cam_mobilenet
    # Load trained classifier
    model = get_mobilenet_v3_small()
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'mobilenet_epoch9.pth'), map_location='cpu'))


compute_mIoU = True

mIoU = 0
num_samples = 10
# Generate pseudo masks
for c, (image_tensor, label, image_name) in enumerate(dataset):
    if c == num_samples:
        break
    excluded_classes = [i for i in range(0,37)]
    included_classes = [0,2,7,16,20,21,37]
    # included_classes = []
    excluded_classes = list(set(excluded_classes) - set(included_classes))


    # === Generate CAM
    grad_cam_list = [
        # negative grad-CAM for false classes
        generate_cam_fn(model, image_tensor, target_class=c, negative=True)
        for c in range(0, 37) if (c not in excluded_classes) and (c != label)
    ]
    # positive grad-CAM for the true labelled class 
    grad_cam_list.append(generate_cam_fn(model, image_tensor, target_class=label, negative=False))
    grad_cam_maxed = np.maximum.reduce(grad_cam_list)

    mask = cam_to_mask(grad_cam_maxed, threshold=0.25)

    filename = os.path.splitext(image_name)[0]
    mask_img = Image.fromarray(mask, mode='L')
    mask_img.save(os.path.join(MASK_DIR, f'{filename}_mask.png'))

    if compute_mIoU:
        # Calculate mIoU
        trimap_path = os.path.join(TRIMAP_DIR, f'{filename}.png')
        # transform to 244x244
        trimap = Image.open(trimap_path).convert('L')
        trimap = transforms.Resize((224, 224))(trimap)
        trimap = np.array(trimap)
        # the value 1 is the foreground, extract the ground truth mask
        gt_mask = (trimap == 1).astype(np.uint8)
        # calculate the intersection and union
        intersection = np.logical_and(gt_mask, mask).sum()
        union = np.logical_or(gt_mask, mask).sum()
        mIoU += intersection / union

mIoU /= num_samples
print(f"Mean IoU over {num_samples} samples: {mIoU}")


print("✅ All pseudo masks generated.")
