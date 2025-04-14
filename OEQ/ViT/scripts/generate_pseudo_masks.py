import os
import sys
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, CHECKPOINT_DIR, MASK_DIR, TRAIN_LIST_FILE
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18, get_resnet50
from utils.cam_utils import generate_cam
from utils.mask_utils import cam_to_mask, otsu_threshold


def generate_pseudo_masks(
    threshold=0.5,
    save_dir=MASK_DIR,
    list_file=TRAIN_LIST_FILE,
    model_name='resnet18',
    checkpoint_name=None
):
    print(f"Generating pseudo masks using {model_name} | threshold = {threshold if isinstance(threshold, float) else threshold.__name__}")
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = PetClassificationDataset(IMAGE_DIR, list_file, transform=transform)

    if model_name == 'resnet18':
        model = get_resnet18(num_classes=37)
        ckpt_name = checkpoint_name or 'resnet18_cls_epoch_10.pth'
    elif model_name == 'resnet50':
        model = get_resnet50(num_classes=37)
        ckpt_name = checkpoint_name or 'resnet50_cls_epoch_10.pth'
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    state_dict = torch.load(model_path, map_location='mps' if torch.backends.mps.is_available() else 'cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    for image_tensor, label, image_name in dataset:
        cam = generate_cam(model, image_tensor, target_class=label)
        th = threshold(cam) if callable(threshold) else threshold
        mask = cam_to_mask(cam, threshold=th)

        filename = os.path.splitext(image_name)[0]
        mask_img = Image.fromarray(mask, mode='L')
        mask_img.save(os.path.join(save_dir, f'{filename}_mask.png'))

    print(f"Pseudo masks saved to {save_dir}")


if __name__ == '__main__':
    generate_pseudo_masks(
        threshold=0.5,
        model_name='resnet50',
        checkpoint_name='resnet50_cls_epoch_10.pth'
    )
