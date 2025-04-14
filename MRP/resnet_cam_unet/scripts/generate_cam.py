import os
import torch
import numpy as np
from PIL import Image
from matplotlib import cm
from torchvision import transforms

from scripts.config import IMAGE_DIR, CHECKPOINT_DIR, CAM_DIR, TRAIN_LIST_FILE
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18
from utils.cam_utils import generate_cam


def overlay_heatmap(cam_array, image_pil, alpha=0.5):

    cam_uint8 = np.uint8(cam_array * 255)
    heatmap = cm.jet(cam_uint8)[:, :, :3] * 255  # RGB only
    heatmap = heatmap.astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).resize(image_pil.size)

    blended = Image.blend(image_pil.convert("RGB"), heatmap_img, alpha=alpha)
    return blended


def generate_cam_overlays(
    model_path=os.path.join(CHECKPOINT_DIR, 'resnet18_cls_epoch_10.pth'),
    list_file=TRAIN_LIST_FILE,
    save_dir=CAM_DIR,
    num_samples=float('inf')
):
    os.makedirs(save_dir, exist_ok=True)
    print(f"📸 Saving CAM overlays to {save_dir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = PetClassificationDataset(IMAGE_DIR, list_file, transform=transform)

    model = get_resnet18(num_classes=37)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    for idx in range(min(int(num_samples), len(dataset))):
        image_tensor, label, image_name = dataset[idx]

        original_path = os.path.join(IMAGE_DIR, image_name)
        original_image = Image.open(original_path).convert('RGB')

        cam = generate_cam(model, image_tensor, target_class=label)

        cam_overlay = overlay_heatmap(cam, original_image)
        save_path = os.path.join(save_dir, image_name.replace('.jpg', '_cam_overlay.jpg'))
        cam_overlay.save(save_path)
        print(f"[{idx+1}] Saved to: {save_path}")

if __name__ == "__main__":

    generate_cam_overlays(
        model_path=os.path.join(CHECKPOINT_DIR, 'resnet18_cls_epoch_10.pth'),
        list_file=TRAIN_LIST_FILE,
        save_dir=CAM_DIR,
        num_samples=float('inf')  
    )