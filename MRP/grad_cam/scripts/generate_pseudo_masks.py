import os
import sys
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from generate_cam import denormalize_image, cam_to_heatmap, overlay_image_and_heatmap
import tqdm
import copy

# Add project root for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import IMAGE_DIR, TRAIN_FILE, CHECKPOINT_DIR, MASK_DIR, TRIMAP_DIR
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18, get_mobilenet_v3_small, get_resnet50
from utils.grad_cam_utils import generate_grad_cam
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


model_name = "mobilenet"
# model_name = "mobilenet"

if model_name == "resnet18":
    final_conv_layer = 'layer4'
    model = get_resnet18(num_classes=37)
    state_dict = torch.load(os.path.join(CHECKPOINT_DIR, 'resnet18_cls_epoch5.pth'), map_location='cpu')
elif model_name == "resnet50":
    final_conv_layer = 'layer4'
    model = get_resnet50()
    state_dict=(torch.load(os.path.join(CHECKPOINT_DIR, 'resnet50_epoch4.pth'), map_location='cpu'))
elif model_name == "mobilenet":
    final_conv_layer = 'features.12.0'
    model = get_mobilenet_v3_small(37)
    state_dict = torch.load(os.path.join(CHECKPOINT_DIR, 'mobilenet_epoch9.pth'), map_location='cpu')
else:
    raise ValueError(f"Invalid model name: {model_name}")

model.load_state_dict(state_dict)

compute_mIoU = True
save_cam = True


num_samples = 40

ds_len = 1000 # len(dataset)

included_classes = [i for i in range(0,37)]
# included_classes = []

# if len(included_classes):
#     threshold = 0.3
# else:
#     threshold = 0.25

threshold_sweep = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
mIoU = [0] * len(threshold_sweep)


# # Generate pseudo masks
# for c, (image_tensor, label, image_name) in enumerate(dataset):
#     if c == num_samples:
#         break
for c in tqdm.tqdm(range(num_samples)):
    image_tensor, label, image_name = dataset[ds_len//num_samples * c]


    # === Generate CAM
    cam = generate_grad_cam(
        model, 
        image_tensor, 
        target_class=label, 
        final_conv_layer=final_conv_layer,
        gather_classes=copy.deepcopy(included_classes),
    )
    # grad_cam_list = [
    #     # negative grad-CAM for false classes
    #     generate_cam_fn(model, image_tensor, target_class=c)
    #     for c in range(0, 37) if (c not in excluded_classes) and (c != label)
    # ]
    # # positive grad-CAM for the true labelled class 
    # grad_cam_list.append(generate_cam_fn(model, image_tensor, target_class=label))
    # grad_cam_maxed = np.maximum.reduce(grad_cam_list)

    if save_cam:
        original_image = denormalize_image(image_tensor)
        heatmap = cam_to_heatmap(cam)
        overlay = overlay_image_and_heatmap(original_image, heatmap, alpha=0.5)
        if len(included_classes):
            overlay.save(os.path.join(MASK_DIR, f'{image_name}_negative_gradcam_heatmap_{model_name}.png'))
        else:
            overlay.save(os.path.join(MASK_DIR, f'{image_name}_gradcam_heatmap_{model_name}.png'))


    # mask = cam_to_mask(grad_cam_maxed, threshold=threshold)
    mask_sweep = [cam_to_mask(cam, threshold=t,keep_largest_cluster=True) for t in threshold_sweep]

    filename = os.path.splitext(image_name)[0]
    # mask_img = Image.fromarray(mask, mode='L')
    # mask_img.save(os.path.join(MASK_DIR, f'{filename}_mask_{model_name}.png'))

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

        for i, mask in enumerate(mask_sweep):
            intersection = np.logical_and(gt_mask, mask).sum()
            union = np.logical_or(gt_mask, mask).sum()
            mIoU[i] += intersection / union

        # intersection = np.logical_and(gt_mask, mask).sum()
        # union = np.logical_or(gt_mask, mask).sum()
        # mIoU += intersection / union

# makemIoU 4 decimal place
mIoU = [round(i/num_samples, 4) for i in mIoU]
print("Model name: ", model_name)
print(f"This is SuppressCAM with included classes: {included_classes}, threshold: {threshold_sweep}")
print(f"Mean IoU over {num_samples} samples: {mIoU}")


print("✅ All pseudo masks generated.")
