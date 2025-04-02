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

from scripts.config import IMAGE_DIR, TRAIN_FILE, CHECKPOINT_DIR, CAM_DIR, DEVICE
from utils.dataset import PetClassificationDataset
from utils.model import get_resnet18, get_mobilenet_v3_small, get_resnet50
from utils.grad_cam_utils import generate_grad_cam, generate_grad_cam_mobilenet
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
    if cam_array.ndim == 3 and cam_array.shape[0] == 1:
        cam_array = cam_array.squeeze(0)
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


if __name__ == '__main__':
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
    dataset = PetClassificationDataset(IMAGE_DIR, TRAIN_FILE, transform=transform)
    print("len(dataset): ", len(dataset))

    for i in range(10):
        model_name = "mobilenet" # "resnet"
        image_tensor, label, image_name = dataset[i*100 + 1]
        # print("label: ", label)

        # === Load model
        if model_name == "resnet18":
            final_conv_layer = 'layer4'
            model = get_resnet18(num_classes=37)
            state_dict = torch.load(os.path.join(CHECKPOINT_DIR, 'resnet18_cls_epoch5.pth'), map_location=DEVICE)
        elif model_name == "resnet50":
            final_conv_layer = 'layer4'
            model = get_resnet50()
            state_dict=(torch.load(os.path.join(CHECKPOINT_DIR, 'resnet50_epoch4.pth'), map_location=DEVICE))
        elif model_name == "mobilenet":
            final_conv_layer = 'features.12.0'
            model = get_mobilenet_v3_small(37)
            state_dict = torch.load(os.path.join(CHECKPOINT_DIR, 'mobilenet_epoch9.pth'), map_location=DEVICE)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        model.load_state_dict(state_dict)




        # choose chose a few false classes (included_classes) to to negative grad-CAM
        # when included_classes is empty, algorithm reduces to normal grad-CAM

        # included_classes = [0,2,7,16,19,37]
        included_classes = [0,2,7,16,20,21]
        included_classes = [i for i in range(0,37)]
        # included_classes = []

        # === Generate CAM
        cam = generate_grad_cam(
            model, 
            image_tensor, 
            target_class=label, 
            final_conv_layer=final_conv_layer,
            gather_classes=included_classes,
        )

        # cam=generate_grad_cam_mobilenet(model, image_tensor, label, negative=False)

        # grad_cam_list = [
        #     # negative grad-CAM for false classes
        #     generate_cam_fn(model, image_tensor, target_class=c, negative=True)
        #     for c in range(0, 37) if (c not in excluded_classes) and (c != label)
        # ]
        # # positive grad-CAM for the true labelled class 
        # grad_cam_list.append(generate_cam_fn(model, image_tensor, target_class=label, negative=False))
        # grad_cam_maxed = np.maximum.reduce(grad_cam_list)


        # === Convert everything
        original_image = denormalize_image(image_tensor)
        heatmap = cam_to_heatmap(cam)
        overlay = overlay_image_and_heatmap(original_image, heatmap, alpha=0.5)

        # # === Save original image
        # save_path = os.path.join(CAM_DIR, f'{image_name}.jpg')
        # original_image.save(save_path)

        # === Save overlay
        save_path = os.path.join(CAM_DIR, f'{image_name}_neggradcam_overlay_{model_name}_average.jpg')
        if len(included_classes)==1:
            save_path = os.path.join(CAM_DIR, f'{image_name}_gradcam_overlay_{model_name}.jpg')
        overlay.save(save_path)
        print(f"✅ neg-grad-CAM overlay img saved to: {save_path}")
