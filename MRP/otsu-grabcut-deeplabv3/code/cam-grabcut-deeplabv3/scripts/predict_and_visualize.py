import os
import sys
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from scripts.config import IMAGE_DIR, TEST_LIST_FILE, CHECKPOINT_DIR, PRED_DIR, CROP_DIR, GRABCUT_DIR
from utils.model import get_segmentor
from utils.dataset import PetDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


model_name = 'deeplabv3' 
# model_name = 'unet'  
model_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_seg_epoch_20.pth')
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = get_segmentor(model_name=model_name, num_classes=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
dataset = PetDataset(IMAGE_DIR, TEST_LIST_FILE, transform=transform)
print(f"Loaded {len(dataset)} images from {IMAGE_DIR} using {TEST_LIST_FILE}")

with torch.no_grad():
    for idx in range(len(dataset)): 
        image = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)

        if model_name == 'deeplabv3':
            output = model(image_tensor)['out']
        else:
            output = model(image_tensor)

        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        original_filename = dataset.samples[idx][0]
        original_path = os.path.join(IMAGE_DIR, original_filename)
        original_image = Image.open(original_path).convert('RGB')
        original_size = original_image.size  
        resized_mask = Image.fromarray(pred_mask, mode='L').resize(original_size, resample=Image.NEAREST)

        save_path = os.path.join(PRED_DIR, original_filename.replace('.jpg', '_pred.png'))
        resized_mask.save(save_path)

        original_np = np.array(original_image)
        mask_np = np.array(resized_mask)
        foreground = original_np * (mask_np[..., None] // 255)

        crop_save_path = os.path.join(CROP_DIR, original_filename.replace('.jpg', '_crop.png'))
        Image.fromarray(foreground).save(crop_save_path)

print("Saved prediction masks to outputs/preds/")
