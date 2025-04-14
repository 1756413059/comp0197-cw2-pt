# generate_pseudo_labels.py
import os
import torch
import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from transformers import AutoModel, AutoImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

def otsu_threshold(image_array):
    pixel_counts = np.bincount(image_array.flatten(), minlength=256)
    total = image_array.size
    sum_total = np.dot(np.arange(256), pixel_counts)
    sum_bg, weight_bg, max_var, threshold = 0.0, 0.0, 0.0, 0
    for t in range(256):
        weight_bg += pixel_counts[t]
        if weight_bg == 0: continue
        weight_fg = total - weight_bg
        if weight_fg == 0: break
        sum_bg += t * pixel_counts[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold / 255.0

def erode(mask, k=3):
    pad = k // 2
    padded = np.pad(mask, pad, mode='constant')
    out = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            out[i, j] = np.min(padded[i:i+k, j:j+k])
    return out

def dilate(mask, k=3):
    pad = k // 2
    padded = np.pad(mask, pad, mode='constant')
    out = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            out[i, j] = np.max(padded[i:i+k, j:j+k])
    return out

def open_then_close(mask):
    return dilate(erode(mask))

def keep_largest_component(mask):
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_region, max_size = [], 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1 and not visited[i, j]:
                stack, region = [(i, j)], [(i, j)]
                visited[i, j] = True
                while stack:
                    x, y = stack.pop()
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < H and 0 <= ny < W and not visited[nx, ny] and mask[nx, ny] == 1:
                                visited[nx, ny] = True
                                stack.append((nx, ny))
                                region.append((nx, ny))
                if len(region) > max_size:
                    max_size = len(region)
                    best_region = region
    final_mask = np.zeros_like(mask)
    for x, y in best_region:
        final_mask[x, y] = 1
    return final_mask.astype(np.uint8)

def gaussian_blur(image, kernel_size=5, sigma=1.0):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    blurred = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            blurred[i, j] = np.sum(region * kernel)
    return blurred

dino_model = AutoModel.from_pretrained("facebook/dino-vits16", output_attentions=True).to(device).eval()
dino_processor = AutoImageProcessor.from_pretrained("facebook/dino-vits16")

def extract_vit_mask(img_pil, selected_layers=[1,3,5,7,9,11]):
    w, h = img_pil.size
    inputs = dino_processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        attn = dino_model(**inputs).attentions
    attn_maps = [attn[i][0].mean(0)[0,1:] for i in selected_layers]
    cls_attn = (torch.stack(attn_maps).mean(0) + torch.stack(attn_maps).max(0).values) / 2
    side = int(cls_attn.shape[0] ** 0.5)
    attn_map = cls_attn.reshape(side, side).cpu().numpy()
    attn_map = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize((w,h))) / 255.0
    norm_attn = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-5)
    smoothed = gaussian_blur(norm_attn)
    t = max(0.12, otsu_threshold((smoothed * 255).astype(np.uint8)) * 0.75)
    binary_mask = (smoothed > t).astype(np.uint8)
    cleaned = keep_largest_component(open_then_close(binary_mask))
    return norm_attn, Image.fromarray(cleaned * 255).convert("L")

def run_generate_pseudo_labels():
    dataset = OxfordIIITPet(
        root='oxford-iiit-pet', split='trainval', target_types='segmentation',
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
        download=True
    )
    os.makedirs("output_vit/images", exist_ok=True)
    os.makedirs("output_vit/masks", exist_ok=True)
    for idx in range(len(dataset)):
        try:
            img_tensor, _ = dataset[idx]
            img_pil = transforms.ToPILImage()(img_tensor)
            _, vit_mask = extract_vit_mask(img_pil)
            orig_path = dataset._images[idx]
            orig_name = os.path.splitext(os.path.basename(orig_path))[0] 
            img_pil.save(os.path.join("output_vit/images", f"{orig_name}.jpg"))
            vit_mask.save(os.path.join("output_vit/masks", f"{orig_name}_mask.png"))

            if idx % 50 == 0:
                print(f"Saved {idx + 1}/{len(dataset)}")
        except Exception as e:
            print(f"Error on index {idx}: {e}")
