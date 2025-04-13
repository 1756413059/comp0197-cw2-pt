import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from transformers import AutoModel, AutoImageProcessor
from sklearn.metrics import jaccard_score, f1_score
from torchvision.models import resnet50, ResNet50_Weights


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = OxfordIIITPet(
    root='./data', split='test', target_types='segmentation',
    transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
    download=True
)

# ========== Utility Functions ==========

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

# ========== ViT + Grad-CAM ==========

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

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations, self.gradients = None, None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_idx = class_idx or output.argmax().item()
        output[:, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam_map = torch.relu((weights * self.activations).sum(1)).squeeze().cpu().numpy()
        return (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-6)

def extract_gradcam_mask(img_tensor):
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    gradcam = GradCAM(model, model.layer4[-1])
    cam_map = gradcam(img_tensor.to(device))
    cam_map = np.array(Image.fromarray((cam_map * 255).astype(np.uint8)).resize((224,224))) / 255.0
    t = otsu_threshold((cam_map * 255).astype(np.uint8))
    binary_mask = (cam_map > t).astype(np.uint8)
    cleaned = keep_largest_component(open_then_close(binary_mask))
    return cam_map, Image.fromarray(cleaned * 255).convert("L")

# ========== Evaluation & Save ==========

def detailed_metrics(pred_mask, gt_mask):
    pred = (np.array(pred_mask) > 128).astype(np.uint8)
    gt = (np.array(gt_mask) == 1).astype(np.uint8)
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = f1_score(gt.flatten(), pred.flatten())
    iou = jaccard_score(gt.flatten(), pred.flatten())
    return {'Dice': f1, 'IoU': iou, 'Precision': precision, 'Recall': recall}

def save_comparison_image(img_pil, gt_mask, vit_mask, grad_mask, idx, out_dir):
    def add_title(im, title):
        title_bar = Image.new("RGB", (im.width, 20), "white")
        draw = ImageDraw.Draw(title_bar)
        draw.text((5, 5), title, fill="black")
        return Image.fromarray(np.vstack((np.array(title_bar), np.array(im.resize((224, 224)).convert("RGB")))))

    tiles = [
        add_title(img_pil, "Original"),
        add_title(gt_mask, "GT"),
        add_title(vit_mask, "ViT"),
        add_title(grad_mask, "GradCAM"),
    ]
    out = Image.new("RGB", (224 * 4, 244))
    for i, im in enumerate(tiles):
        out.paste(im, (i * 224, 0))
    out.save(os.path.join(out_dir, "comparison", f"{idx:04d}.jpg"))

def evaluate_and_save(dataset):
    out_dir = "./results_test"
    os.makedirs(os.path.join(out_dir, "vit_masks"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "gradcam_masks"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "comparison"), exist_ok=True)
    txt_path = os.path.join(out_dir, "metrics_summary.txt")

    vit_scores, grad_scores = [], []

    for idx in range(len(dataset)):
        try:
            img_tensor, mask_tensor = dataset[idx]
            img_pil = transforms.ToPILImage()(img_tensor)
            img_tensor_unsq = img_tensor.unsqueeze(0)
            gt_mask = mask_tensor.resize((224, 224)).convert("L")

            vit_map, vit_mask = extract_vit_mask(img_pil)
            grad_map, grad_mask = extract_gradcam_mask(img_tensor_unsq)

            vit_mask.save(os.path.join(out_dir, "vit_masks", f"{idx:04d}.png"))
            grad_mask.save(os.path.join(out_dir, "gradcam_masks", f"{idx:04d}.png"))

            save_comparison_image(img_pil, gt_mask, vit_mask, grad_mask, idx, out_dir)

            vit_scores.append(detailed_metrics(vit_mask, gt_mask))
            grad_scores.append(detailed_metrics(grad_mask, gt_mask))
        except:
            continue

    def summarize(scores):
        return {
            "Dice": np.mean([s["Dice"] for s in scores]),
            "IoU": np.mean([s["IoU"] for s in scores]),
            "Precision": np.mean([s["Precision"] for s in scores]),
            "Recall": np.mean([s["Recall"] for s in scores]),
        }

    vit_summary = summarize(vit_scores)
    grad_summary = summarize(grad_scores)

    with open(txt_path, "w") as f:
        f.write("=== Average Metrics on Test Dataset ===\n")
        f.write(f"{'Method':<10} | {'Dice':<6} | {'IoU':<6} | {'Prec':<6} | {'Recall':<6}\n")
        f.write("-" * 50 + "\n")
        for name, summary in zip(["ViT", "Grad-CAM"], [vit_summary, grad_summary]):
            f.write(f"{name:<10} | {summary['Dice']:.4f} | {summary['IoU']:.4f} | {summary['Precision']:.4f} | {summary['Recall']:.4f}\n")

evaluate_and_save(dataset)

# Debugging Visualization Error Distribution (Red=FP, Blue=FN, Green=TP)
# def visualize_fp_fn(img_pil, pred_mask, gt_mask):
#     img = np.array(img_pil).copy()
#     pred = (np.array(pred_mask) > 128).astype(np.uint8)
#     gt = (np.array(gt_mask) == 1).astype(np.uint8)
#     tp = np.logical_and(pred == 1, gt == 1)
#     fp = np.logical_and(pred == 1, gt == 0)
#     fn = np.logical_and(pred == 0, gt == 1)
#     overlay = img.copy()
#     overlay[fp] = [255, 0, 0]
#     overlay[fn] = [0, 0, 255]
#     overlay[tp] = [0, 255, 0]
#     Image.fromarray(overlay).save("debug_overlay.png")
#
# # Example Use (Uncomment Call)
# # idx = 42
# # img_tensor, mask_tensor = dataset[idx]
# # img_pil = transforms.ToPILImage()(img_tensor)
# # gt_mask = mask_tensor.resize((224, 224)).convert("L")
# # _, vit_mask = extract_vit_mask(img_pil)
# # visualize_fp_fn(img_pil, vit_mask, gt_mask)
