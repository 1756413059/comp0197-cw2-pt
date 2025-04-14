import torch
import torch.nn.functional as F
import numpy as np

def generate_cam(model, image_tensor, target_class, final_conv_layer='layer4'):
    model.eval()
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output)

    # Register hook to save output of final conv layer
    hook = dict(model.named_modules())[final_conv_layer].register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0))  # Add batch dimension

    hook.remove()

    # Get weights from last linear layer (classifier)
    fc_weights = model.fc.weight[target_class]  # shape: [C]

    # Feature map shape: [1, C, H, W] → [C, H, W]
    fmap = feature_maps[0].squeeze(0)

    # Compute weighted sum over channels
    cam = torch.zeros(fmap.shape[1:], dtype=torch.float32)
    for i, w in enumerate(fc_weights):
        cam += w * fmap[i, :, :]

    # ReLU and normalize
    cam = F.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    # Upsample CAM to match image size
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze()

    return cam.detach().cpu().numpy()
