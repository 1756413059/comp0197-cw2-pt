import torch
import torch.nn.functional as F

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

def generate_gradcam_pp(model, input_tensor, target_class, target_layer_name='layer4'):
    """
    Generate Grad-CAM++ heatmap for a given input image and model.

    Args:
        model: Pretrained classification model (e.g., ResNet).
        input_tensor: Input image tensor of shape [3, H, W] (no batch dim).
        target_class: Integer class label.
        target_layer_name: Name of the conv layer (e.g., 'layer4').

    Returns:
        cam: numpy array [H, W], values in [0, 1]
    """
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Register hooks
    for name, module in model.named_modules():
        if name == target_layer_name:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
            break
    else:
        raise ValueError(f"Layer {target_layer_name} not found in model.")

    # Add batch dim
    input_tensor = input_tensor.unsqueeze(0).requires_grad_()

    # Forward + backward
    output = model(input_tensor)
    target = output[0, target_class]
    model.zero_grad()
    target.backward()

    A = activations[0]      # shape: [1, C, H, W]
    dY = gradients[0]       # shape: [1, C, H, W]

    # Grad-CAM++ formula
    grads_power_2 = dY ** 2
    grads_power_3 = dY ** 3
    sum_A = A.sum(dim=(2, 3), keepdim=True)

    eps = 1e-8
    alpha_num = grads_power_2
    alpha_denom = 2 * grads_power_2 + sum_A * grads_power_3
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.tensor(eps))
    alphas = alpha_num / alpha_denom
    weights = (alphas * F.relu(dY)).sum(dim=(2, 3), keepdim=True)

    cam = F.relu((weights * A).sum(1, keepdim=True))  # [1, 1, H, W]
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)

    return cam

