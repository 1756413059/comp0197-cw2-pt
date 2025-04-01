import torch
import torch.nn.functional as F

def generate_grad_cam(model, image_tensor, target_class, final_conv_layer='layer4', negative=False):
    '''
    Arguments:
        model: PyTorch model
        image_tensor: Tensor of shape [C, H, W]
        target_class: Index of target class, when negative=False it is the label class, when negative=True it is the false class
        final_conv_layer: Name of final convolutional layer
        negative: Boolean to invert the cam, i.e. find the features that suppress the target class the most
    Returns:
        cam: Class activation map of shape [H, W]
    '''
    model.eval()
    feature_maps = []
    gradients = []

    # Hook to get feature maps
    def forward_hook(module, input, output):
        feature_maps.append(output)

    # Hook to get gradients w.r.t. feature maps
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register both hooks
    module = dict(model.named_modules())[final_conv_layer]
    fwd_handle = module.register_forward_hook(forward_hook)
    bwd_handle = module.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor.unsqueeze(0))  # [1, num_classes]

    # Zero grads and backward for specific class
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    # Remove hooks
    fwd_handle.remove()
    bwd_handle.remove()

    # Get captured features and gradients
    fmap = feature_maps[0].squeeze(0)      # [C, H, W]
    grads = gradients[0].squeeze(0)        # [C, H, W]

    # Compute channel-wise weights via global average pooling of gradients
    weights = grads.mean(dim=(1, 2))       # [C]

    # Weighted combination
    cam = torch.zeros_like(fmap[0])
    for i, w in enumerate(weights):
        cam += w * fmap[i]

    if negative:
        cam = -cam

    # ReLU and normalize
    cam = F.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    # Upsample to input size
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze()

    return cam.detach().cpu().numpy()


def generate_grad_cam_mobilenet(model, image_tensor, target_class, negative=False):
    '''
    MobileNetV3-specific Grad-CAM.
    Args:
        model: MobileNetV3 model
        image_tensor: Input tensor [C, H, W]
        target_class: Target class index
        negative: Invert CAM if True
    Returns:
        cam: Grad-CAM heatmap [H, W]
    '''
    model.eval()
    feature_maps = []
    gradients = []

    # Hook to capture feature maps (MobileNetV3-Small: 'features.12.block.2')
    def forward_hook(module, input, output):
        feature_maps.append(output)

    # Hook to capture gradients BEFORE SE layer
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])  # Gradients before SE

    # Register hooks on the final conv block
    final_conv_name = 'features.12.0'
    # print(dict(model.named_modules()).keys())
    module = dict(model.named_modules())[final_conv_name]
    fwd_handle = module.register_forward_hook(forward_hook)
    bwd_handle = module.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor.unsqueeze(0))  # [1, num_classes]

    # Backward pass for target class
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    # Remove hooks
    fwd_handle.remove()
    bwd_handle.remove()

    # Compute CAM
    fmap = feature_maps[0].squeeze(0)  # [C, H, W]
    grads = gradients[0].squeeze(0)    # [C, H, W]
    weights = grads.mean(dim=(1, 2))   # [C]

    cam = torch.zeros_like(fmap[0])
    for i, w in enumerate(weights):
        cam += w * fmap[i]

    if negative:
        cam = -cam

    # ReLU and normalize
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Upsample to input size (MobileNetV3 uses 224x224 by default)
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=image_tensor.shape[-2:], mode='bilinear', align_corners=False)
    return cam.squeeze().detach().cpu().numpy()