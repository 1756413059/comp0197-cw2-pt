import torch
import torch.nn.functional as F

def generate_cam(model, image_tensor, target_class, final_conv_layer='layer4'):
    """
    Generate a Class Activation Map (CAM) for a given image and target class.

    Args:
        model (torch.nn.Module): A trained CNN classification model (e.g., ResNet).
        image_tensor (torch.Tensor): Preprocessed input image of shape [3, H, W].
        target_class (int): Class index (0-based) for which to compute the CAM.
        final_conv_layer (str): Name of the final convolutional layer from which to extract features (default: 'layer4').

    Returns:
        np.ndarray: A 2D CAM heatmap of shape [224, 224], normalized to range [0, 1].

    Notes:
        - The CAM is computed as a weighted sum of the feature maps from the last convolutional layer,
          using the weights from the fully connected (fc) layer corresponding to the target class.
        - The returned CAM is upsampled to 224x224 using bilinear interpolation.
        - The model must be a standard architecture with a named `fc` layer and a known conv block (e.g., ResNet).
        - The input image should already be normalized and resized.

    Example:
        cam = generate_cam(model, image_tensor, target_class=5)
    """

    model.eval()
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output)

    hook = dict(model.named_modules())[final_conv_layer].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0))  

    hook.remove()

    fc_weights = model.fc.weight[target_class]  

    fmap = feature_maps[0].squeeze(0)

    cam = torch.zeros(fmap.shape[1:], dtype=torch.float32)
    for i, w in enumerate(fc_weights):
        cam += w * fmap[i, :, :]

    cam = F.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    # Upsample CAM to match image size
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze()

    return cam.detach().cpu().numpy()
