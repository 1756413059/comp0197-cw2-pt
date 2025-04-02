import torch
import torch.nn.functional as F
import sys
import numpy as np

# def generate_grad_cam(model, image_tensor, target_class, final_conv_layer='layer4', negative=False):
#     '''
#     Arguments:
#         model: PyTorch model
#         image_tensor: Tensor of shape [C, H, W]
#         target_class: Index of target class, when negative=False it is the label class, when negative=True it is the false class
#         final_conv_layer: Name of final convolutional layer
#         negative: Boolean to invert the cam, i.e. find the features that suppress the target class the most
#     Returns:
#         cam: Class activation map of shape [H, W]
#     '''
#     model.eval()
#     feature_maps = []
#     gradients = []

#     # Hook to get feature maps
#     def forward_hook(module, input, output):
#         feature_maps.append(output)

#     # Hook to get gradients w.r.t. feature maps
#     def backward_hook(module, grad_in, grad_out):
#         gradients.append(grad_out[0])

#     # Register both hooks
#     module = dict(model.named_modules())[final_conv_layer]
#     fwd_handle = module.register_forward_hook(forward_hook)
#     bwd_handle = module.register_backward_hook(backward_hook)

#     # Forward pass
#     output = model(image_tensor.unsqueeze(0))  # [1, num_classes]

#     # Zero grads and backward for specific class
#     model.zero_grad()
#     class_score = output[0, target_class]
#     class_score.backward()

#     # Remove hooks
#     fwd_handle.remove()
#     bwd_handle.remove()

#     # Get captured features and gradients
#     fmap = feature_maps[0].squeeze(0)      # [C, H, W]
#     grads = gradients[0].squeeze(0)        # [C, H, W]

#     # Compute channel-wise weights via global average pooling of gradients
#     weights = grads.mean(dim=(1, 2))       # [C]

#     # Weighted combination
#     cam = torch.zeros_like(fmap[0])
#     for i, w in enumerate(weights):
#         cam += w * fmap[i]

#     if negative:
#         cam = -cam

#     # ReLU and normalize
#     cam = F.relu(cam)
#     cam -= cam.min()
#     cam /= cam.max() + 1e-8

#     # Upsample to input size
#     cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
#     cam = cam.squeeze()

#     return cam.detach().cpu().numpy()


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




def generate_grad_cam(
    model, 
    image_tensor, 
    target_class,
    final_conv_layer,
    gather_classes,
    aggregate_fn=lambda x:np.mean(x, axis=0)
    # aggregate_fn=lambda x: np.maximum.reduce(x)
):
    '''
    Arguments:
        model: PyTorch model
        image_tensor: Tensor of shape [C, H, W]
        target_class: Target class index (required)
        final_conv_layer: Name of final convolutional layer
        gather_classes: List of class indices to aggregate
        aggregate_fn: Aggregation function: List[np.ndarray] → np.ndarray
    Returns:
        Aggregated heatmap of shape [224, 224] of:
        - Positive activation for target_class
        - Negative activation for other classes
    '''
    model.eval()
    feature_maps = []
    gradients = []

    # make sure gather_classes includes target_class
    if target_class not in gather_classes:
        gather_classes.append(target_class)

    # Hooks
    def forward_hook(module, input, output):
        feature_maps.clear()  # Ensure only one feature map exists
        feature_maps.append(output)
    def backward_hook(module, grad_in, grad_out):
        gradients.clear()
        gradients.append(grad_out[0])

    # print("named_modules: ", dict(model.named_modules()).keys())
    # sys.exit()


    module = dict(model.named_modules())[final_conv_layer]
    fwd_handle = module.register_forward_hook(forward_hook)
    bwd_handle = module.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor.unsqueeze(0))  # [1, num_classes]
    num_classes = output.shape[1]

    # Initialize CAM tensor
    # cams = torch.zeros((len(gather_classes), *feature_maps[0].shape[2:]), 
    #                   device=image_tensor.device)
    cams=[]
    target_cls_cam = None

    # Compute gradients for gather_classes
    for class_idx in gather_classes:
        model.zero_grad()

        gradients = []
        output[0, class_idx].backward(retain_graph=True)
        
        fmap = feature_maps[-1].squeeze(0)  # [C, H, W]
        grads = gradients[-1].squeeze(0)  # [C, H_fmap, W_fmap]
        weights = grads.mean(dim=(1, 2))  # [C]
        
        # Weighted feature combination
        cam = torch.zeros_like(fmap[0])
        for i, w in enumerate(weights):
            cam += w * fmap[i]


        
        # # Negate for non-target classes
        if class_idx != target_class:
            cam = -cam

        ####
        # ReLU and normalize
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Upsample to input size (MobileNetV3 uses 224x224 by default)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=image_tensor.shape[-2:], mode='bilinear', align_corners=False)
        
        if class_idx == target_class:
            target_cls_cam = cam.squeeze().detach().cpu().numpy()
        else:
            cams.append(cam.squeeze().detach().cpu().numpy())

    # # strategy 1: aggregate the negative cams and max with the positive cam
    # # aggregate the negative cams and max with the positive cam
    if len(cams) == 0:
        cams = target_cls_cam
    else:
        cams = aggregate_fn(cams)
        # optinally normalize the negative cams
        # cams = (cams - cams.min()) / (cams.max() - cams.min() + 1e-8)
        # target_cls_cam = (target_cls_cam - target_cls_cam.min()) / (target_cls_cam.max() - target_cls_cam.min() + 1e-8)
        # max with the positive cam
        cams = np.maximum.reduce([cams, target_cls_cam])

    # # strategy 2: aggregate all cams by averaging
    # cams.append(target_cls_cam)
    # cams = aggregate_fn(cams)
    # cams = (cams - cams.min()) / (cams.max() - cams.min() + 1e-8)

    # # strategy 3: average all cams by maxing
    # cams.append(target_cls_cam)
    # cams = np.maximum.reduce(cams)
    # cams = (cams - cams.min()) / (cams.max() - cams.min() + 1e-8)




    # cams = torch.stack(cams, dim=0)

    # Cleanup
    fwd_handle.remove()
    bwd_handle.remove()

    # # ReLU and normalize per-CAM
    # cams = F.relu(cams)
    # cams = (cams - cams.min(dim=1, keepdim=True)[0]) / \
    #        (cams.max(dim=1, keepdim=True)[0] + 1e-8)

    # # Upsample 
    # cams = F.interpolate(cams.unsqueeze(1), size=(224, 224), 
    #                     mode='bilinear', align_corners=False)
    # # torch.Size([6, 1, 224, 224])
    
    # # gather classes
    # cams = aggregate_fn(cams).detach().cpu().numpy()
    # # (1, 224, 224)

    
    return cams