import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class CAM:
    def __init__(self, model, target_layer, device, mode):
        """
        Args:
            model: The neural network model.
            target_layer: The layer from which to capture activations and gradients.
            device: The device on which to perform computations.
            mode (str): One of 'gradcam++', 'gradcam', or 'cam'.
                - 'gradcam++' computes CAM using the Grad-CAM++ weighting scheme.
                - 'gradcam' computes CAM with weights via global average pooling of gradients.
                - 'cam' computes the original CAM using the classifier (fully-connected) layer weights.
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.mode = mode
        self.fc_layer = model.fc
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        # A forward hook to capture activations
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # A backward hook to capture gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        # Remove all hooks when they are no longer needed
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        """
        Computes the CAM for the given input tensor and target class

        Args:
            input_tensor: Input image tensor.
            class_idx (int, optional): Target class index. If not provided, uses the predicted class.

        Returns:
            A numpy array with the computed CAM map.
        """
        if self.mode == 'gradcam++':
            return self._gradcampp(input_tensor, class_idx)
        elif self.mode == 'gradcam':
            return self._gradcam(input_tensor, class_idx)
        elif self.mode == 'cam':
            return self._cam(input_tensor, class_idx)
        else:
            raise ValueError("Invalid mode. Supported modes: 'gradcam++', 'gradcam', 'cam'")

    def _gradcampp(self, input_tensor, class_idx):
        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        grads_squared = gradients ** 2
        grads_cubed = gradients ** 3
        denominator = 2 * grads_squared + activations * grads_cubed
        denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        alpha = grads_squared / (denominator + 1e-7)
        alpha = alpha.sum(dim=(1, 2), keepdim=True)

        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))
        weights = weights.view(-1, 1, 1)

        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        return cam.cpu().numpy()

    def _gradcam(self, input_tensor, class_idx):
        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        # Compute weights by global average pooling the gradients
        weights = gradients.mean(dim=(1, 2))
        weights = weights.view(-1, 1, 1)

        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        return cam.cpu().numpy()

    def _cam(self, input_tensor, class_idx):
        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Extract the weight vector corresponding to the target class
        weight = self.fc_layer.weight[class_idx]

        # Use the hooked activations from the target layer
        activations = self.activations[0]

        # Perform the weighted combination
        cam = (weight.view(-1, 1, 1) * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        return cam.detach().cpu().numpy()


def generate_and_save_cam(dataset, CAM, split, output_dir, threshold=128):
    """
    Processes a dataset to generate CAMs

    Args:
        dataset: A dataset instance; __getitem__ should return (image_tensor, label)
        CAM: An instance of CAM class
        split (str): A string indicating the dataset split ('train' or 'test')
        output_dir: Directory for saving the outputs
        threshold(int): The threshold value for the pseudo-mask
    """
    output_dir = os.path.join(output_dir, split)
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(dataset)):
        # Retrieve the preprocessed image and label
        image_tensor, label = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0)

        # Compute the low-resolution CAM
        raw_output = CAM(input_tensor)

        # Convert CAM to a tensor and add batch and channel dimensions
        cam_tensor = torch.tensor(raw_output).unsqueeze(0).unsqueeze(0)

        # Get original image dimensions
        original_filepath = dataset.data_info[idx]['image']
        with Image.open(original_filepath) as orig_img:
            original_width, original_height = orig_img.size

        # Upsample the CAM to the original image size
        cam_upscaled = F.interpolate(cam_tensor, size=(original_height, original_width), mode='bilinear',
                                     align_corners=False)
        cam_upscaled = cam_upscaled.squeeze().numpy()

        # Normalize the upscaled CAM to the range [0, 255]
        cam_norm = (cam_upscaled - cam_upscaled.min()) / (cam_upscaled.max() - cam_upscaled.min() + 1e-7)
        cam_uint8 = (cam_norm * 255).astype(np.uint8)

        # Convert numpy array to a PIL Image
        cam_image = Image.fromarray(cam_uint8)

        # Use the original image's filename for saving the CAM
        original_filename = os.path.basename(original_filepath)
        cam_filename = os.path.splitext(original_filename)[0] + ".png"

        # Save the image into the output directory
        output_path = os.path.join(output_dir, cam_filename)
        cam_image.save(output_path)