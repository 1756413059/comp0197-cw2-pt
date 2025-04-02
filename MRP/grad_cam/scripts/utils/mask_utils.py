
import numpy as np

import torch
import torchvision.ops as ops
import cv2

def get_largest_cluster(binary_tensor):
    binary_tensor = (binary_tensor > 0).to(torch.uint8)
    binary_np = binary_tensor.cpu().numpy()
    
    num_labels, labeled = cv2.connectedComponents(binary_np, connectivity=4)
    if num_labels <= 1:  # Only background (0)
        return torch.zeros_like(binary_tensor)
    
    areas = torch.bincount(torch.tensor(labeled.flatten()))[1:]  # Skip background
    largest_label = torch.argmax(areas) + 1
    output = torch.tensor(labeled == largest_label, device=binary_tensor.device)
    return output.to(torch.uint8)

# def get_largest_cluster(binary_tensor):
#     """
#     Input: Binary tensor (0=black, 1=white) of shape (H, W).
#     Output: Binary tensor with only the largest white cluster retained.
#     """
#     # Label connected components (custom function since torchvision lacks direct CC)
#     labeled, num_labels = label_components(binary_tensor)
    
#     if num_labels == 0:
#         raise ValueError("No connected components found")
#         # return binary_tensor  # No white pixels
    
#     # Compute area of each label
#     areas = torch.bincount(labeled.flatten())[1:]  # Skip background (0)
#     largest_label = torch.argmax(areas) + 1  # +1 to adjust for background
    
#     # Create output mask
#     output = (labeled == largest_label).to(torch.uint8)
#     return output

def label_components(binary_tensor):
    """Custom connected-components labeling for PyTorch (4-connectivity)."""
    device = binary_tensor.device
    h, w = binary_tensor.shape
    labeled = torch.zeros_like(binary_tensor)
    current_label = 1
    
    for i in range(h):
        for j in range(w):
            if binary_tensor[i, j] == 1 and labeled[i, j] == 0:
                # Flood fill (BFS)
                queue = [(i, j)]
                labeled[i, j] = current_label
                while queue:
                    x, y = queue.pop(0)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-connectivity
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if binary_tensor[nx, ny] == 1 and labeled[nx, ny] == 0:
                                labeled[nx, ny] = current_label
                                queue.append((nx, ny))
                current_label += 1
    
    return labeled, current_label - 1

if __name__ == "__main__":
    # Example usage
    binary_image = np.array([
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 1]
    ], dtype=np.uint8)

    result = get_largest_cluster(torch.tensor(binary_image))
    print("Largest cluster kept:\n", result)

def cam_to_mask(cam, threshold=0.25, keep_largest_cluster=False):
    """
    Convert normalized CAM [H, W] (values in [0, 1]) → binary mask [H, W] (0 or 255)
    """
    assert isinstance(cam, np.ndarray), "CAM must be a numpy array"
    assert cam.max() <= 1.0 and cam.min() >= 0.0, "CAM should be normalized"

    mask = np.uint8(cam > threshold) # * 255

    if keep_largest_cluster:
        mask = get_largest_cluster(torch.tensor(mask)).cpu().numpy()

    return mask  # dtype: uint8, values: 0 (bg), 255 (fg)
