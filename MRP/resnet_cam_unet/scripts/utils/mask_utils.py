import numpy as np

def cam_to_mask(cam, threshold=0.25):
    """
    Converts a normalized Class Activation Map (CAM) to a binary mask.

    Args:
        cam (np.ndarray): A 2D CAM array with values in [0, 1].
        threshold (float): A float threshold in [0, 1]. Pixels with CAM > threshold are foreground.

    Returns:
        np.ndarray: A binary mask with shape [H, W], values in {0, 255} (uint8).
                    255 represents foreground, 0 represents background.

    Raises:
        AssertionError: If `cam` is not a numpy array or not normalized to [0, 1].

    Example:
        mask = cam_to_mask(cam, threshold=0.3)
    """

    assert isinstance(cam, np.ndarray), "CAM must be a numpy array"
    assert cam.max() <= 1.0 and cam.min() >= 0.0, "CAM should be normalized"

    mask = np.uint8(cam > threshold) * 255
    return mask 
 
def otsu_threshold(image_array):
    """
    Computes an adaptive threshold using Otsu's method for a grayscale heatmap.

    Args:
        image_array (np.ndarray): 2D float array with values in [0, 1] (e.g. a CAM).

    Returns:
        float: Optimal threshold value in [0, 1] computed by Otsu's method.

    Notes:
        - The input array is converted to uint8 in range [0, 255].
        - Otsu's method maximizes the between-class variance to separate foreground and background.
        - Commonly used for automatic CAM binarization.

    Example:
        th = otsu_threshold(cam)
        mask = cam_to_mask(cam, threshold=th)
    """

    image_uint8 = np.uint8(image_array * 255)

    pixel_counts = np.bincount(image_uint8.flatten(), minlength=256)
    total = image_uint8.size
    sum_total = np.dot(np.arange(256), pixel_counts)
    sum_bg, weight_bg, max_var, threshold = 0.0, 0.0, 0.0, 0
    for t in range(256):
        weight_bg += pixel_counts[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * pixel_counts[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    return threshold / 255.0
