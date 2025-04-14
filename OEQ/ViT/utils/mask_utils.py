import numpy as np

def cam_to_mask(cam, threshold=0.25):
    """
    Convert normalized CAM [H, W] (values in [0, 1]) → binary mask [H, W] (0 or 255)
    """
    assert isinstance(cam, np.ndarray), "CAM must be a numpy array"
    assert cam.max() <= 1.0 and cam.min() >= 0.0, "CAM should be normalized"

    mask = np.uint8(cam > threshold) * 255
    return mask  # dtype: uint8, values: 0 (bg), 255 (fg)

def otsu_threshold(image_array):
    """
    image_array: CAM map, float32 in [0, 1]
    return: int threshold in range [0, 255]
    """
    # Convert CAM float image to uint8 [0, 255]
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

    # Convert threshold back to float in [0, 1]
    return threshold / 255.0
