import numpy as np

def cam_to_mask(cam, threshold=0.25):
    """
    Convert normalized CAM [H, W] (values in [0, 1]) → binary mask [H, W] (0 or 255)
    """
    assert isinstance(cam, np.ndarray), "CAM must be a numpy array"
    assert cam.max() <= 1.0 and cam.min() >= 0.0, "CAM should be normalized"

    mask = np.uint8(cam > threshold) * 255
    return mask  
