import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from utils.model import get_segmentor
from utils.dataset import PetDataset
from scripts.config import IMAGE_DIR, CHECKPOINT_DIR, PRED_DIR



def predict_and_save_masks(
    model_name='unet',
    model_path=None,
    test_list_file=None,
    image_dir=IMAGE_DIR,
    output_dir=PRED_DIR,
    threshold=0.5,
    res = 18,
    cam = 0.5,
    epochs = 15
):
    """
    Perform segmentation inference using a trained model and save predicted masks.

    This function loads a trained segmentation model (UNet or DeepLabV3),
    performs forward inference on a set of test images, and saves the predicted
    binary masks to the specified output directory.

    Args:
        model_name (str): Type of segmentation model to use. Options: 'unet' or 'deeplabv3'.
        model_path (str): Path to the trained model weights (.pth file). If None, it will be auto-constructed
                          using `res`, `cam`, and `epochs`.
        test_list_file (str): Path to list file containing test image names (e.g., list.txt or test.txt).
        image_dir (str): Directory containing the input images.
        output_dir (str): Directory to save the predicted mask PNG files.
        threshold (float): Sigmoid threshold (in [0, 1]) used to binarize output logits into masks.
        res (int): Backbone identifier (for checkpoint filename construction, e.g., 18 or 50).
        cam (float): CAM threshold used during pseudo-label generation (for filename).
        epochs (int): Number of training epochs (used to locate model checkpoint).

    Returns:
        None. Saves binary mask images (0 or 255) with filenames matching original images.

    Notes:
        - The model is assumed to output logits in [0, 1] due to sigmoid activation.
        - Output masks are resized back to original image resolution using nearest interpolation.
        - Saved filenames will be in format: `xxx_pred.png`.

    Example:
        >>> predict_and_save_masks(
                model_name='unet',
                model_path='checkpoints/unet_seg_resnet50_cam_0.5_epoch_15.pth',
                test_list_file='data/annotations/test.txt',
                threshold=0.5
            )
    """
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    model = get_segmentor(model_name=model_name, num_classes=1).to(device)

    if model_path is None:
        model_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_seg_res{res}_cam_{cam}_epoch_{epochs}.pth")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = PetDataset(image_dir, test_list_file, transform=transform)
    print(f"Loaded {len(dataset)} images from {image_dir}")

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx in range(len(dataset)):
            image = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)

            output = model(image_tensor)
            if model_name == 'deeplabv3':
                output = output['out']
                probs = torch.sigmoid(output)  
            else:
                probs = output  

            pred_mask = probs.squeeze().cpu().numpy()  # [H, W]
            pred_mask = (pred_mask > threshold).astype(np.uint8) * 255

            # Resize back to original
            original_filename = dataset.samples[idx][0]
            original_path = os.path.join(image_dir, original_filename)
            original_image = Image.open(original_path).convert('RGB')
            original_size = original_image.size

            resized_mask = Image.fromarray(pred_mask, mode='L').resize(original_size, resample=Image.NEAREST)

            # Save
            save_path = os.path.join(output_dir, original_filename.replace('.jpg', '_pred.png'))
            resized_mask.save(save_path)

    print(f"Saved predicted masks to: {output_dir}")

if __name__ == '__main__':
    from scripts.config import TEST_LIST_FILE

    predict_and_save_masks(
        model_name='unet',
        model_path=os.path.join(CHECKPOINT_DIR, 'unet_seg_resnet18_cam_otsu_epoch_30.pth'),
        test_list_file=TEST_LIST_FILE,
        threshold=0.5
    )
