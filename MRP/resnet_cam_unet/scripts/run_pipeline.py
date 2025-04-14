import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_classifier import train_classifier
from scripts.generate_pseudo_masks import generate_pseudo_masks
from scripts.train_segmentor import train_segmentor
from scripts.predict_and_visualize import predict_and_save_masks
from scripts.utils.metrics import compute_metrics_for_split 

from scripts.config import (
    IMAGE_DIR, TRAIN_LIST_FILE, TEST_LIST_FILE,
    CHECKPOINT_DIR, PRED_DIR
)
from utils.mask_utils import otsu_threshold

def run_pipeline(
    model_name='unet',
    classifier_model='resnet18',
    threshold=0.5,  
    use_otsu=False,
    epochs_cls=10,
    epochs_seg=15
):
    """
    End-to-end weakly-supervised segmentation pipeline.

    This function executes the full segmentation pipeline:
        0. Train a classification model (ResNet18/50)
        1. Generate CAM-based pseudo masks from classifier
        2. Train segmentation model (UNet or DeepLabV3)
        3. Predict on test set
        4. Evaluate performance using IoU and Dice

    Args:
        model_name (str): Segmentation model to use ('unet' or 'deeplabv3').
        classifier_model (str): Classification backbone ('resnet18' or 'resnet50').
        threshold (float): Fixed threshold for CAM binarization (ignored if use_otsu=True).
        use_otsu (bool): If True, apply Otsu's method for per-image CAM thresholding.
        epochs_cls (int): Number of epochs for classifier training.
        epochs_seg (int): Number of epochs for segmentation model training.

    Notes:
        - All outputs are saved in `outputs/` (masks, predictions, checkpoints).
        - If model checkpoints already exist, they may be loaded to skip retraining.
        - Requires dataset and config paths defined in `scripts/config.py`.

    Example:
        run_pipeline(model_name='unet', classifier_model='resnet50', threshold=0.4, use_otsu=False)
    """
    
    print("Step 0: Train image classifier...")
    classifier_ckpt = f'{classifier_model}_cls_epoch_{epochs_cls}.pth'
    train_classifier(
        data_root=IMAGE_DIR,
        train_list=TRAIN_LIST_FILE,
        val_list=TEST_LIST_FILE,
        save_path=os.path.join(CHECKPOINT_DIR, classifier_ckpt),
        num_classes=37,
        batch_size=32,
        epochs=epochs_cls,
        lr=1e-4,
        weight_decay=5e-4,
        model=classifier_model,  # 'resnet18' or 'resnet50'
    )

    print("\nStep 1: Generate pseudo masks using CAM...")
    generate_pseudo_masks(
        threshold=otsu_threshold if use_otsu else threshold,
        model_name=classifier_model,
        checkpoint_name=classifier_ckpt
    )

    print("\nStep 2: Train segmentation model...")
    seg_ckpt = f"{model_name}_seg_{classifier_model}_cam_{threshold if not use_otsu else 'otsu'}_epoch_{epochs_seg}.pth"
    train_segmentor(
        model_name=model_name,
        epochs=epochs_seg,
        lr=1e-4,
        batch_size=32,
        save_path=os.path.join(CHECKPOINT_DIR, seg_ckpt)
    )

    print("\nStep 3: Predict on test set...")
    predict_and_save_masks(
        model_name=model_name,
        model_path=os.path.join(CHECKPOINT_DIR, seg_ckpt),
        test_list_file=TEST_LIST_FILE,
        threshold=0.5
    )

    print("\nStep 4: Evaluate segmentation results...")
    compute_metrics_for_split('test', PRED_DIR)



if __name__ == '__main__':
    run_pipeline(
        model_name='unet',        # or 'deeplabv3'
        classifier_model='resnet50',  # or 'resnet50'
        threshold=0.5,            # or use_otsu=True
        use_otsu=False,          # or True
        epochs_cls=10,
        epochs_seg=30
    )
