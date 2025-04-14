import os

# === Project structure ===
# This file lives in: comp0197-cw2-pt/MRP/supervised_baseline/scripts/

# Current directory: scripts/
CURRENT_DIR = os.path.dirname(__file__)

# Project root: resnet_cam_unet/
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# Top-level project: comp0197-cw2-pt/
TOP_LEVEL_DIR = os.path.abspath(os.path.join(PROJECT_DIR, '..', '..'))

# === Shared data directory ===
DATA_DIR = os.path.join(TOP_LEVEL_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotations')

TRAINVAL_LIST_FILE = os.path.join(ANNOTATION_DIR, 'trainval.txt')
TEST_LIST_FILE = os.path.join(ANNOTATION_DIR, 'test.txt')

# === Experiment identifier ===
# Change this name to organize outputs for different models
EXPERIMENT_NAME = 'supervised_baseline'  # <--- change this per experiment

# === Output paths scoped to the current experiment ===
OUTPUT_DIR = os.path.join(TOP_LEVEL_DIR, 'outputs', EXPERIMENT_NAME)

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')       # classifier + segmentor weights
PRED_DIR       = os.path.join(OUTPUT_DIR, 'preds')             # segmentation output masks
MASK_DIR       = os.path.join(OUTPUT_DIR, 'gt_masks')      # CAM → masks