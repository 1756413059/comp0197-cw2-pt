import os

# === Current script directory: MRP/resnet-cam-unet/scripts/
CURRENT_DIR = os.path.dirname(__file__)

# === Project root: MRP/resnet-cam-unet/
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# === Top-level repo: comp0197-cw2-pt/
TOP_LEVEL_DIR = os.path.abspath(os.path.join(PROJECT_DIR, '..', '..'))

# === Top-level data and output directories
DATA_DIR = os.path.join(TOP_LEVEL_DIR, 'data')
OUTPUT_DIR = os.path.join(TOP_LEVEL_DIR, 'outputs')

# === Data subdirectories
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotations')
LIST_FILE = os.path.join(ANNOTATION_DIR, 'list.txt')

# === Output subdirectories
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
CAM_DIR = os.path.join(OUTPUT_DIR, 'cams')
MASK_DIR = os.path.join(OUTPUT_DIR, 'pseudo_masks')
PRED_DIR = os.path.join(OUTPUT_DIR, 'preds')
