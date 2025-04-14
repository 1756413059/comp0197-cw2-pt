import os

# === Project structure ===
# This file lives in: comp0197-cw2-pt/OEQ/ViT/scripts/

CURRENT_DIR = os.path.dirname(__file__)

PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

TOP_LEVEL_DIR = os.path.abspath(os.path.join(PROJECT_DIR, '..', '..'))

DATA_DIR = os.path.join(TOP_LEVEL_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotations')
LIST_FILE = os.path.join(ANNOTATION_DIR, 'list.txt')
TRAIN_LIST_FILE = os.path.join(ANNOTATION_DIR, 'trainval.txt')
TEST_LIST_FILE = os.path.join(ANNOTATION_DIR, 'test.txt')
GT_DIR = os.path.join(ANNOTATION_DIR, 'trimaps')

EXPERIMENT_NAME = 'resnet_cam_unet'

OUTPUT_DIR = os.path.join(TOP_LEVEL_DIR, 'outputs', EXPERIMENT_NAME)

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')  
CAM_DIR        = os.path.join(OUTPUT_DIR, 'cams')   
MASK_DIR       = os.path.join(OUTPUT_DIR, 'pseudo_masks') 
PRED_DIR       = os.path.join(OUTPUT_DIR, 'preds') 
