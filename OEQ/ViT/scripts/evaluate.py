import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import PRED_DIR
from utils.metrics import compute_metrics_for_split

if __name__ == "__main__":
    iou_mean, dice_mean = compute_metrics_for_split('test', PRED_DIR)
