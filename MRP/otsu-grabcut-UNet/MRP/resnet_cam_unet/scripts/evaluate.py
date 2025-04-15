import os
from metrics import compute_metrics_for_split  
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import PRED_DIR


if __name__ == "__main__":

    miou_test = compute_metrics_for_split('test', PRED_DIR)
