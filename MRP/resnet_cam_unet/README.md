# 🐾 COMP0197 Coursework 2: Weakly-Supervised Semantic Segmentation

This project implements a full pipeline for weakly-supervised semantic segmentation using Class Activation Maps (CAM) and pseudo mask generation, targeting the Oxford-IIIT Pet dataset.

---

## Project Structure Overview

```
COMP0197-CW2-PT/
├── data/                      # Raw dataset directory
├── MRP/
│   ├── cam_comparison/       # CAM ablation experiments (optional)
│   └── resnet_cam_unet/      # Main pipeline implementation
│       ├── scripts/
│       │   ├── utils/
│       │   │   ├── cam_utils.py         # CAM generation & processing
│       │   │   ├── dataset.py           # Dataset classes
│       │   │   ├── mask_utils.py        # Mask binarization, thresholding
│       │   │   ├── metrics.py           # Evaluation: IoU, Dice
│       │   │   └── model.py             # Classifier & segmentor models
│       │   ├── config.py                # Path configuration
│       │   ├── evaluate.py              # Evaluation script
│       │   ├── generate_cam.py          # CAM visualization (optional)
│       │   ├── generate_pseudo_masks.py # CAM → Mask
│       │   ├── predict_and_visualize.py # Predict and save segmentations
│       │   ├── run_pipeline.py          # 🔁 Run full pipeline (one-click)
│       │   ├── train_classifier.py      # ResNet classifier training
│       │   └── train_segmentor.py       # UNet / DeepLab training
├── supervised_baseline/      # Fully-supervised baseline code
├── outputs/                  # All predictions, pseudo masks, checkpoints
├── OEQ/                      # (Optional) Coursework reflection & reports
└── README.md                 # You're reading this!
```

---

## Pipeline Stages

1. **Train a ResNet classifier**  
2. **Generate CAM heatmaps + pseudo masks**
3. **Train segmentor using pseudo masks**
4. **Predict & save segmentation masks**
5. **Evaluate performance (Dice / IoU)**

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch torchvision scikit-learn
```

### 2. Run full pipeline
```bash
python MRP/resnet_cam_unet/scripts/run_pipeline.py
```

---

## Evaluation Metrics

- **Mean IoU** (Foreground vs Background, ignoring boundary)
- **Dice Score** (F1 for binary segmentation)

---

## Example Config (from `run_pipeline.py`)
```python
run_pipeline(
    model_name='unet',
    classifier_model='resnet50',
    threshold=0.5,
    use_otsu=False,
    epochs_cls=10,
    epochs_seg=15
)
```

---

## Dataset: Oxford-IIIT Pet

Each sample includes:
- `image_name.jpg`: RGB image
- `trimap`: 3-class pixel-level mask: {1=foreground, 2=background, 3=boundary}
- 37 pet breeds as classification target

---

## Author

- group 8 
- COMP0197: Applied Deep Learning [T2] 24/25

---