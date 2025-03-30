
# 🐾 Weakly-Supervised Semantic Segmentation of Pets (ResNet-CAM-UNet)

This project implements a weakly-supervised semantic segmentation pipeline for pet images, located at:

```
comp0197-cw2-pt/
└── MRP/
    └── resnet-cam-unet/    <- this is the main project directory
```

We use only image-level labels (breed categories) from the Oxford-IIIT Pet dataset to create pseudo pixel-level labels using Class Activation Maps (CAM), and train a UNet segmentation model with them.

---

## 📁 Project Structure

```
resnet-cam-unet/
├── data/
│   ├── images/                  # Raw pet images (.jpg)
│   └── annotations/            # list.txt + trimaps/ + xmls/
│
├── outputs/
│   ├── checkpoints/            # Trained models (.pth)
│   ├── cams/                   # Visual CAM heatmaps
│   ├── pseudo_masks/          # CAM → binary masks
│   └── preds/                 # UNet predictions
│
├── scripts/
│   ├── train_classifier.py        # Train ResNet18 classifier
│   ├── generate_cam.py            # Extract CAM for one image
│   ├── generate_pseudo_masks.py   # Generate binary pseudo labels
│   ├── train_segmentor.py         # Train UNet on pseudo masks
│   ├── predict_and_visualize.py  # Predict masks and save visualizations
│   └── utils/
│       ├── dataset.py
│       ├── model.py
│       ├── cam_utils.py
│       ├── mask_utils.py
│       └── metrics.py (optional)
│
├── README.md
├── .gitignore
└── requirements.txt (optional)
```

---

## 📦 Dataset Instructions

Download the Oxford-IIIT Pet Dataset manually:

- [images.tar.gz](https://www.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz)
- [annotations.tar.gz](https://www.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz)

Extract to `data/` directory:

```bash
tar -xvzf images.tar.gz -C data/
tar -xvzf annotations.tar.gz -C data/
```

---

## 🚀 Pipeline Summary

```bash
# Step 1: Train classification model
python scripts/train_classifier.py

# Step 2: Generate CAMs and pseudo masks
python scripts/generate_pseudo_masks.py

# Step 3: Train UNet with pseudo masks
python scripts/train_segmentor.py

# Step 4: Visualize UNet predictions
python scripts/predict_and_visualize.py
```

---

## 🧪 Notes

- Uses ResNet18 + CAM for localization
- Thresholded CAMs generate pseudo masks
- UNet trained using pseudo labels (no pixel GT used)
- Optional: evaluate using trimaps for mIoU

---

## ✍️ Author


UCL COMP0197 Coursework2 Group 8
