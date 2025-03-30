
# 📁 UCL COMP0197 Coursework Repository

This is the top-level repository for the UCL COMP0197 module coursework.

It contains multiple sub-projects related to different components of the course.

---

## 📂 Structure

```
comp0197-cw2-pt/
├── data/                            # ⬅️ Shared dataset (Oxford-IIIT Pet)
│   ├── images/
│   └── annotations/
│       ├── list.txt
│       └── trimaps/
│
├── outputs/                         # ⬅️ All experiment outputs (grouped by experiment name)
│   └── resnet_cam_unet/
│       ├── checkpoints/            # Classifier + segmentor model weights
│       ├── cams/                   # Optional CAM heatmaps
│       ├── pseudo_masks/          # Generated masks from CAM
│       └── preds/                 # UNet prediction masks
│
├── MRP/
│   └── resnet_cam_unet/            # Main model code
│       ├── scripts/
│       │   ├── train_classifier.py
│       │   ├── train_segmentor.py
│       │   ├── generate_pseudo_masks.py
│       │   ├── generate_cam.py
│       │   ├── predict_and_visualize.py
│       │   └── utils/
│       │       ├── config.py
│       │       ├── model.py, dataset.py, cam_utils.py, etc.
│       ├── README.md
│       └── requirements.txt
│
├── OEQ/                             # Optional open-ended work (currently placeholder)
│   └── (empty or in progress)
│
├── .gitignore                       # Ignores data/, outputs/, cache, etc.
└── README.md                        # Top-level README (project overview)


---

## 📌 Projects

### 🧠 MRP: Weakly-supervised Pet Segmentation

- Location: `MRP/resnet-cam-unet/`
- A complete weakly-supervised semantic segmentation pipeline using the Oxford-IIIT Pet Dataset.
- Uses ResNet18 for classification, CAM for localization, and UNet for segmentation.
- See the [MRP README](MRP/resnet-cam-unet/README.md) for full details.

### 🧪 OEQ: Open-Ended Question (Optional / In Progress)

- Location: `OEQ/`
- Currently empty or in preparation.

---

## ✍️ Author

UCL COMP0197 Coursework2 Group 8
