# 🐾 Weakly-Supervised Semantic Segmentation of Pets

This project implements a weakly-supervised semantic segmentation pipeline using the Oxford-IIIT Pet dataset.  
Unlike fully-supervised methods that require pixel-level annotations, we only use **image-level labels** (breed categories) as weak supervision. The pipeline leverages **Class Activation Maps (CAM)** to generate pseudo pixel-level masks, which are then used to train a semantic segmentation model.

---

## 📌 Project Overview

### 🧠 Objective

To segment pets from images using **weak supervision only**, with the following pipeline:

1. **Image-level labels only** (no pixel masks during training)
2. Train an image classification model (ResNet18)
3. Extract **Class Activation Maps (CAMs)** to localize pet regions
4. Convert CAMs to binary masks as **pseudo ground truth**
5. Train a segmentation model (e.g., UNet) using pseudo labels
6. Evaluate using ground-truth pixel masks

---

## 🗂️ Project Structure

```
weakly_segmentation_project/
├── data/                   # [Ignored] Pet dataset (see below)
│   ├── images/             # Raw images (.jpg)
│   └── annotations/        # Includes list.txt and trimaps/
├── outputs/                # CAMs, pseudo masks, model checkpoints
├── scripts/                # Python scripts (training, CAM, segmentation)
│   └── utils/              # Dataset loader, models, CAM utilities, etc.
├── README.md
└── .gitignore
```

---

## 📦 Dataset Instructions

We use the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which contains:

- 7,390 images of 37 cat and dog breeds
- Image-level labels (breed)
- Pixel-level segmentation masks (for evaluation only)

> ⚠️ Due to file size, the dataset is **not included in this repository**.

### 🧾 Download links

- [images.tar.gz](https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz)
- [annotations.tar.gz](https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz)

### 📂 After extracting, the structure should be:

```
data/
├── images/
└── annotations/
    ├── list.txt
    ├── trimaps/
    └── xmls/
```

Use the following commands to extract:

```bash
tar -xvzf images.tar.gz -C data/
tar -xvzf annotations.tar.gz -C data/
```

---

## 🚀 Getting Started

```bash
# Step 1: Train image classification model (ResNet18)
python scripts/train_classifier.py

# Step 2: Generate CAMs and pseudo masks
python scripts/generate_pseudo_masks.py

# Step 3: Train segmentation model using pseudo masks
python scripts/train_segmentor.py

# Step 4: Evaluate against ground-truth segmentation
python scripts/evaluate.py
```

---

## 📊 Evaluation Metrics

The trained segmentation model is evaluated using:

- **Pixel Accuracy**
- **Mean Intersection over Union (mIoU)**

> Ground-truth masks are located in `annotations/trimaps/`.

---

## 🛠️ Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- torchvision
- numpy, matplotlib, opencv-python
- tqdm

You can install dependencies with:

```bash
pip install -r requirements.txt
```

---

## ✍️ Author

*Your Name*  
COMP0197 Coursework Project (UCL)

---

## 📄 License

This project is for academic use only. No commercial redistribution of the Oxford Pet dataset.
