potato-disease-esp32/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── kaggle_plantvillage/       # Raw Kaggle download (gitignored)
│   ├── processed/                  # Cached .pkl splits (gitignored)
│   └── raw/                        # Live ESP32-CAM captures
│
├── notebooks/                      # Experimentation only
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_experiments.ipynb
│   └── 03_error_analysis.ipynb
│
├── src/                            # ← THE REUSABLE CODE
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── convert_tflite.py
│
├── ml/                             # Saved models (gitignored)
│   ├── potato_mobilenet.h5
│   └── potato_model.tflite
│
├── firmware/                       # ESP32 code
│   └── potato_classifier/
│       ├── potato_classifier.ino
│       └── model.h
│
├── python/                         # Runtime scripts
│   ├── live_camera_collector.py
│   └── mqtt_logger.py
│
└── docs/                           # Reports, plots, diagrams
    ├── potato_samples.png
    ├── potato_training_history.png
    └── potato_confusion_matrix.png

    notebooks/01_load_dataset.ipynb
    └── imports from src/config.py         (paths, class names)
    └── imports from src/preprocessing.py  (load_dataset, split_dataset)

notebooks/02_train.ipynb
    └── imports from src/config.py         (hyperparameters)
    └── imports from src/augmentation.py   (augmentation pipeline)
    └── imports from src/preprocessing.py  (normalize)

python/live_camera_classifier.py
    └── imports from src/predict.py        (PotatoClassifier class)
    └── imports from src/config.py         (paths)

src/train.py (CLI)
    └── imports from src/preprocessing.py
    └── imports from src/augmentation.py
    └── imports from src/config.py

    I trained MobileNetV2 on two datasets:
- Potato (3 classes, 6.6:1 imbalance): 94.7% accuracy but macro F1 only 0.90, with poor precision (0.62) on the minority Healthy class.
- Tomato (4 classes, 2:1 imbalance): 94.7% accuracy with macro F1 of 0.9355, and no class below 0.85 F1.

This demonstrates that dataset balance is more important than dataset size for real-world deployment."

"I introduced metadata files alongside each trained model to prevent silent incompatibilities. This practice was motivated by an actual bug I encountered (loading a potato model for tomato evaluation), which produced 27% accuracy with no error message. The metadata-based assertion system catches such cases immediately.

Load Model from SD Card (Recommended)
Your ESP32-CAM has an SD card slot. Save the .tflite file to the SD card, and the firmware reads it into PSRAM at boot.

Pros: No flash size limit, model stays modifiable, no recompiling firmware to update model

Cons: Slightly slower boot (~1s to load 2MB)

Verdict: Best approach for your project

This is a classic sign of overfitting, where your model stops learning general patterns and starts memorizing the training data. At epoch 3, the model reaches its peak ability to generalize, but by epoch 9 and beyond, it becomes "too smart" for its own good, causing its performance on validation data to drop.
🛠️ Quick Ways to Fix It
Early Stopping: This is the most direct solution. You can set up a callback to stop training automatically if the validation accuracy stops improving for a set number of epochs (called patience). Since epoch 3 was your best, you would stop training around epoch 5 or 6 and save the weights from epoch 3.
Add Dropout: Insert Dropout layers (e.g., rate of 0.2 to 0.5) between your hidden layers. This randomly switches off neurons during training, forcing the network to learn more robust, redundant features.
Data Augmentation: Increase the size and variety of your training data. For images, apply random rotations or flips. For text or tabular data, introduce slight noise.
Weight Regularization: Add L1 (Lasso) or L2 (Ridge) regularization to your layers. This penalizes excessively large weights, keeping the model simpler.
Reduce Model Complexity: If your model has too many layers or parameters, it has the capacity to memorize noise. Try reducing the number of neurons or removing a layer.

# 🌿 Plant Disease Classification with Edge-AI Deployment

A computer vision system for classifying plant leaf diseases using transfer
learning (MobileNetV2), designed for deployment on ESP32-CAM for real-time
field inference.

Part of a two-project portfolio demonstrating breadth across **embedded systems**,
**machine learning**, and **edge AI** for intelligent agricultural systems.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Error Analysis](#error-analysis)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## 🎯 Overview

This project builds a **3-class potato** and **4-class tomato** leaf disease
classifier using transfer learning on MobileNetV2, with the goal of deploying
to an ESP32-CAM for on-device inference in agricultural settings.

**Two models were trained and compared:**

| Model | Classes | Test Accuracy | Macro F1 | Notes |
|-------|---------|---------------|----------|-------|
| Potato | 3 | 94.7% | 0.900 | Severe class imbalance (6.6:1) |
| Tomato | 4 | **94.5%** | **0.934** | Balanced dataset (2:1) |

The comparison demonstrates that **dataset balance matters more than task
difficulty** for edge-deployable plant disease classifiers.


