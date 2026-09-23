#  Plant Disease Classification with Edge-AI Deployment

A computer vision system for classifying plant leaf diseases using transfer
learning (MobileNetV2), designed for deployment on ESP32-CAM for real-time
field inference.

Part of a two-project portfolio demonstrating breadth across **embedded systems**,
**machine learning**, and **edge AI** for intelligent agricultural systems.


---


##  Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Error Analysis](#error-analysis)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)


---


##  Overview

This project builds a **3-class potato** and **4-class tomato** leaf disease
classifier using transfer learning on MobileNetV2, with the goal of deploying
to an ESP32-CAM for on-device inference in agricultural settings.

**Two models were trained and compared:**

| Model | Classes | Test Accuracy | Macro F1 | Notes |
|-------|---------|---------------|----------|-------|
| Potato | 3 | 93.81% | 0.890 | Severe class imbalance (6.6:1) |
| Tomato | 4 | **95.28%** | **0.942** | Balanced dataset (2:1) |

The comparison demonstrates that **dataset balance matters more than task difficulty**
for edge-deployable plant disease classifiers.


**Key achievement:** No class falls below 0.880 F1. The model is production-safe
because the **Healthy class achieves 0.996 recall** — meaning the model almost
never misses a healthy plant 

### Potato Classifier (Research Comparison)
Test Accuracy: 94.43%
Macro F1: 0.890

precision recall f1-score support
Early Blight 0.99 0.97 0.98 150
Late Blight 0.96 0.91 0.93 150
Healthy 0.63 0.96 0.76 23 ← collapse

**Key finding:** The Healthy class precision collapsed to 0.63 due to severe
class imbalance (only 152 healthy images vs 1,000 each for the disease classes).
This became the primary motivation for switching to **tomato**.


---


##  System Architecture
┌──────────────┐ ┌───────────────┐ ┌──────────────┐ ┌────────────┐
│ Dataset │───▶│ Preprocessing│───▶│ Training │───▶│ TFLite │
│ (Kaggle) │ │ + Augment. │ │ (MobileNet) │ │ Export │
└──────────────┘ └───────────────┘ └──────────────┘ └─────┬──────┘
│
▼
┌──────────────┐ ┌───────────────┐ ┌──────────────┐ ┌────────────┐
│ Dashboard │◀───│ MQTT / Serial│◀───│ Edge Infer. │◀───│ ESP32-CAM │
│ (Streamlit) │ │ (Metadata) │ │ (INT8) │ │ Firmware │
└──────────────┘ └───────────────┘ └──────────────┘ └────────────┘


---


##  Dataset

**Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
(Hughes & Salathé, 2015) — 54,306 images, 38 classes.

### Classes Used

**Tomato (4 classes):**
- `Tomato_Early_blight` — 1,000 images
- `Tomato_Late_blight` — 1,909 images
- `Tomato_Bacterial_spot` — 2,127 images
- `Tomato_healthy` — 1,591 images

**Potato (3 classes):**
- `Potato___Early_blight` — 1,000 images
- `Potato___Late_blight` — 1,000 images
- `Potato___healthy` — **152 images**  (documented imbalance)

### Preprocessing

- Resized to **96×96 pixels** (ESP32-CAM constraint)
- Stratified 70/15/15 train/val/test split
- Training augmentation: flip, rotation, translation, zoom, brightness, contrast


---


##  Methodology

### Model

- **Backbone:** MobileNetV2 (pretrained on ImageNet, frozen)
- **Head:** GlobalAvgPool → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.4) → Softmax
- **Total params:** ~2.4M (~230 KB after INT8 quantization)

### Training

- **Optimizer:** Adam (initial LR: 1e-3)
- **Loss:** Sparse categorical crossentropy
- **Batch size:** 32
- **Callbacks:** EarlyStopping (patience=6), ReduceLROnPlateau, ModelCheckpoint
- **Class weights:** Balanced (computed from training distribution)

### Why MobileNetV2?

- Designed for mobile/embedded devices
- 4.2M params, fits in ESP32-CAM PSRAM
- Well-supported by TensorFlow Lite for Microcontrollers
- Pretrained on ImageNet → transfer learning works with small datasets


---


## Results

### Training Curves

![Training History](docs/tomato_training_history.png)

Training stopped at **epoch 5** (EarlyStopping) after validation accuracy
plateaued at 0.9396. The best model was restored automatically.

### Confusion Matrix

![Confusion Matrix](docs/tomato_error_confusion.png)

### ROC Curves

![ROC Curves](docs/tomato_roc_curves_v2.png)

All classes achieve AUC > 0.98, confirming strong discriminative ability.


---


## Project Structure


```mermaid
flowchart TB

    %% =========================
    %% PROJECT ROOT
    %% =========================

    ROOT[" Plant Disease Classifier"]

    %% =========================
    %% MAIN COMPONENTS
    %% =========================

    ROOT --> DATA[" data<br/>Datasets & Processed Data"]
    ROOT --> SRC[" src<br/>Machine Learning Pipeline"]
    ROOT --> NB[" notebooks<br/>Experiments & Analysis"]
    ROOT --> PY[" python<br/>Runtime & Integration"]
    ROOT --> ML[" ml<br/>Model Artifacts"]
    ROOT --> FW[" firmware<br/>Embedded Deployment"]
    ROOT --> DOCS[" docs<br/>Results & Documentation"]

    %% =========================
    %% DATA
    %% =========================

    DATA --> RAW["Raw Dataset"]
    DATA --> PROC["Processed Data"]
    DATA --> SPLITS["Train / Validation / Test Splits"]

    %% =========================
    %% MACHINE LEARNING PIPELINE
    %% =========================

    SRC --> PRE["preprocessing.py<br/>Data Preparation"]
    SRC --> AUG["augmentation.py<br/>Data Augmentation"]
    SRC --> TRAIN["train.py<br/>Model Training"]
    SRC --> EVAL["evaluate.py<br/>Evaluation & Metrics"]
    SRC --> PRED["predict.py<br/>Inference"]
    SRC --> CONVERT["convert_tflite.py<br/>Edge Conversion"]
    SRC --> CONFIG["config.py<br/>Project Configuration"]

    %% =========================
    %% EXPERIMENTS
    %% =========================

    NB --> POTATO[" Potato<br/>Experiments"]
    NB --> TOMATO[" Tomato<br/>Experiments"]

    POTATO --> POTATO_LOAD["Dataset Loading"]
    POTATO --> POTATO_TRAIN["Model Training"]
    POTATO --> POTATO_ERROR["Error Analysis"]

    TOMATO --> TOMATO_LOAD["Dataset Loading"]
    TOMATO --> TOMATO_TRAIN["Model Training"]
    TOMATO --> TOMATO_ERROR["Error Analysis"]

    %% =========================
    %% MODEL ARTIFACTS
    %% =========================

    ML --> KERAS["MobileNetV2<br/>Keras Models"]
    ML --> TFLITE["TensorFlow Lite<br/>Models"]
    ML --> META["Model Metadata<br/>JSON"]

    %% =========================
    %% PYTHON INTEGRATION
    %% =========================

    PY --> CAMERA["live_camera_classifier.py<br/>Real-Time Inference"]
    PY --> MQTT["mqtt_logger.py<br/>MQTT Prediction Logging"]

    %% =========================
    %% EMBEDDED DEPLOYMENT
    %% =========================

    FW --> ESP32["ESP32<br/>Embedded Classifier"]
    FW --> HEADER["model.h<br/>Embedded Model"]

    %% =========================
    %% DOCUMENTATION
    %% =========================

    DOCS --> HISTORY["Training History"]
    DOCS --> CONF["Confusion Matrix"]
    DOCS --> ROC["ROC Curves"]
    DOCS --> ERRORS["Misclassified Images"]

    %% =========================
    %% FLOW CONNECTIONS
    %% =========================

    DATA --> SRC
    SRC --> ML
    ML --> PY
    ML --> FW
    SRC --> DOCS
```

###  Machine Learning Workflow

```mermaid
flowchart LR

    A[" PlantVillage Dataset"]
    B[" Preprocessing"]
    C[" Dataset Splitting"]
    D[" Data Augmentation"]
    E[" MobileNetV2"]
    F[" Evaluation"]
    G[" Error Analysis"]
    H[" Edge Deployment"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    F --> H

    H --> I[" TensorFlow Lite"]
    I --> J[" ESP32"]
```

###  Repository Organization

| Component       | Purpose                                  |
| --------------- | ---------------------------------------- |
|  `src/`       | Core machine-learning pipeline           |
|  `notebooks/` | Experiments, training and error analysis |
|  `data/`      | Dataset and processed data               |
|  `ml/`        | Trained models and deployment artifacts  |
|  `python/`    | Real-time inference and MQTT integration |
|  `firmware/`  | ESP32 embedded deployment                |
|  `docs/`      | Training results and visual analysis     |

```

**One important thing:** I would use **both diagrams**, but give them different jobs:

- **Project Architecture** → shows how your repository is organized.
- **Machine Learning Workflow** → shows the actual engineering pipeline.

That is much stronger for your GitHub portfolio because someone reviewing it can understand **both the software architecture and the ML workflow without digging through every folder.**
```


---


## Error Analysis

I introduced metadata files alongside each trained model to prevent silent incompatibilities. This practice was motivated by an actual bug I encountered (loading a potato model for tomato evaluation), which produced 27% accuracy with no error message. The metadata-based assertion system catches such cases immediately.

- **4.7% overall error rate** (47 / 995 misclassified)
- **Most common error:** Early Blight → Late Blight (visually similar fungal lesions)
- **Error concentration:** Errors cluster in low-confidence (<70%) predictions,
  enabling a confidence-thresholded rejection strategy in production

*Finding 1*: Confusion Between Early and Late Blight
The most common misclassification is Early Blight → Late Blight
(visually similar fungal lesions with brown/black spots). At 96×96
resolution, the subtle visual differences are partially lost.

Recommendation: Higher-resolution input (192×192) or a two-stage
classifier could reduce this error.

*Finding 2*: Healthy Class(Tomato) is Highly Reliable
Recall = 1.000 for the Healthy class means false positives (healthy
misclassified as diseased) are eliminated. This is the correct
behaviour for production: we prefer to miss a diseased leaf than
falsely alarm on a healthy one.

*Finding 3*: Class Imbalance(Potato) Kills Minority-Class Precision
The potato comparison revealed that with a 6.6:1 imbalance, class weights
are not enough — the minority class precision collapsed to 0.62.
Switching to tomato (2:1 imbalance) fixed this without any other changes.


---


## Limitations

- Lab-captured images. PlantVillage uses uniform gray backgrounds and
studio lighting. Real-world performance on field images (soil, hands,
natural lighting) will likely be lower due to domain shift.

- Limited classes. This is a proof-of-concept covering 3-4 classes.
Production systems need 20+ classes across multiple crops.

- No diseased-leaf validation from ESP32-CAM yet. The model is trained
and evaluated on PlantVillage data; ESP32-CAM field testing is a
planned next step.

- Resolution trade-off. 96×96 is chosen for ESP32 memory constraints.
This limits the model's ability to distinguish visually similar diseases.


---


## Future Work


Domain adaptation. Fine-tune on 200-300 field-captured images from
the ESP32-CAM to close the lab-vs-field gap.

Smaller backbone. MobileNetV3-Small @ 64×64 to fit model in ESP32
flash (currently requires SD card loading).

Two-stage cascade. Binary Healthy-vs-Diseased first, then
disease-type classifier — improves both accuracy and explainability.

Grad-CAM visualization. Show which leaf regions drive the model's
decision, useful for agricultural trust and diagnosis.

Extend classes. Add Tomato_Leaf_Mold, Tomato_Septoria_leaf_spot,
Tomato__Tomato_mosaic_virus to cover more real-world cases.

MQTT dashboard. Real-time field monitoring with alerts on disease
detection.


---


## References

Hughes, D. P., & Salathé, M. (2015). An open access repository of images
on plant health. arXiv preprint arXiv:1511.08060.

Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear
Bottlenecks. CVPR.

Warden, P., & Situnayake, D. (2019). TinyML: Machine Learning with
TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers. O'Reilly.