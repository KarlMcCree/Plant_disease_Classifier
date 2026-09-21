import os

# === PATHS ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR        = os.path.join(PROJECT_ROOT, "data")
KAGGLE_DIR      = os.path.join(DATA_DIR, "kaggle_plantvillage", "PlantVillage")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
ML_DIR          = os.path.join(PROJECT_ROOT, "ml")
DOCS_DIR        = os.path.join(PROJECT_ROOT, "docs")


TOMATO_CLASSES = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_healthy",
]

TOMATO_CLASS_DISPLAY = {
    "Tomato_Early_blight":     "Early Blight",
    "Tomato_Late_blight":      "Late Blight",
    "Tomato_Bacterial_spot":   "Bacterial Spot",
    "Tomato_healthy":          "Healthy",
}

TOMATO_CACHE_FILE   = os.path.join(PROCESSED_DIR, "tomato_dataset_cache.pkl")
TOMATO_SPLITS_FILE  = os.path.join(PROCESSED_DIR, "tomato_splits.pkl")
TOMATO_MODEL        = os.path.join(ML_DIR, "tomato_mobilenet.keras")
TOMATO_MODEL_BEST   = os.path.join(ML_DIR, "tomato_mobilenet_v2.keras")

TOMATO_MODEL_TFLITE     = os.path.join(ML_DIR, "tomato_model.tflite")
TOMATO_MODEL_META       = os.path.join(ML_DIR, "tomato_mobilenet_meta.json")


CACHE_FILE      = os.path.join(PROCESSED_DIR, "potato_dataset_cache.pkl")
SPLITS_FILE     = os.path.join(PROCESSED_DIR, "potato_splits.pkl")
MODEL_H5        = os.path.join(ML_DIR, "potato_mobilenet.h5")
MODEL_TFLITE    = os.path.join(ML_DIR, "potato_model.tflite")

# === DATASET ===
CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]
CLASS_DISPLAY = {
    "Potato___Early_blight": "Early Blight",
    "Potato___Late_blight":  "Late Blight",
    "Potato___healthy":      "Healthy",
}
# ESP32 firmware will use these integer IDs
CLASS_IDS = {cls: idx for idx, cls in enumerate(CLASSES)}

# === IMAGE ===
IMG_SIZE = 96          # 96x96 for ESP32-CAM
IMG_CHANNELS = 3

# === TRAINING ===
BATCH_SIZE  = 32
EPOCHS      = 20
LEARNING_RATE = 1e-3
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
RANDOM_SEED = 42
MAX_IMAGES_PER_CLASS = 1000

# === AUGMENTATION ===
AUG_ROTATION     = 25
AUG_SHIFT        = 0.15
AUG_ZOOM         = 0.15
AUG_BRIGHTNESS   = (0.8, 1.2)