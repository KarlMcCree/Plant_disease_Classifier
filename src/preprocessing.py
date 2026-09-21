import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from . import config


def load_image(path, img_size=config.IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_dataset(use_cache=True):
 
    if use_cache and os.path.exists(config.CACHE_FILE):
        print(f" Loading cache: {config.CACHE_FILE}")
        with open(config.CACHE_FILE, "rb") as f:
            return pickle.load(f)

    X, y = [], []
    class_map = config.CLASS_IDS

    for class_name in config.CLASSES:
        class_path = os.path.join(config.KAGGLE_DIR, class_name)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"Missing: {class_path}")

        files = [f for f in os.listdir(class_path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files = files[:config.MAX_IMAGES_PER_CLASS]

        print(f" {class_name}: {len(files)} images")
        for fname in tqdm(files, desc=f"  {class_name[:25]}"):
            img = load_image(os.path.join(class_path, fname))
            if img is not None:
                X.append(img)
                y.append(class_map[class_name])

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int32)

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    with open(config.CACHE_FILE, "wb") as f:
        pickle.dump({"X": X, "y": y, "classes": config.CLASSES}, f)

    return {"X": X, "y": y, "classes": config.CLASSES}


def split_dataset(X, y, save=True):
    """Stratified train/val/test split (70/15/15)."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config.VAL_SPLIT + config.TEST_SPLIT,
        random_state=config.RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=config.RANDOM_SEED, stratify=y_temp
    )

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "classes": config.CLASSES,
        "class_display": config.CLASS_DISPLAY,
    }

    if save:
        with open(config.SPLITS_FILE, "wb") as f:
            pickle.dump(splits, f)
        print(f" Saved splits → {config.SPLITS_FILE}")

    return splits


def normalize(X):
    """uint8 [0,255] → float32 [0,1]."""
    return X.astype("float32") / 255.0