import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from . import config
from .preprocessing import load_dataset, split_dataset, normalize
from .augmentation import get_train_datagen


def build_model(num_classes):
    """MobileNetV2 + custom classification head."""
    base = MobileNetV2(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_class_weights(y_train):
    """Handle imbalanced classes (healthy has fewer images)."""
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    return {i: w for i, w in enumerate(weights)}


def train():
    # Load or create splits
    if not os.path.exists(config.SPLITS_FILE):
        data = load_dataset()
        splits = split_dataset(data["X"], data["y"])
    else:
        with open(config.SPLITS_FILE, "rb") as f:
            splits = pickle.load(f)

    X_train = normalize(splits["X_train"])
    y_train = splits["y_train"]
    X_val   = normalize(splits["X_val"])
    y_val   = splits["y_val"]

    model = build_model(num_classes=len(config.CLASSES))
    model.summary()

    class_weights = get_class_weights(y_train)
    datagen = get_train_datagen()
    datagen.fit(X_train)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE),
        steps_per_epoch=len(X_train) // config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True,
                          monitor="val_accuracy"),
            ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss"),
            ModelCheckpoint(config.MODEL_H5, save_best_only=True,
                            monitor="val_accuracy"),
        ],
    )
    print(f" Best val accuracy: {max(history.history['val_accuracy']):.4f}")
    return model, history


if __name__ == "__main__":
    import os
    train()