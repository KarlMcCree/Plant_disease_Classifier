from tensorflow.keras.preprocessing.image import ImageDataGenerator
from . import config


def get_train_datagen():
    """Augmentation for training set only."""
    return ImageDataGenerator(
        rotation_range=config.AUG_ROTATION,
        width_shift_range=config.AUG_SHIFT,
        height_shift_range=config.AUG_SHIFT,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=config.AUG_ZOOM,
        brightness_range=config.AUG_BRIGHTNESS,
        fill_mode="nearest",
    )


def get_plain_datagen():
    """No augmentation (for val/test if needed)."""
    return ImageDataGenerator()