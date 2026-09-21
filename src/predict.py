import json
import numpy as np
import tensorflow as tf
from . import config
from .preprocessing import load_image


class PotatoClassifier:
    def __init__(self, model_path=config.MODEL_H5):
        self.model = tf.keras.models.load_model(model_path)
        self.classes = config.CLASSES
        self.display = config.CLASS_DISPLAY

    def predict(self, image_path):
        """Returns dict: {class_name, display_name, confidence, all_probs}"""
        img = load_image(image_path, config.IMG_SIZE)
        if img is None:
            raise ValueError(f"Could not load {image_path}")

        x = np.expand_dims(img.astype("float32") / 255.0, axis=0)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))

        return {
            "class": self.classes[idx],
            "display_name": self.display[self.classes[idx]],
            "confidence": float(probs[idx]),
            "all_probs": {self.display[c]: float(p)
                          for c, p in zip(self.classes, probs)},
        }


if __name__ == "__main__":
    import sys
    clf = PotatoClassifier()
    result = clf.predict(sys.argv[1])
    print(f" {result['display_name']} "
          f"({result['confidence']*100:.1f}% confidence)")


class TomatoClassifier:
    def __init__(self, model_path=config.TOMATO_MODEL_H5,
                 meta_path=config.TOMATO_MODEL_META):
        print(f" Loading model: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.classes = self.meta["classes"]
        self.display = self.meta["class_display"]
        self.img_size = self.meta["img_size"]
        self.uses_preprocess_input = self.meta["uses_preprocess_input"]

        print(f" Model ready: {self.meta['num_classes']} classes "
              f"@ {self.img_size}px")

    def preprocess_array(self, img_rgb):
        """Resize to model's expected size. DO NOT normalize if the model
        uses preprocess_input internally."""
        import cv2
        img = cv2.resize(img_rgb, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_AREA)
        if self.uses_preprocess_input:
            return img.astype("float32")            # keep [0, 255]
        else:
            return img.astype("float32") / 255.0    # [0, 1]

    def predict_array(self, img_rgb):
        x = np.expand_dims(self.preprocess_array(img_rgb), axis=0)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return {
            "class": self.classes[idx],
            "display_name": self.display[self.classes[idx]],
            "confidence": float(probs[idx]),
            "all_probs": {self.display[c]: float(p)
                          for c, p in zip(self.classes, probs)},
        }