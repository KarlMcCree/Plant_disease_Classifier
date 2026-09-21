import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from . import config
from .preprocessing import normalize


def evaluate_model(model_path=config.MODEL_H5):
    with open(config.SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)

    X_test = normalize(splits["X_test"])
    y_test = splits["y_test"]
    display_names = [config.CLASS_DISPLAY[c] for c in config.CLASSES]

    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"🎯 Test accuracy: {acc:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("\n" + classification_report(y_test, y_pred,
                                        target_names=display_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=display_names, yticklabels=display_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{config.DOCS_DIR}/confusion_matrix.png", dpi=100)
    plt.show()

    return acc, y_pred


if __name__ == "__main__":
    evaluate_model()