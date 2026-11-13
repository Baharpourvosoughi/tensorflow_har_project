from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .data import load_data
from .model import build_model
from .evaluate import evaluate_model


def main():
    tf.keras.utils.set_random_seed(42)

    X_train_full, y_train_full, X_test, y_test, scaler = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )

    model = build_model(input_dim=X_train.shape[1], num_classes=6)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    Path("models").mkdir(exist_ok=True)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    model = tf.keras.models.load_model("models/best_model.keras")
    metrics, report = evaluate_model(model, X_test, y_test)

    # Save the training-only scaler values so new samples can be transformed consistently.
    np.savez(
        "models/scaler_parameters.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
    )
    with open("reports/training_history.json", "w") as f:
        json.dump(
            {
                "loss": [float(v) for v in history.history["loss"]],
                "val_loss": [float(v) for v in history.history["val_loss"]],
                "accuracy": [float(v) for v in history.history["accuracy"]],
                "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
            },
            f,
            indent=2,
        )

    print("\nTest-set metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nDetailed classification report:\n")
    print(report)
    print("\nSaved model to models/best_model.keras")
    print("Saved metrics and plots under reports/")


if __name__ == "__main__":
    main()
