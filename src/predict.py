import numpy as np
import tensorflow as tf

from .data import ACTIVITY_NAMES


def main():
    model = tf.keras.models.load_model("models/best_model.keras")
    params = np.load("models/scaler_parameters.npz")

    # Demonstration only: use the first official test example.
    from .data import load_data
    _, _, X_test, _, _ = load_data()

    sample = X_test[0:1]
    probabilities = model.predict(sample, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    print("Predicted activity:", ACTIVITY_NAMES[predicted_index + 1])
    print(f"Model confidence: {probabilities[predicted_index]:.3f}")
    print("\nClass probabilities:")
    for index, probability in enumerate(probabilities):
        print(f"{ACTIVITY_NAMES[index + 1]:22s} {probability:.3f}")


if __name__ == "__main__":
    main()
