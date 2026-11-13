from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .data import ACTIVITY_NAMES


def evaluate_model(model, X_test, y_test):
    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
    }

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    labels = list(range(6))
    names = [ACTIVITY_NAMES[i + 1] for i in labels]
    report = classification_report(
        y_test, predictions, labels=labels, target_names=names, digits=4
    )
    (Path("reports") / "classification_report.txt").write_text(report)

    with (Path("reports") / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(cm, cmap="Blues")
    ax.set(
        xticks=labels, yticks=labels,
        xticklabels=names, yticklabels=names,
        xlabel="Predicted activity", ylabel="True activity",
        title="UCI HAR — Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig("reports/figures/confusion_matrix.png", dpi=160)
    plt.close(fig)

    return metrics, report
