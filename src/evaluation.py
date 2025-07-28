import os
from typing import Sequence
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

logger = logging.getLogger(__name__)

def confusion_matrix_(y_true, y_pred, labels, path: str) -> None:
    """
    Compute and display the confusion matrix for true vs. predicted labels.
    """
    assert len(y_true) == len(y_pred)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    logger.info("Confusion Matrix:\n%s", cm)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {path}")

def classification_report_(y_true: Sequence[int], y_pred: Sequence[int], path: str, target_names=["negative", "positive"]) -> None:
    """
    Generate and print a classification report with precision, recall, and F1-score.
    """
    assert len(y_true) == len(y_pred)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report_text = classification_report(y_true, y_pred, target_names=target_names)
    logger.info("Classification Report:\n%s", report_text)
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    with open(path, "w") as f:
        json.dump(report_dict, f, indent=4)
    logger.info(f"Saved classification report to {path}")

def metrics_summary(y_true, y_pred, path: str) -> None:
    """
    Summarize evaluation metrics across all models into a DataFrame.
    """
    assert len(y_true) == len(y_pred)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
    for metric, value in summary.items():
        logger.info(f"{metric.capitalize()}: {value:.3f}")
    with open(path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved metrics summary to {path}")

