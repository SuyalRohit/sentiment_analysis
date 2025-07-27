# src/evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def save_confusion_matrix(y_true, y_pred, labels, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_classification_report(y_true, y_pred, path):
    report = classification_report(y_true, y_pred, target_names=["negative", "positive"], output_dict=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=4)

def save_metrics_summary(y_true, y_pred, path):
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=4)