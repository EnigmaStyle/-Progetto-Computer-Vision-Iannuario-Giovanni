from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from src.utils import CLASS_NAMES


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
                                        target_names=CLASS_NAMES, zero_division=0),
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_per_class_f1(y_true: List[int], y_pred: List[int], save_path: str = None) -> plt.Figure:
    scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(CLASS_NAMES, scores, color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-class F1 Score")
    for i, s in enumerate(scores):
        ax.text(i, s + 0.02, f"{s:.2f}", ha="center", fontsize=9)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = None,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train")
    ax2.plot(epochs, val_accs, label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
