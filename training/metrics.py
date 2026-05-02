import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_auc_score,
)


def compute_metrics(gts, preds, probs):
    f1 = f1_score(gts, preds, average='binary')
    precision = precision_score(gts, preds, average='binary', zero_division=0)
    recall = recall_score(gts, preds, average='binary', zero_division=0)
    acc = accuracy_score(gts, preds)
    auc = roc_auc_score(gts, probs)

    cm = confusion_matrix(gts, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': acc,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
    }


def print_metrics(metrics):
    print(metrics['confusion_matrix'])
    print(f"Sensitivity: {metrics['sensitivity']:.6f}  Specificity: {metrics['specificity']:.6f}")
    print(
        f"Precision: {metrics['precision']:.6f}  "
        f"Recall: {metrics['recall']:.6f}  "
        f"F1: {metrics['f1']:.6f}  "
        f"Accuracy: {metrics['accuracy']:.6f}"
    )
    print(f"AUC: {metrics['auc']:.6f}")
