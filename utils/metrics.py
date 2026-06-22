import numpy as np
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score, confusion_matrix,roc_auc_score)


def compute_metrics(y_true, y_pred, y_prob=None, average="binary"):
    """
    Unified medical metrics function.

    Parameters:
    ----------
    y_true : list or np.array
    y_pred : list or np.array
    y_prob : probabilities (optional, for AUC)
    average : "binary" or "macro" (use macro for multi-class)

    Returns:
    -------
    dict with all metrics
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ==========================
    # Confusion Matrix
    # ==========================
    cm = confusion_matrix(y_true, y_pred)

    # For binary classification
    if cm.shape == (2, 2):

        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp + 1e-8)
        sensitivity = recall_score(y_true, y_pred, average=average)

    else:
        specificity = None
        sensitivity = recall_score(y_true, y_pred, average=average)

    # ==========================
    # Core metrics
    # ==========================
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = sensitivity
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    # ==========================
    # AUC (only if probabilities exist)
    # ==========================
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except:
            auc = None

    # ==========================
    # Return everything
    # ==========================
    return {
        "accuracy": acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": specificity,
        "f1_score": f1,
        "auc": auc,
        "confusion_matrix": cm
    }
    

