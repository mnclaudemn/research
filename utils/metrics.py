import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score
)


def compute_metrics(
    y_true,
    y_pred,
    y_prob=None,
    average="macro"
):
    """
    Robust medical metrics function (binary + multi-class safe)
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ==========================
    # Confusion Matrix
    # ==========================
    cm = confusion_matrix(y_true, y_pred)

    # ==========================
    # Core Metrics
    # ==========================
    acc = accuracy_score(y_true, y_pred)

    prec = precision_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    rec = recall_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    bal_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    # ==========================
    # AUC (safe handling)
    # ==========================
    auc = None

    if y_prob is not None:
        try:
            y_prob = np.array(y_prob)

            if len(np.unique(y_true)) > 2:
                auc = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="macro"
                )
            else:
                auc = roc_auc_score(
                    y_true,
                    y_prob[:, 1]
                )

        except Exception:
            auc = None

    # ==========================
    # Return clean dict
    # ==========================
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "confusion_matrix": cm
    }
