"""Metrics tools."""

import numpy as np

from typing import Optional

from .utils import check_dataset_consistency
from .utils import target_reshape, target_labels


def confusion_matrix(y_pred, y_true):
    """Confusion matrix."""
    y_pred = target_reshape(y_pred)
    y_true = target_reshape(y_true)

    label_number = len(target_labels(y_true))
    label_number = label_number if label_number >= 2 else 2

    cm = np.zeros((label_number, label_number), dtype=int)

    for i in range(y_true.size):
        cm[y_true[i]][y_pred[i]] += 1

    return cm


def accuracy_score(y_pred, y_true, count: bool = False):
    """Accuracy score.

    If `count` option is True, then it returns the sum of all accurate predictions.
    Else, it returns the mean of all accurate predictions.
    """
    if not check_dataset_consistency(y_pred, y_true):
        raise ValueError("input must have as many rows")

    cm = confusion_matrix(y_pred, y_true)

    accuracies = np.diag(cm).astype(float)

    if count:
        return np.sum(accuracies)
    return np.mean(accuracies)


def precision_score(y_pred, y_true, average: Optional[str] = None):
    """Precision score.

    In the case of multiclass labels, an average strategy can be set:

    **none**
    The score for each label is returned.

    **macro**
    The mean of each label score is returned.
    """
    if not check_dataset_consistency(y_pred, y_true):
        raise ValueError("input must have as many rows")

    cm = confusion_matrix(y_pred, y_true)

    numerator = np.diag(cm).astype(float)
    denominator = np.sum(cm, axis=0).astype(float)

    precisions = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )

    label_number = len(target_labels(y_true))
    if label_number <= 2:
        return np.sum(precisions)
    elif not average:
        return precisions
    elif average == "macro":
        return np.mean(precisions)
    else:
        raise ValueError(f"average {average} not supported")


def recall_score(y_pred, y_true, average: Optional[str] = None):
    """Recall score.

    In the case of multiclass labels, an average strategy can be set:

    **none**
    The score for each label is returned.

    **macro**
    The mean of each label score is returned.
    """
    if not check_dataset_consistency(y_pred, y_true):
        raise ValueError("input must have as many rows")

    cm = confusion_matrix(y_pred, y_true)

    numerator = np.diag(cm).astype(float)
    denominator = np.sum(cm, axis=1).astype(float)

    recalls = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )

    label_number = len(target_labels(y_true))
    if label_number <= 2:
        return np.sum(recalls)
    elif not average:
        return recalls
    elif average == "macro":
        return np.mean(recalls)
    else:
        raise ValueError(f"average {average} not supported")


def fbeta_score(y_pred, y_true, beta, average: Optional[str] = None):
    """F-beta score.

    In the case of multiclass labels, an average strategy can be set:

    **none**
    The score for each label is returned.

    **macro**
    The mean of each label score is returned.
    """
    if not check_dataset_consistency(y_pred, y_true):
        raise ValueError("input must have as many rows")

    precisions = precision_score(y_pred, y_true, average=None)
    recalls = recall_score(y_pred, y_true, average=None)

    numerator = (1 + beta ** 2) * precisions * recalls
    denominator = (beta ** 2) * precisions + recalls

    f_beta_scores = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )

    label_number = len(target_labels(y_true))
    if label_number <= 2:
        return np.sum(f_beta_scores)
    elif not average:
        return f_beta_scores
    elif average == "macro":
        return np.mean(f_beta_scores)
    else:
        raise ValueError(f"average {average} not supported")


def f1_score(y_pred, y_true, average: Optional[str] = None):
    """F1 score.

    In the case of multiclass labels, an average strategy can be set:

    **none**
    The score for each label is returned.

    **macro**
    The mean of each label score is returned.
    """
    return fbeta_score(y_pred, y_true, 1, average=average)


def r2_score(y_pred, y_true):
    """Coefficient of determination score."""
    if not check_dataset_consistency(y_pred, y_true):
        raise ValueError("input must have as many rows")

    ss_res = np.sum(np.square(y_true - y_pred), axis=0)
    ss_total = np.sum(np.square(y_true - np.mean(y_true, axis=0)), axis=0)

    if ss_total == 0:
        return 1

    return np.mean((1 - (ss_res / ss_total)))
