"""Metrics tools."""

import numpy as np

from .utils import check_dataset_consistency


def r2_score(y_pred, y_true):
    """Coefficient of determination score."""
    if not check_dataset_consistency(y_pred, y_true):
        raise ValueError("input must have as many rows")

    ss_res = np.sum(np.square(y_true - y_pred), axis=0)
    ss_total = np.sum(np.square(y_true - np.mean(y_true, axis=0)), axis=0)

    if ss_total == 0:
        return 1

    return np.mean((1 - (ss_res / ss_total)))
