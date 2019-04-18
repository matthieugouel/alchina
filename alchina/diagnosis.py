"""Diagnose the performances of an algorithm."""

import numpy as np


def r2_score(y_pred, y_true):
    """Coefficient of determination score."""
    ss_res = np.sum(np.square(y_true - y_pred), axis=0)
    ss_total = np.sum(np.square(y_true - np.mean(y_true, axis=0)), axis=0)

    if ss_total == 0:
        return 1

    return np.mean((1 - (ss_res / ss_total)))


def split_dataset(X, y, train: float = 0.7, shuffle: bool = True):
    """Split the dataset into train/test sets."""
    if train < 0 or train > 1:
        raise ValueError("train proportion must be between 0 and 1")

    if shuffle:
        indices = np.random.permutation(X.shape[0])
        X = X[indices[:, np.newaxis], np.arange(X.shape[1])]
        y = y[indices[:, np.newaxis], np.arange(y.shape[1])]

    index = round(train * X.shape[0])

    return X[:index], y[:index], X[index:], y[index:]
