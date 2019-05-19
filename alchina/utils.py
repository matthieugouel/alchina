"""Utils."""

import numpy as np


from typing import Tuple


# --- Dataset ---


def check_dataset_consistency(X: np.ndarray, y: np.ndarray) -> bool:
    """Check the length consistency between the features set and the target set."""
    return X.shape[0] == y.shape[0]


def shuffle_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly shuffle the dataset."""
    indices = np.random.permutation(X.shape[0])
    X = X[indices[:, np.newaxis], np.arange(X.shape[1])]
    y = y[indices[:, np.newaxis], np.arange(y.shape[1])]
    return X, y


# --- Features ---


def features_reshape(X: np.ndarray) -> np.ndarray:
    """Reshape features matrix to be a 2-D array."""
    if X.ndim == 1:
        return X.reshape(1, -1)
    return X


# --- Target ---


def target_reshape(y: np.ndarray) -> np.ndarray:
    """Reshape the target to be a 1-D array."""
    return y.reshape(-1)


def target_labels(*args) -> np.ndarray:
    """Get the target labels."""
    return np.unique(np.vstack(args))


def target_label_count(*args) -> int:
    """Get the target label count."""
    label_count = len(target_labels(*args))
    return label_count if label_count >= 2 else 2


def target_label_to_index(y: np.ndarray, labels=None) -> np.ndarray:
    """Transform label to index."""
    if labels is None:
        labels = target_labels(y)

    translation = {label: index for index, label in enumerate(labels)}
    return np.vectorize(translation.get)(y)
