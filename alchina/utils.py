"""Utils."""

import numpy as np


from typing import Tuple


def check_dataset_consistency(X: np.ndarray, y: np.ndarray) -> bool:
    """Check the length consistency between the features set and the target set."""
    return X.shape[0] == y.shape[0]


def shuffle_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly shuffle the dataset."""
    indices = np.random.permutation(X.shape[0])
    X = X[indices[:, np.newaxis], np.arange(X.shape[1])]
    y = y[indices[:, np.newaxis], np.arange(y.shape[1])]
    return X, y


def target_reshape(y: np.ndarray) -> np.ndarray:
    """Reshape the target to be a 1-D array."""
    return y.reshape(-1)


def target_labels(y: np.ndarray) -> np.ndarray:
    """Get the target labels."""
    return np.unique(y)
