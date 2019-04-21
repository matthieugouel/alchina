"""Utils."""

import numpy as np


def shuffle_dataset(X, y):
    """Randomly shuffle the dataset."""
    indices = np.random.permutation(X.shape[0])
    X = X[indices[:, np.newaxis], np.arange(X.shape[1])]
    y = y[indices[:, np.newaxis], np.arange(y.shape[1])]
    return X, y
