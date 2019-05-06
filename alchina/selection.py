"""Model selection."""

from .utils import shuffle_dataset


def split_dataset(X, y, train: float = 0.7, shuffle: bool = True):
    """Split the dataset into train/test sets."""
    if train < 0 or train > 1:
        raise ValueError("train proportion must be between 0 and 1")

    if shuffle:
        X, y = shuffle_dataset(X, y)

    index = round(train * X.shape[0])

    return X[:index], y[:index], X[index:], y[index:]
