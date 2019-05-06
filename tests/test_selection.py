"""Selection tests."""

import numpy as np
import pytest

from alchina.selection import split_dataset


def test_split_dataset():
    """Test of `split_dataset` function."""
    X = np.arange(9).reshape(3, 3)
    y = np.arange(3).reshape(3, 1)

    X_train, y_train, X_test, y_test = split_dataset(X, y)

    assert X_train.shape == (2, 3)
    assert y_train.shape == (2, 1)

    assert X_test.shape == (1, 3)
    assert y_test.shape == (1, 1)


def test_split_dataset_no_shuffle():
    """Test of `split_dataset` function with no shuffle."""
    X = np.arange(9).reshape(3, 3)
    y = np.arange(3).reshape(3, 1)

    X_train, y_train, X_test, y_test = split_dataset(X, y, shuffle=False)

    assert X_train.shape == (2, 3)
    assert y_train.shape == (2, 1)

    assert X_test.shape == (1, 3)
    assert y_test.shape == (1, 1)


def test_split_dataset_invalid_input():
    """Test of `split_dataset` function with invalid input."""
    X = np.arange(9).reshape(3, 3)
    y = np.arange(3).reshape(3, 1)

    with pytest.raises(ValueError):
        split_dataset(X, y, train=2)
