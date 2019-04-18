"""Diagnosis tests."""

import numpy as np
import pytest

from alchina.diagnosis import r2_score
from alchina.diagnosis import split_dataset


def test_r2_score_row():
    """Test of `r2_score` function with row input."""
    y_pred = np.array([3, -0.5, 2, 7])
    y_true = np.array([2.5, 0.0, 2, 8])

    assert r2_score(y_pred, y_true) == pytest.approx(0.957, rel=1e-3)


def test_r2_score_column():
    """Test of `r2_score` function with column input."""
    y_pred = np.array([3, -0.5, 2, 7]).T
    y_true = np.array([2.5, 0.0, 2, 8]).T

    assert r2_score(y_pred, y_true) == pytest.approx(0.957, rel=1e-3)


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
