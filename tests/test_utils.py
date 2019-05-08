"""Utils tests."""

import numpy as np

from alchina.utils import check_dataset_consistency, target_reshape, target_labels


# --- Check dataset consistency ---


def test_check_dataset_consistency_valid():
    """Test of `check_dataset_consistency` with valid input."""
    X = np.arange(9).reshape(3, 3)
    y = np.arange(3).reshape(3, 1)

    assert check_dataset_consistency(X, y)


def test_check_dataset_consistency_invalid():
    """Test of `check_dataset_consistency` with invalid input ."""
    X = np.arange(9).reshape(3, 3)
    y = np.arange(4).reshape(2, 2)

    assert not check_dataset_consistency(X, y)


# --- Target reshape ---


def test_target_reshape_1D():
    """Test of `target_reshape` with a 1-D array."""

    y = np.array([0, 1])

    assert np.array_equal(target_reshape(y), np.array([0, 1]))


def test_target_reshape_2D():
    """Test of `target_reshape` with a 2-D array."""

    y = np.array([[0], [1]])

    assert np.array_equal(target_reshape(y), np.array([0, 1]))


# --- Target labels ----


def test_target_labels():
    """Test of `target_labels`."""

    y = np.array([0, 1, 1, 2, 3])

    assert np.array_equal(target_labels(y), np.array([0, 1, 2, 3]))
