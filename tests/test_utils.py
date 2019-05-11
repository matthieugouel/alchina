"""Utils tests."""

import numpy as np

from alchina.utils import (
    check_dataset_consistency,
    target_reshape,
    target_labels,
    target_label_count,
    target_label_to_index,
)


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


# --- Target label count ---


def test_target_label_count_single_label():
    """Test of `target_label_count` with binary classification but single label."""
    y = np.array([0])

    assert target_label_count(y) == 2


def test_target_label_count_binary_labels():
    """Test of `target_label_count` with binary classification."""
    y = np.array([0, 1, 1, 0, 1])

    assert target_label_count(y) == 2


def test_target_label_count_multi_labels():
    """Test of `target_label_count` with multiclass classification."""
    y = np.array([0, 1, 2, 0, 1])

    assert target_label_count(y) == 3


# --- Target label to index ---


def test_target_label_to_index_single_label():
    """Test of `target_label_to_index` with binary classification but single label."""
    y = np.array(["a"])

    assert np.array_equal(target_label_to_index(y), np.array([0]))


def test_target_label_to_index_binary_labels():
    """Test of `target_label_to_index` with binary classification."""
    y = np.array(["a", "b", "a"])

    assert np.array_equal(target_label_to_index(y), np.array([0, 1, 0]))


def test_target_label_to_index_multi_labels():
    """Test of `target_label_to_index` with multiclass classification."""
    y = np.array(["a", "b", "c", "b"])

    assert np.array_equal(target_label_to_index(y), np.array([0, 1, 2, 1]))
