"""Utils tests."""

import numpy as np

from alchina.utils import check_dataset_consistency


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
