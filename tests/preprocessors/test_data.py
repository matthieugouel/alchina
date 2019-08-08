"""Data Preprocessors tests."""

import numpy as np
import pytest

from alchina.preprocessors import Normalization, Standardization


# --- Normalization ---


def test_normalization():
    """Test of `Normalization` class."""
    normalize = Normalization()

    X = np.array([1, 10, 100, 5, 0.01])
    X_norm = normalize(X)

    assert np.all((X_norm >= 0) & (X_norm <= 1))


# --- Standardization ---


def test_standardization():
    """Test of `Standardization` class."""
    standardize = Standardization()

    X = np.array([1, 10, 100, 5, 0.01])
    X_stand = standardize(X)

    assert np.mean(X_stand) == pytest.approx(0)
    assert np.std(X_stand) == pytest.approx(1)
