"""Preprocessors tests."""

import numpy as np
import pytest

from alchina.preprocessors import Normalization, Standardization, PCA


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


# --- PCA ---


def test_pca_with():
    """Test of `PCA` class."""
    X = np.random.normal(0, 1, (150, 2))

    pca = PCA()
    pca.transform(X)
    assert pca.score(X) < 0


def test_pca_with_specified_n_components():
    """Test of `PCA` class."""
    X = np.random.normal(0, 1, (150, 2))

    pca = PCA(n_components=1)
    pca.transform(X)
    assert pca.score(X) < 0


def test_pca_with_invalid_n_components():
    """Test of `PCA` class."""
    X = np.random.normal(0, 1, (150, 2))

    pca = PCA(n_components=5)

    with pytest.raises(ValueError):
        pca.transform(X)
