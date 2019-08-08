"""Principal Component Analysis tests."""

import numpy as np
import pytest

from alchina.exceptions import InvalidInput, NotFitted
from alchina.preprocessors import PCA


# --- PCA ---


def test_pca():
    """Test of `PCA` class."""
    X = np.random.normal(0, 1, (150, 2))

    pca = PCA()
    pca.fit_transform(X)
    assert pca.score(X) < 0


def test_pca_with_specified_n_components():
    """Test of `PCA` class with components number specified."""
    X = np.random.normal(0, 1, (150, 2))

    pca = PCA(n_components=1)
    pca.fit_transform(X)
    assert pca.score(X) < 0


def test_pca_with_invalid_n_components():
    """Test of `PCA` class with invalid components number."""
    X = np.random.normal(0, 1, (150, 2))

    pca = PCA(n_components=5)

    with pytest.raises(InvalidInput):
        pca.fit_transform(X)


def test_pca_transform_not_fitted():
    """Test of `PCA` class with transformation without fit."""
    X = np.random.normal(0, 1, (150, 2))

    pca = PCA()

    with pytest.raises(NotFitted):
        pca.transform(X)


def test_pca_score_not_fitted():
    """Test of `PCA` class using score calculation without fit."""
    X = np.random.normal(0, 1, (150, 2))

    pca = PCA()

    with pytest.raises(NotFitted):
        pca.score(X)
