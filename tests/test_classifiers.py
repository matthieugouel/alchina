"""Classifiers tests."""

import numpy as np
import pytest

from alchina.classifiers import LinearClassifier, RidgeClassifier


# --- Linear classifier ---


def test_linear_classifier():
    """Test of `LinearClassifier` class."""
    lc = LinearClassifier(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[1]])
    y = np.array([[1]])

    lc.fit(X, y)

    assert lc.score(X, y) == 1


def test_linear_classifier_standardized():
    """Test of `LinearClassifier` class with standardization."""
    lc = LinearClassifier(learning_rate=0.1, iterations=1, standardize=True)

    X = np.array([[1]])
    y = np.array([[1]])

    lc.fit(X, y)

    assert lc.score(X, y) == 1


def test_linear_classifier_history_enabled():
    """Test of `LinearClassifier` when history enabled."""
    lc = LinearClassifier(
        learning_rate=0.1, iterations=1, history=True, standardize=False
    )

    assert lc.history == []

    X = np.array([[1]])
    y = np.array([[1]])

    lc.fit(X, y)

    assert lc.history is not None


def test_linear_classifier_history_disabled():
    """Test of `LinearClassifier` when history disabled."""
    lc = LinearClassifier(
        learning_rate=0.1, iterations=1, history=False, standardize=False
    )

    assert lc.history is None

    X = np.array([[1]])
    y = np.array([[1]])

    lc.fit(X, y)

    assert lc.history is None


def test_linear_classifier_dataset_inconsistancy():
    """Test of `LinearClassifier` with dataset inconsistancy."""
    lc = LinearClassifier(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[1], [1]])
    y = np.array([[1]])

    with pytest.raises(ValueError):
        lc.fit(X, y)


# --- Ridge classifier ---


def test_ridge_classifier():
    """Test of `RidgeClassifier` class."""
    rc = RidgeClassifier(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[1]])
    y = np.array([[1]])

    rc.fit(X, y)

    assert rc.score(X, y) == 1


def test_ridge_classifier_standardized():
    """Test of `RidgeClassifier` class with standardization."""
    rc = RidgeClassifier(learning_rate=0.1, iterations=1, standardize=True)

    X = np.array([[1]])
    y = np.array([[1]])

    rc.fit(X, y)

    assert rc.score(X, y) == 1


def test_ridge_classifier_history_enabled():
    """Test of `RidgeClassifier` when history enabled."""
    rc = RidgeClassifier(
        learning_rate=0.1, iterations=1, history=True, standardize=False
    )

    assert rc.history == []

    X = np.array([[1]])
    y = np.array([[1]])

    rc.fit(X, y)

    assert rc.history is not None


def test_ridge_classifier_history_disabled():
    """Test of `RidgeClassifier` when history disabled."""
    rc = RidgeClassifier(
        learning_rate=0.1, iterations=1, history=False, standardize=False
    )

    assert rc.history is None

    X = np.array([[1]])
    y = np.array([[1]])

    rc.fit(X, y)

    assert rc.history is None


def test_ridge_classifier_dataset_inconsistancy():
    """Test of `RidgeClassifier` with dataset inconsistancy."""
    rc = RidgeClassifier(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[1], [1]])
    y = np.array([[1]])

    with pytest.raises(ValueError):
        rc.fit(X, y)
