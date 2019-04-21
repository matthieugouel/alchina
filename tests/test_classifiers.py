"""Classifiers tests."""

import numpy as np

from alchina.classifiers import LinearClassifier, RidgeClassifier


# --- Linear classifier ---


def test_linear_classifier():
    """Test of `LinearClassifier` class."""
    lr = LinearClassifier(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.score(X, y) == 1


def test_linear_classifier_standardized():
    """Test of `LinearClassifier` class with standardization."""
    lr = LinearClassifier(learning_rate=0.1, iterations=1, standardize=True)

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.score(X, y) == 1


def test_linear_classifier_history_enabled():
    """Test of `LinearClassifier` when history enabled."""

    lr = LinearClassifier(
        learning_rate=0.1, iterations=1, history=True, standardize=False
    )

    assert lr.history == []

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.history is not None


def test_linear_classifier_history_disabled():
    """Test of `LinearClassifier` when history disabled."""

    lr = LinearClassifier(
        learning_rate=0.1, iterations=1, history=False, standardize=False
    )

    assert lr.history is None

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.history is None


# --- Ridge classifier ---


def test_ridge_classifier():
    """Test of `RidgeClassifier` class."""
    lr = RidgeClassifier(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.score(X, y) == 1


def test_ridge_classifier_standardized():
    """Test of `RidgeClassifier` class with standardization."""
    lr = RidgeClassifier(learning_rate=0.1, iterations=1, standardize=True)

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.score(X, y) == 1


def test_ridge_classifier_history_enabled():
    """Test of `RidgeClassifier` when history enabled."""

    lr = RidgeClassifier(
        learning_rate=0.1, iterations=1, history=True, standardize=False
    )

    assert lr.history == []

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.history is not None


def test_ridge_classifier_history_disabled():
    """Test of `RidgeClassifier` when history disabled."""

    lr = RidgeClassifier(
        learning_rate=0.1, iterations=1, history=False, standardize=False
    )

    assert lr.history is None

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.history is None
