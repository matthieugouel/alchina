"""K-Nearest Neighbors tests."""

import numpy as np
import pytest

from alchina.classifiers import KNNClassifier
from alchina.exceptions import NotFitted


# --- Linear classifier ---


def test_knn_classifier():
    """Test of `KNNClassifier` class."""
    knn = KNNClassifier(1)

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    knn.fit(X, y)

    assert knn.score(X, y) == 1


def test_knn_classifier_predict():
    """Test of `KNNClassifier` class with a prediction."""
    knn = KNNClassifier(1)

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    knn.fit(X, y)

    assert np.equal(knn.predict(np.array([0])), np.array([0]))


def test_knn_classifier_multiclass():
    """Test of `LinearClassifier` with multiclass."""
    knn = KNNClassifier(1)

    X = np.array([[0], [1], [2]])
    y = np.array([[0], [1], [2]])

    knn.fit(X, y)

    assert knn.score(X, y) == 1


def test_knn_classifier_predict_not_fitted():
    """Test of `KNNClassifier` class with prediction without fit."""
    knn = KNNClassifier(1)

    X = np.array([[0], [1]])

    with pytest.raises(NotFitted):
        knn.predict(X)


def test_knn_classifier_score_not_fitted():
    """Test of `KNNClassifier` class with score calculation without fit."""
    knn = KNNClassifier(1)

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    with pytest.raises(NotFitted):
        knn.score(X, y) == 1
