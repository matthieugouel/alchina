"""Linear Classifiers tests."""

import numpy as np
import pytest

from alchina.classifiers import LinearClassifier, RidgeClassifier
from alchina.exceptions import InvalidInput, NotFitted


# --- Linear classifier ---


def test_linear_classifier():
    """Test of `LinearClassifier` class."""
    lc = LinearClassifier(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    lc.fit(X, y)

    assert lc.score(X, y) == 1


def test_linear_classifier_standardized():
    """Test of `LinearClassifier` class with standardization."""
    lc = LinearClassifier(learning_rate=0.1, iterations=1, standardize=True)

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    lc.fit(X, y)

    assert lc.score(X, y) == 1


def test_linear_classifier_predict():
    """Test of `LinearClassifier` class with a prediction."""
    lc = LinearClassifier(learning_rate=0.1, iterations=1, standardize=True)

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    lc.fit(X, y)

    assert np.equal(lc.predict(np.array([0])), np.array([0]))


def test_linear_classifier_history_enabled():
    """Test of `LinearClassifier` when history enabled."""
    lc = LinearClassifier(learning_rate=0.1, iterations=1, history=True)

    assert lc.history == []

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    lc.fit(X, y)

    assert lc.history is not None


def test_linear_classifier_history_disabled():
    """Test of `LinearClassifier` when history disabled."""
    lc = LinearClassifier(learning_rate=0.1, iterations=1, history=False)

    assert lc.history is None

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    lc.fit(X, y)

    assert lc.history is None


def test_linear_classifier_multiclass():
    """Test of `LinearClassifier` with multiclass."""
    lc = LinearClassifier(learning_rate=0.1, iterations=2)

    X = np.array([[0], [1], [2]])
    y = np.array([[0], [1], [2]])

    lc.fit(X, y)

    assert lc.score(X, y) == 1


def test_linear_classifier_one_class_target():
    """Test of `LinearClassifier` with dataset inconsistancy."""
    lc = LinearClassifier(learning_rate=0.1, iterations=1)

    X = np.array([[1]])
    y = np.array([[1]])

    with pytest.raises(InvalidInput):
        lc.fit(X, y)


def test_linear_classifier_dataset_inconsistancy():
    """Test of `LinearClassifier` with dataset inconsistancy."""
    lc = LinearClassifier(learning_rate=0.1, iterations=1)

    X = np.array([[1], [1]])
    y = np.array([[1]])

    with pytest.raises(InvalidInput):
        lc.fit(X, y)


def test_linear_classifier_predict_not_fitted():
    """Test of `LinearClassifier` class with prediction without fit."""
    lc = LinearClassifier(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[0], [1]])

    with pytest.raises(NotFitted):
        lc.predict(X)


def test_linear_classifier_score_not_fitted():
    """Test of `LinearClassifier` class with score calculation without fit."""
    lc = LinearClassifier(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    with pytest.raises(NotFitted):
        lc.score(X, y) == 1


# --- Ridge classifier ---


def test_ridge_classifier():
    """Test of `RidgeClassifier` class."""
    rc = RidgeClassifier(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    rc.fit(X, y)

    assert rc.score(X, y) == 1


def test_ridge_classifier_standardized():
    """Test of `RidgeClassifier` class with standardization."""
    rc = RidgeClassifier(learning_rate=0.1, iterations=1, standardize=True)

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    rc.fit(X, y)

    assert rc.score(X, y) == 1


def test_ridge_classifier_predict():
    """Test of `LinearClassifier` class with a prediction."""
    lc = RidgeClassifier(learning_rate=0.1, iterations=1, standardize=True)

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    lc.fit(X, y)

    assert np.equal(lc.predict(np.array([0])), np.array([0]))


def test_ridge_classifier_history_enabled():
    """Test of `RidgeClassifier` when history enabled."""
    rc = RidgeClassifier(learning_rate=0.1, iterations=1, history=True)

    assert rc.history == []

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    rc.fit(X, y)

    assert rc.history is not None


def test_ridge_classifier_history_disabled():
    """Test of `RidgeClassifier` when history disabled."""
    rc = RidgeClassifier(learning_rate=0.1, iterations=1, history=False)

    assert rc.history is None

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    rc.fit(X, y)

    assert rc.history is None


def test_ridge_classifier_multiclass():
    """Test of `LinearClassifier` with no multiclass."""
    rc = RidgeClassifier(learning_rate=0.1, iterations=2, history=True)

    X = np.array([[0], [1], [2]])
    y = np.array([[0], [1], [2]])

    rc.fit(X, y)

    assert rc.score(X, y) == 1


def test_ridge_classifier_dataset_inconsistancy():
    """Test of `RidgeClassifier` with dataset inconsistancy."""
    rc = RidgeClassifier(learning_rate=0.1, iterations=1)

    X = np.array([[1], [1]])
    y = np.array([[1]])

    with pytest.raises(InvalidInput):
        rc.fit(X, y)


def test_ridge_classifier_predict_not_fitted():
    """Test of `RidgeClassifier` class with prediction without fit."""
    lc = RidgeClassifier(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[0], [1]])

    with pytest.raises(NotFitted):
        lc.predict(X)


def test_ridge_classifier_score_not_fitted():
    """Test of `RidgeClassifier` class with score calculation without fit."""
    lc = RidgeClassifier(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    with pytest.raises(NotFitted):
        lc.score(X, y) == 1
