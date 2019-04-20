"""Regressors tests."""

import numpy as np

from alchina.regressors import LinearRegressor, LogisticRegressor


def test_linear_regressor():
    """Test of `LinearRegressor` class."""
    lr = LinearRegressor(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.score(X, y) == 1


def test_linear_regressor_standardized():
    """Test of `LinearRegressor` class with standardization."""
    lr = LinearRegressor(learning_rate=0.1, iterations=1, standardize=True)

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.score(X, y) == 1


def test_linear_regressor_normal():
    """Test of `normal` method of `LinearRegressor` class."""

    lr = LinearRegressor(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[1]])
    y = np.array([[1]])

    lr.normal(X, y)

    assert lr.score(X, y) == 1


def test_linear_regressor_history_enabled():
    """Test of `LinearRegressor` history when enabled."""

    lr = LinearRegressor(
        learning_rate=0.1, iterations=1, history=True, standardize=False
    )

    assert lr.history == []

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.history is not None


def test_linear_regressor_history_disabled():
    """Test of `LinearRegressor` history when disabled."""

    lr = LinearRegressor(
        learning_rate=0.1, iterations=1, history=False, standardize=False
    )

    assert lr.history is None

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.history is None


def test_logistic_regressor():
    """Test of `LogisticRegressor` class."""
    lr = LogisticRegressor(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.score(X, y) == 1


def test_logistic_regressor_standardized():
    """Test of `LogisticRegressor` class with standardization."""
    lr = LogisticRegressor(learning_rate=0.1, iterations=1, standardize=True)

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.score(X, y) == 1


def test_logistic_regressor_history_enabled():
    """Test of `LogisticRegressor` history when enabled."""

    lr = LogisticRegressor(
        learning_rate=0.1, iterations=1, history=True, standardize=False
    )

    assert lr.history == []

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.history is not None


def test_logistic_regressor_history_disabled():
    """Test of `LogisticRegressor` history when disabled."""

    lr = LogisticRegressor(
        learning_rate=0.1, iterations=1, history=False, standardize=False
    )

    assert lr.history is None

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.history is None
