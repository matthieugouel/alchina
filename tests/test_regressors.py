"""Regressors tests."""

import numpy as np

from alchina.regressors import LinearRegressor, RidgeRegressor


# --- Linear regressor ---


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
    """Test of `LinearRegressor` when history enabled."""

    lr = LinearRegressor(
        learning_rate=0.1, iterations=1, history=True, standardize=False
    )

    assert lr.history == []

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.history is not None


def test_linear_regressor_history_disabled():
    """Test of `LinearRegressor` when history disabled."""

    lr = LinearRegressor(
        learning_rate=0.1, iterations=1, history=False, standardize=False
    )

    assert lr.history is None

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.history is None


# --- Ridge regressor ---


def test_ridge_regressor():
    """Test of `RidgeRegressor` class."""
    lr = RidgeRegressor(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.score(X, y) == 1


def test_ridge_regressor_standardized():
    """Test of `RidgeRegressor` class with standardization."""
    lr = RidgeRegressor(learning_rate=0.1, iterations=1, standardize=True)

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.score(X, y) == 1


def test_ridge_regressor_normal():
    """Test of `normal` method of `RidgeRegressor` class."""

    lr = RidgeRegressor(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([[1]])
    y = np.array([[1]])

    lr.normal(X, y)

    assert lr.score(X, y) == 1


def test_ridge_regressor_history_enabled():
    """Test of `RidgeRegressor` when history enabled."""

    lr = RidgeRegressor(
        learning_rate=0.1, iterations=1, history=True, standardize=False
    )

    assert lr.history == []

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.history is not None


def test_ridge_regressor_history_disabled():
    """Test of `RidgeRegressor` when history disabled."""

    lr = RidgeRegressor(
        learning_rate=0.1, iterations=1, history=False, standardize=False
    )

    assert lr.history is None

    X = np.array([[1]])
    y = np.array([[1]])

    lr.fit(X, y)

    assert lr.history is None
