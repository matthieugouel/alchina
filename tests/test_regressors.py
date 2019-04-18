"""Regressors tests."""

import numpy as np

from alchina.regressors import LinearRegressor


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
