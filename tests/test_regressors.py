"""Regressors tests."""

import numpy as np
import pytest

from alchina.exceptions import InvalidInput, NotFitted
from alchina.regressors import LinearRegressor, RidgeRegressor


# --- Linear regressor ---


def test_linear_regressor():
    """Test of `LinearRegressor` class."""
    lr = LinearRegressor(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([1])
    y = np.array([1])

    lr.fit(X, y)

    assert lr.score(X, y) == 1


def test_linear_regressor_standardized():
    """Test of `LinearRegressor` class with standardization."""
    lr = LinearRegressor(learning_rate=0.1, iterations=1, standardize=True)

    X = np.array([1])
    y = np.array([1])

    lr.fit(X, y)

    assert lr.score(X, y) == 1


def test_linear_regressor_normal():
    """Test of `normal` method of `LinearRegressor` class."""
    lr = LinearRegressor(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([1])
    y = np.array([1])

    lr.normal(X, y)

    assert lr.score(X, y) == 1


def test_linear_regressor_history_enabled():
    """Test of `LinearRegressor` when history enabled."""
    lr = LinearRegressor(learning_rate=0.1, iterations=1, history=True)

    assert lr.history == []

    X = np.array([1])
    y = np.array([1])

    lr.fit(X, y)

    assert lr.history is not None


def test_linear_regressor_history_disabled():
    """Test of `LinearRegressor` when history disabled."""
    lr = LinearRegressor(learning_rate=0.1, iterations=1, history=False)

    assert lr.history is None

    X = np.array([1])
    y = np.array([1])

    lr.fit(X, y)

    assert lr.history is None


def test_linear_regressor_dataset_inconsistancy():
    """Test of `LinearRegressor` with dataset inconsistancy."""
    lr = LinearRegressor(learning_rate=0.1, iterations=1)

    X = np.array([[1], [1]])
    y = np.array([[1]])

    with pytest.raises(InvalidInput):
        lr.fit(X, y)


def test_linear_regressor_predict_not_fitted():
    """Test of `LinearRegressor` class with prediction without fit."""
    lr = LinearRegressor(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([1])

    with pytest.raises(NotFitted):
        lr.predict(X)


def test_linear_regressor_score_not_fitted():
    """Test of `LinearRegressor` class with score calculation without fit."""
    lr = LinearRegressor(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([1])
    y = np.array([1])

    with pytest.raises(NotFitted):
        lr.score(X, y)


# --- Ridge regressor ---


def test_ridge_regressor():
    """Test of `RidgeRegressor` class."""
    rr = RidgeRegressor(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([1])
    y = np.array([1])

    rr.fit(X, y)

    assert rr.score(X, y) == 1


def test_ridge_regressor_standardized():
    """Test of `RidgeRegressor` class with standardization."""
    rr = RidgeRegressor(learning_rate=0.1, iterations=1, standardize=True)

    X = np.array([1])
    y = np.array([1])

    rr.fit(X, y)

    assert rr.score(X, y) == 1


def test_ridge_regressor_normal():
    """Test of `normal` method of `RidgeRegressor` class."""
    rr = RidgeRegressor(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([1])
    y = np.array([1])

    rr.normal(X, y)

    assert rr.score(X, y) == 1


def test_ridge_regressor_history_enabled():
    """Test of `RidgeRegressor` when history enabled."""
    rr = RidgeRegressor(learning_rate=0.1, iterations=1, history=True)

    assert rr.history == []

    X = np.array([1])
    y = np.array([1])

    rr.fit(X, y)

    assert rr.history is not None


def test_ridge_regressor_history_disabled():
    """Test of `RidgeRegressor` when history disabled."""
    rr = RidgeRegressor(learning_rate=0.1, iterations=1, history=False)

    assert rr.history is None

    X = np.array([1])
    y = np.array([1])

    rr.fit(X, y)

    assert rr.history is None


def test_ridge_regressor_dataset_inconsistancy():
    """Test of `RidgeRegressor` with dataset inconsistancy."""
    rr = RidgeRegressor(learning_rate=0.1, iterations=1)

    X = np.array([[1], [1]])
    y = np.array([[1]])

    with pytest.raises(InvalidInput):
        rr.fit(X, y)


def test_ridge_regressor_predict_not_fitted():
    """Test of `RidgeRegressor` class with prediction without fit."""
    lr = RidgeRegressor(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([1])

    with pytest.raises(NotFitted):
        lr.predict(X)


def test_ridge_regressor_score_not_fitted():
    """Test of `RidgeRegressor` class with score calculation without fit."""
    lr = RidgeRegressor(learning_rate=0.1, iterations=1, standardize=False)

    X = np.array([1])
    y = np.array([1])

    with pytest.raises(NotFitted):
        lr.score(X, y)
