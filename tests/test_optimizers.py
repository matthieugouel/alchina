"""Optimizers tests."""

import pytest
import numpy as np

from alchina.exceptions import InvalidInput, NotBuilt
from alchina.optimizers import GradientDescent, SGD, MBGD


# --- Gradient descent ---


def test_gradient_descent():
    """Test of `GradientDescent`."""
    gd = GradientDescent(iterations=1)

    X = np.array([[1]])
    y = np.array([[2]])

    gd.build(lambda X, y, theta: theta, lambda X, y, theta: theta)
    assert gd(X, y) == np.zeros(1)


def test_gradient_descent_history_enabled():
    """Test of `GradientDescent` when history enabled."""
    gd = GradientDescent(iterations=1, history=True)

    X = np.array([[1]])
    y = np.array([[2]])

    gd.build(lambda X, y, theta: theta, lambda X, y, theta: theta)
    assert gd(X, y) == np.zeros(1)


def test_gradient_descent_no_build():
    """Test of `GradientDescent` when no build."""
    gd = GradientDescent(iterations=1)

    X = np.array([[1]])
    y = np.array([[1]])

    with pytest.raises(NotBuilt):
        gd(X, y)


def test_gradient_descent_dataset_inconsistancy():
    """Test of `GradientDescent` with dataset inconsistancy."""
    gd = GradientDescent(iterations=1)

    X = np.array([[1], [1]])
    y = np.array([[1]])

    gd.build(lambda X, y, theta: theta, lambda X, y, theta: theta)

    with pytest.raises(InvalidInput):
        gd(X, y)


# --- Stochastic gradient descent ---


def test_sgd():
    """Test of `SGD`."""
    sgd = SGD(iterations=1)

    X = np.array([[1]])
    y = np.array([[2]])

    sgd.build(lambda X, y, theta: theta, lambda X, y, theta: theta)
    assert sgd(X, y) == np.zeros(1)


def test_sgd_history_enabled():
    """Test of `SGD` when history enabled."""
    sgd = SGD(iterations=1, history=True)

    X = np.array([[1]])
    y = np.array([[2]])

    sgd.build(lambda X, y, theta: theta, lambda X, y, theta: theta)
    assert sgd(X, y) == np.zeros(1)


def test_sgd_no_build():
    """Test of `SGD` when no build."""
    sgd = SGD(iterations=1)

    X = np.array([[1]])
    y = np.array([[1]])

    with pytest.raises(NotBuilt):
        sgd(X, y)


def test_sgd_dataset_inconsistancy():
    """Test of `SGD` with dataset inconsistancy."""
    sgd = SGD(iterations=1)

    X = np.array([[1], [1]])
    y = np.array([[1]])

    sgd.build(lambda X, y, theta: theta, lambda X, y, theta: theta)

    with pytest.raises(InvalidInput):
        sgd(X, y)


# --- Mini-batch gradient descent ---


def test_mbgd():
    """Test of `MBGD`."""
    mbgd = MBGD(iterations=1)

    X = np.array([[1]])
    y = np.array([[2]])

    mbgd.build(lambda X, y, theta: theta, lambda X, y, theta: theta)
    assert mbgd(X, y) == np.zeros(1)


def test_mbgd_history_enabled():
    """Test of `MBGD` when history enabled."""
    mbgd = MBGD(iterations=1, history=True)

    X = np.array([[1]])
    y = np.array([[2]])

    mbgd.build(lambda X, y, theta: theta, lambda X, y, theta: theta)
    assert mbgd(X, y) == np.zeros(1)


def test_mbgd_no_build():
    """Test of `MBGD` when no build."""
    mbgd = MBGD(iterations=1)

    X = np.array([[1]])
    y = np.array([[1]])

    with pytest.raises(NotBuilt):
        mbgd(X, y)


def test_mbgd_dataset_inconsistancy():
    """Test of `MBGD` with dataset inconsistancy."""
    mbgd = MBGD(iterations=1)

    X = np.array([[1], [1]])
    y = np.array([[1]])

    mbgd.build(lambda X, y, theta: theta, lambda X, y, theta: theta)

    with pytest.raises(InvalidInput):
        mbgd(X, y)
