"""Optimizers tests."""

import pytest
import numpy as np

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

    with pytest.raises(ValueError):
        gd(X, y)


# --- Stochastic gradient descent ---


def test_sgd():
    """Test of `SGD`."""
    gd = SGD(iterations=1)

    X = np.array([[1]])
    y = np.array([[2]])

    gd.build(lambda X, y, theta: theta, lambda X, y, theta: theta)
    assert gd(X, y) == np.zeros(1)


def test_sgd_history_enabled():
    """Test of `SGD` when history enabled."""
    gd = SGD(iterations=1, history=True)

    X = np.array([[1]])
    y = np.array([[2]])

    gd.build(lambda X, y, theta: theta, lambda X, y, theta: theta)
    assert gd(X, y) == np.zeros(1)


def test_sgd_no_build():
    """Test of `SGD` when no build."""

    gd = SGD(iterations=1)

    X = np.array([[1]])
    y = np.array([[1]])

    with pytest.raises(ValueError):
        gd(X, y)


# --- Stochastic gradient descent ---


def test_mbgd():
    """Test of `MBGD`."""
    gd = MBGD(iterations=1)

    X = np.array([[1]])
    y = np.array([[2]])

    gd.build(lambda X, y, theta: theta, lambda X, y, theta: theta)
    assert gd(X, y) == np.zeros(1)


def test_mbgd_history_enabled():
    """Test of `MBGD` when history enabled."""
    gd = MBGD(iterations=1, history=True)

    X = np.array([[1]])
    y = np.array([[2]])

    gd.build(lambda X, y, theta: theta, lambda X, y, theta: theta)
    assert gd(X, y) == np.zeros(1)


def test_mbgd_no_build():
    """Test of `MBGD` when no build."""

    gd = MBGD(iterations=1)

    X = np.array([[1]])
    y = np.array([[1]])

    with pytest.raises(ValueError):
        gd(X, y)
