"""Optimizers tests."""

import pytest
import numpy as np

from alchina.optimizers import GradientDescent


def test_gradient_descent_no_build():
    """Test of `GradientDescent` when no build."""

    gd = GradientDescent(iterations=1)

    X = np.array([[1]])
    y = np.array([[1]])

    with pytest.raises(ValueError):
        gd(X, y)
