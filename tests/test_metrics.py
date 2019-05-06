"""Metrics tests."""

import numpy as np
import pytest

from alchina.metrics import r2_score


def test_r2_score_row():
    """Test of `r2_score` function with row input."""
    y_pred = np.array([3, -0.5, 2, 7])
    y_true = np.array([2.5, 0.0, 2, 8])

    assert r2_score(y_pred, y_true) == pytest.approx(0.957, rel=1e-3)


def test_r2_score_column():
    """Test of `r2_score` function with column input."""
    y_pred = np.array([3, -0.5, 2, 7]).T
    y_true = np.array([2.5, 0.0, 2, 8]).T

    assert r2_score(y_pred, y_true) == pytest.approx(0.957, rel=1e-3)


def test_r2_score_inconsistancy():
    """Test of `r2_score` with input inconsistency."""
    y_pred = np.array([3, -0.5, 2, 7])
    y_true = np.array([2.5, 0.0, 2])

    with pytest.raises(ValueError):
        r2_score(y_pred, y_true)
