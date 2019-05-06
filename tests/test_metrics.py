"""Metrics tests."""

import numpy as np
import pytest

from alchina.metrics import accuracy_score, r2_score


def test_accuracy_score():
    """Test of `accuracy_score` function."""
    y_pred = np.array([0, 2, 1, 3])
    y_true = np.array([0, 1, 2, 3])

    assert accuracy_score(y_pred, y_true) == 0.5


def test_accuracy_score_count():
    """Test of `accuracy_score` function with count option."""
    y_pred = np.array([0, 2, 1, 3])
    y_true = np.array([0, 1, 2, 3])

    assert accuracy_score(y_pred, y_true, count=True) == 2


def test_accuracy_score_inconsistency():
    """Test of `accuracy_score` function."""
    y_pred = np.array([3, -0.5, 2, 7])
    y_true = np.array([2.5, 0.0, 2])

    with pytest.raises(ValueError):
        accuracy_score(y_pred, y_true)


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
