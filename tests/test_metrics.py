"""Metrics tests."""

import numpy as np
import pytest

from alchina.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    f1_score,
    r2_score,
)


# --- Confusion matrix ---


def test_onfusion_matrix_no_prediction():
    """Test of `confusion_matrix` with no prediction."""
    y_pred = np.array([0])
    y_true = np.array([1])

    assert np.array_equal(confusion_matrix(y_pred, y_true), np.array([[0, 0], [1, 0]]))


def test_confusion_matrix_binary_labels():
    """Test of `confusion_matrix` with binary labels."""
    y_pred = np.array([1, 1, 1, 0, 1])
    y_true = np.array([0, 1, 1, 1, 1])

    assert np.array_equal(confusion_matrix(y_pred, y_true), np.array([[0, 1], [1, 3]]))


def test_confusion_matrix_multiclass_labels():
    """Test of `confusion_matrix` with multiclass_labels."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])

    assert np.array_equal(
        confusion_matrix(y_pred, y_true), np.array([[2, 0, 0], [1, 0, 1], [0, 2, 0]])
    )


# --- Accuracy score ---


def test_accuracy_score():
    """Test of `accuracy_score` function."""
    y_pred = np.array([0, 2, 1, 3])
    y_true = np.array([0, 1, 2, 3])

    assert accuracy_score(y_pred, y_true) == 0.5


def test_accuracy_score_no_normalization():
    """Test of `accuracy_score` function with count option."""
    y_pred = np.array([0, 2, 1, 3])
    y_true = np.array([0, 1, 2, 3])

    assert accuracy_score(y_pred, y_true, normalize=False) == 2


def test_accuracy_score_inconsistency():
    """Test of `accuracy_score` function."""
    y_pred = np.array([0, 1])
    y_true = np.array([1])

    with pytest.raises(ValueError):
        accuracy_score(y_pred, y_true)


# --- Precision score ---


def test_precision_score_no_prediction():
    """Test of `precision_score` with no prediction."""
    y_pred = np.array([0])
    y_true = np.array([1])

    assert precision_score(y_pred, y_true) == 0


def test_precision_score_binary_labels():
    """Test of `precision_score` with binary labels."""
    y_pred = np.array([1, 1, 1, 0, 1])
    y_true = np.array([0, 1, 1, 1, 1])

    assert precision_score(y_pred, y_true) == 0.75


def test_precision_score_multiclass_labels():
    """Test of `precision_score` with multiclass labels and no average."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])

    assert np.allclose(
        precision_score(y_pred, y_true), np.array([0.66, 0, 0]), atol=1e-2
    )


def test_precision_score_multiclass_labels_macro():
    """Test of `precision_score` with multiclass labels and `macro` average."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])

    precision = precision_score(y_pred, y_true, average="macro")

    assert precision == pytest.approx(0.22, rel=2e-2)


def test_precision_score_unsupported_average():
    """Test of `precision_score` with unsupported average parameter."""
    y_pred = np.array([0, 1, 2])
    y_true = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        precision_score(y_pred, y_true, average="foobar")


def test_precision_score_inconsistancy():
    """Test of `precision_score` with input inconsistency."""
    y_pred = np.array([0, 1])
    y_true = np.array([1])

    with pytest.raises(ValueError):
        precision_score(y_pred, y_true)


# --- Recall score ---


def test_recall_score_no_prediction():
    """Test of `recall_score` with no prediction."""
    y_pred = np.array([0])
    y_true = np.array([1])

    assert recall_score(y_pred, y_true) == 0


def test_recall_score_binary_labels():
    """Test of `recall_score` with binary labels."""
    y_pred = np.array([1, 1, 1, 0, 1])
    y_true = np.array([0, 1, 1, 1, 1])

    assert recall_score(y_pred, y_true) == 0.75


def test_recall_score_multiclass_labels():
    """Test of `recall_score` with multiclass labels and no average."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])

    assert np.allclose(recall_score(y_pred, y_true), np.array([1, 0, 0]))


def test_recall_score_multiclass_labels_macro():
    """Test of `recall_score` with multiclass labels and `macro` average."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])

    recall = recall_score(y_pred, y_true, average="macro")

    assert recall == pytest.approx(0.33, rel=3e-2)


def test_recall_score_unsupported_average():
    """Test of `recall_score` with unsupported average parameter."""
    y_pred = np.array([0, 1, 2])
    y_true = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        recall_score(y_pred, y_true, average="foobar")


def test_recall_score_inconsistancy():
    """Test of `recall_score` with input inconsistency."""
    y_pred = np.array([0, 1])
    y_true = np.array([1])

    with pytest.raises(ValueError):
        recall_score(y_pred, y_true)


# --- F-Beta score ---


def test_fbeta_score_no_prediction():
    """Test of `fbeta_score` with no prediction."""
    y_pred = np.array([0])
    y_true = np.array([1])

    assert fbeta_score(y_pred, y_true, 0.5) == 0


def test_fbeta_score_binary_labels():
    """Test of `fbeta_score` with binary labels."""
    y_pred = np.array([1, 1, 1, 0, 1])
    y_true = np.array([0, 1, 1, 1, 1])

    assert fbeta_score(y_pred, y_true, 0.5) == 0.75


def test_fbeta_score_multiclass_labels():
    """Test of `fbeta_score` with multiclass labels and no average."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])

    assert np.allclose(
        fbeta_score(y_pred, y_true, 0.5), np.array([0.71, 0, 0]), atol=1e-2
    )


def test_fbeta_score_multiclass_labels_macro():
    """Test of `fbeta_score` with multiclass labels and `macro` average."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])

    fbeta = fbeta_score(y_pred, y_true, 0.5, average="macro")

    assert fbeta == pytest.approx(0.23, rel=4e-2)


def test_fbeta_score_unsupported_average():
    """Test of `fbeta_score` with unsupported average parameter."""
    y_pred = np.array([0, 1, 2])
    y_true = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        fbeta_score(y_pred, y_true, 0.5, average="foobar")


def test_fbeta_score_inconsistancy():
    """Test of `fbeta_score` with input inconsistency."""
    y_pred = np.array([0, 1])
    y_true = np.array([1])

    with pytest.raises(ValueError):
        fbeta_score(y_pred, y_true, 2)


# --- F1 score ---


def test_f1_score_no_prediction():
    """Test of `f1_score` with no prediction."""
    y_pred = np.array([0])
    y_true = np.array([1])

    assert f1_score(y_pred, y_true) == 0


def test_f1_score_binary_labels():
    """Test of `f1_score` with binary labels."""
    y_pred = np.array([1, 1, 1, 0, 1])
    y_true = np.array([0, 1, 1, 1, 1])

    assert f1_score(y_pred, y_true) == 0.75


def test_f1_score_multiclass_labels():
    """Test of `f1_score` with multiclass labels and no average."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])

    assert np.allclose(f1_score(y_pred, y_true), np.array([0.8, 0, 0]))


def test_f1_score_multiclass_labels_macro():
    """Test of `f1_score` with multiclass labels and `macro` average."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])

    assert f1_score(y_pred, y_true, average="macro") == pytest.approx(0.26, rel=6e-2)


def test_f1_score_unsupported_average():
    """Test of `f1_score` with unsupported average parameter."""
    y_pred = np.array([0, 1, 2])
    y_true = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        f1_score(y_pred, y_true, average="foobar")


def test_f1_score_inconsistancy():
    """Test of `f1_score` with input inconsistency."""
    y_pred = np.array([0, 1])
    y_true = np.array([1])

    with pytest.raises(ValueError):
        f1_score(y_pred, y_true)


# --- R2 score ---


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
    y_pred = np.array([0, 1])
    y_true = np.array([1])

    with pytest.raises(ValueError):
        r2_score(y_pred, y_true)
