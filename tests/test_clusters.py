"""Clusters tests."""

import numpy as np

from alchina.clusters import KMeans


# --- K-Means ---


def test_kmeans_fitting():
    """Test of `KMeans` model fitting."""
    km = KMeans(n_centroids=2)

    X = np.array([[0], [1]])

    km.fit(X)

    assert km.score(X) == 0


def test_kmeans_prediction():
    """Test of `KMeans` model prediction."""
    km = KMeans(n_centroids=2)

    X = np.array([[0], [1]])

    km.fit(X)
    prediction = km.predict(X)

    assert np.array_equal(np.sort(prediction), np.array([0, 1]))
