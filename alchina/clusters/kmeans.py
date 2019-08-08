"""K-Means."""

import numpy as np


class KMeans(object):
    """K-Means algorithm."""

    def __init__(self, n_centroids, max_iterations=300):
        self.n_centroids = n_centroids
        self.centroids = None
        self.indexes = []

        self.iterations = 0
        self.max_iterations = max_iterations

    def init_centroids(self, X):
        """Initialize the centroids based on the dataset."""
        self.centroids = np.random.permutation(X)[0 : self.n_centroids]
        return self.centroids

    def find_closest_centroid(self, x):
        """Find the closest centroid.

        Returns the centroid index and the distance between the sample and the centroid.
        """
        distances = [np.linalg.norm(x - c) for c in self.centroids]

        centroid_distance = min(distances)
        centroid_index = distances.index(centroid_distance)
        return centroid_index, centroid_distance

    def update_centroids(self, X, indexes):
        """Update the centroids."""
        for i in range(self.centroids.shape[0]):
            self.centroids[i] = np.mean(X[np.array(indexes) == i], axis=0)

    def fit(self, X):
        """Fit the model."""
        self.init_centroids(X)
        for self.iterations in range(self.max_iterations):
            new_indexes = [self.find_closest_centroid(x)[0] for x in X]
            self.update_centroids(X, new_indexes)
            if self.indexes == new_indexes:
                break
            self.indexes = new_indexes

    def predict(self, X):
        """Predict a target given features."""
        return np.array([self.find_closest_centroid(x)[0] for x in X])

    def score(self, X):
        """Score of the model."""
        costs = np.array([self.find_closest_centroid(x)[1] for x in X])
        return -np.sum(costs ** 2)
