"""K-Nearest Neighbors"""

import numpy as np

from collections import Counter

from alchina.exceptions import NotFitted
from alchina.metrics import accuracy_score


class KNNClassifier(object):
    """K-Nearest Neighbors algorithm"""

    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

        self.X_fit = None
        self.y_fit = None

    def euclidian(self, a, b):
        """Compute the euclidian distance between two samples."""
        return np.linalg.norm(a - b)

    def fit(self, X, y):
        """Train the model."""
        self.X_fit = X
        self.y_fit = y

    def predict(self, X):
        """Predict a target given features."""
        if self.X_fit is None or self.y_fit is None:
            raise NotFitted("the model must be fitted before usage")

        labels = []
        for x in X:
            distances_labels = [
                (self.euclidian(x, x_fit), y_fit)
                for x_fit, y_fit in zip(self.X_fit, self.y_fit)
            ]
            neighbors = sorted(distances_labels, key=lambda d: d[0])[: self.n_neighbors]
            neighbors_labels = [neighbor[1][0] for neighbor in neighbors]
            labels.append(
                sorted(
                    neighbors_labels, key=Counter(neighbors_labels).get, reverse=True
                )[0]
            )
        return np.array(labels).reshape(-1, 1)

    def score(self, X, y):
        """Score of the model."""
        if self.X_fit is None or self.y_fit is None:
            raise NotFitted("the model must be fitted before usage")
        return accuracy_score(self.predict(X), y)
