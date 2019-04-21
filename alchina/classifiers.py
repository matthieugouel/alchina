"""Classifiers."""

import numpy as np

from alchina.regressors import AbstractRegressor


class LinearClassifier(AbstractRegressor):
    """Linear classifier (logistic regressor)."""

    def sigmoid(self, z):
        """Logistic function."""
        return 1 / (1 + np.exp(-z))

    def hypothesis(self, X, theta):
        """Logistic hypothesis."""
        return self.sigmoid(np.dot(X, theta))

    def cost(self, X, y, theta):
        """Cost function."""
        return (
            -y.T.dot(np.log(self.hypothesis(X, theta)))
            - (1 - y).T.dot(np.log(1 - self.hypothesis(X, theta)))
        ).flat[0]

    def gradient(self, X, y, theta):
        """Gradient."""
        return X.T.dot(self.hypothesis(X, theta) - y)


class RidgeClassifier(LinearClassifier):
    """Regularized linear classifier."""

    def __init__(self, *args, regularization: float = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.regularization = regularization

    def sigmoid(self, z):
        """Logistic function."""
        return 1 / (1 + np.exp(-z))

    def hypothesis(self, X, theta):
        """Logistic hypothesis."""
        return self.sigmoid(np.dot(X, theta))

    def cost(self, X, y, theta):
        """Regularized cost function."""
        return (
            -y.T.dot(np.log(self.hypothesis(X, theta)))
            - (1 - y).T.dot(np.log(1 - self.hypothesis(X, theta)))
        ).flat[0] + self.regularization * np.sum(np.square(theta[:, 1:]), axis=0)

    def gradient(self, X, y, theta):
        """Regularized gradient."""
        return (
            X.T.dot(self.hypothesis(X, theta) - y)
            + self.regularization * np.c_[np.zeros((theta.shape[0], 1)), theta[:, 1:]]
        )
