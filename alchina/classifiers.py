"""Classifiers."""

import numpy as np

from abc import ABC, abstractmethod

from .diagnosis import r2_score
from .optimizers import GradientDescent
from .preprocessors import Standardization
from .utils import check_dataset_consistancy


class AbstractClassifier(ABC):
    """Abstract class for classifiers algorithms."""

    def __init__(self, *args, optimizer=None, standardize: bool = True, **kwargs):
        self.standardize = Standardization() if standardize else None
        self.optimizer = optimizer if optimizer else GradientDescent(*args, **kwargs)
        self.optimizer.build(self.cost, self.gradient)
        self.parameters = None
        self.labels = None

    @abstractmethod
    def cost(self, X, y, theta):
        """Cost function."""
        pass  # pragma: no cover

    @abstractmethod
    def gradient(self, X, y, theta):
        """Gradient."""
        pass  # pragma: no cover

    @property
    def history(self):
        return self.optimizer.history

    def fit(self, X, y):
        """Fit the model."""
        if not check_dataset_consistancy(X, y):
            raise ValueError("the features set and target set must have as many rows")

        if self.standardize is not None:
            X = self.standardize(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        self.labels = np.unique(y)
        labels_number = np.size(self.labels)
        if labels_number < 2:
            raise ValueError("target must have at least two different classes")
        elif labels_number == 2:
            self.parameters = self.optimizer(X, y)
        else:
            self.parameters = self.optimizer(X, (y == self.labels).astype(int))

    def predict_probability(self, X):
        """Predict the probability of a target given features."""
        if self.standardize is not None:
            X = self.standardize(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return self.hypothesis(X, self.parameters)

    def predict(self, X):
        """Predict a target given features."""
        probability = self.predict_probability(X)
        if np.size(probability, axis=1) > 1:
            return self.labels[np.argmax(probability, axis=1).reshape(-1, 1)]
        return self.labels[np.around(probability).astype("int")]

    def score(self, X, y):
        """Score of the model."""
        return r2_score(self.predict(X), y)


class LinearClassifier(AbstractClassifier):
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
