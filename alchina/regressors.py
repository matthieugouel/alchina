"""Regressors."""

import numpy as np

from abc import ABC, abstractmethod

from .diagnosis import r2_score
from .optimizers import GradientDescent
from .preprocessors import Standardization


class AbstractRegressor(ABC):
    """Abstract class for regressors algorithms."""

    def __init__(self, *args, optimizer=None, standardize: bool = True, **kwargs):
        self.standardize = Standardization() if standardize else None
        self.optimizer = optimizer if optimizer else GradientDescent(*args, **kwargs)
        self.optimizer.build(self.cost_function, self.gradient)

    @abstractmethod
    def cost_function(self, X, y, theta):
        """Cost function."""
        pass  # pragma: no cover

    @abstractmethod
    def gradient(self, X, y, theta):
        """Gradient."""
        pass  # pragma: no cover

    @property
    def parameters(self):
        return self.optimizer.parameters

    @property
    def history(self):
        return self.optimizer.history

    def fit(self, X, y):
        """Fit the model."""
        if self.standardize is not None:
            X = self.standardize(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.optimizer(X, y)

    def predict(self, X):
        """Predict a target given features."""
        if self.standardize is not None:
            X = self.standardize(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return self.hypothesis(X, self.parameters)

    def score(self, X, y):
        """Score of the model."""
        return r2_score(self.predict(X), y)


class LinearRegressor(AbstractRegressor):
    """Linear regressor."""

    def hypothesis(self, X, theta):
        """Linear hypothesis."""
        return np.dot(X, theta)

    def cost_function(self, X, y, theta):
        """Cost function."""
        return (1 / 2 * y.shape[0]) * (self.hypothesis(X, theta) - y).T.dot(
            self.hypothesis(X, theta) - y
        ).flat[0]

    def gradient(self, X, y, theta):
        """Gradient."""
        return (1 / y.shape[0]) * X.T.dot(self.hypothesis(X, theta) - y)

    def normal(self, X, y):
        """Use normal equation to compute the parameters."""
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.optimizer.parameters = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)


class LogisticRegressor(AbstractRegressor):
    """Logistic regressor."""

    def sigmoid(self, z):
        """Logistic function."""
        return 1 / (1 + np.exp(-z))

    def hypothesis(self, X, theta):
        """Logistic hypothesis."""
        return self.sigmoid(np.dot(X, theta))

    def cost_function(self, X, y, theta):
        """Cost function."""
        return (1 / y.shape[0]) * (
            -y.T.dot(np.log(self.hypothesis(X, theta)))
            - (1 - y).T.dot(np.log(1 - self.hypothesis(X, theta)))
        ).flat[0]

    def gradient(self, X, y, theta):
        """Gradient."""
        return (1 / y.shape[0]) * X.T.dot(self.hypothesis(X, theta) - y)
