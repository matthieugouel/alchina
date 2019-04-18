"""Regressors."""

import numpy as np

from abc import ABC, abstractmethod

from .diagnosis import r2_score
from .preprocessors import Standardization


class AbstractRegressor(ABC):
    """Abstract class for regression algorithms."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        iterations: int = 100,
        standardize: bool = True,
    ):
        self.learning_rate: float = learning_rate
        self.iterations: int = iterations
        self.standardize = Standardization() if standardize else None

        self.parameters = None
        self.history: list = []

    @abstractmethod
    def hypothesis(self, X):
        """Regression hypothesis."""
        pass  # pragma: no cover

    def cost(self, X, y):
        """Calculate the cost."""
        return (1 / 2 * y.shape[0]) * (self.hypothesis(X) - y).T.dot(
            self.hypothesis(X) - y
        ).flat[0]

    def gradient_descent(self, X, y):
        """Batch Gradient Descent algorithm."""
        for _ in range(self.iterations):
            self.parameters = self.parameters - (
                self.learning_rate / y.shape[0]
            ) * X.T.dot(self.hypothesis(X) - y)

            self.history.append(self.cost(X, y))

    def fit(self, X, y):
        """Fit the model."""
        if self.standardize is not None:
            X = self.standardize(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.parameters = np.zeros((X.shape[1], 1))
        self.gradient_descent(X, y)

    def predict(self, X):
        """Predict a target given features."""
        if self.standardize is not None:
            X = self.standardize(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return self.hypothesis(X)

    def score(self, X, y):
        """Score of the model."""
        return r2_score(self.predict(X), y)


class LinearRegressor(AbstractRegressor):
    """Linear regressor."""

    def hypothesis(self, X):
        """Linear hypothesis."""
        return np.dot(X, self.parameters)

    def normal(self, X, y):
        """Use normal equation to compute the parameters."""
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.parameters = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
