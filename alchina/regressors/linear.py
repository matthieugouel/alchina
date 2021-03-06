"""Regressors."""

import numpy as np

from abc import ABC, abstractmethod

from alchina.exceptions import InvalidInput, NotFitted
from alchina.metrics import r2_score
from alchina.optimizers import GradientDescent
from alchina.preprocessors import Standardization
from alchina.utils import check_dataset_consistency, features_reshape


class AbstractLinearRegressor(ABC):
    """Abstract class for regressors algorithms."""

    def __init__(self, *args, optimizer=None, standardize: bool = True, **kwargs):
        self.standardize = Standardization() if standardize else None
        self.optimizer = optimizer if optimizer else GradientDescent(*args, **kwargs)
        self.optimizer.build(self.cost, self.gradient)

    @abstractmethod
    def hypothesis(self, X, theta):
        """Hypothesis."""
        pass  # pragma: no cover

    @abstractmethod
    def cost(self, X, y, theta):
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
        X = features_reshape(X)
        if not check_dataset_consistency(X, y):
            raise InvalidInput("the features set and target set must have as many rows")

        if self.standardize is not None:
            X = self.standardize(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.optimizer(X, y)

    def predict(self, X):
        """Predict a target given features."""
        if self.parameters is None:
            raise NotFitted("the model must be fitted before usage")

        X = features_reshape(X)
        if self.standardize is not None:
            X = self.standardize(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return self.hypothesis(X, self.parameters)

    def score(self, X, y):
        """Score of the model."""
        if self.parameters is None:
            raise NotFitted("the model must be fitted before usage")
        return r2_score(self.predict(X), y)


class LinearRegressor(AbstractLinearRegressor):
    """Linear regressor."""

    def hypothesis(self, X, theta):
        """Linear hypothesis."""
        return np.dot(X, theta)

    def cost(self, X, y, theta):
        """Cost function."""
        return (1 / 2) * (self.hypothesis(X, theta) - y).T.dot(
            self.hypothesis(X, theta) - y
        ).flat[0]

    def gradient(self, X, y, theta):
        """Gradient."""
        return X.T.dot(self.hypothesis(X, theta) - y)

    def normal(self, X, y):
        """Use normal equation to compute the parameters."""
        X = features_reshape(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.optimizer.parameters = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)


class RidgeRegressor(AbstractLinearRegressor):
    """Ridge regressor."""

    def __init__(self, *args, regularization: float = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.regularization = regularization

    def hypothesis(self, X, theta):
        """Linear hypothesis."""
        return np.dot(X, theta)

    def cost(self, X, y, theta):
        """Regularized cost function."""
        return (1 / 2) * (self.hypothesis(X, theta) - y).T.dot(
            self.hypothesis(X, theta) - y
        ).flat[0] + self.regularization * np.sum(np.square(theta[:, 1:]), axis=0)

    def gradient(self, X, y, theta):
        """Regularized gradient."""
        return (
            X.T.dot(self.hypothesis(X, theta) - y)
            + self.regularization * np.c_[np.zeros((theta.shape[0], 1)), theta[:, 1:]]
        )

    def normal(self, X, y):
        """Use normal equation regularized to compute the parameters."""
        X = features_reshape(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        L = np.identity(X.shape[0])
        L[0, 0] = 0
        self.optimizer.parameters = (
            np.linalg.pinv(X.T.dot(X) + self.regularization * L).dot(X.T).dot(y)
        )
