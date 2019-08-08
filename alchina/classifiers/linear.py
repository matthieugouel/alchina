"""Linear Classifiers."""

import numpy as np

from abc import ABC, abstractmethod

from alchina.exceptions import InvalidInput, NotFitted
from alchina.metrics import accuracy_score
from alchina.optimizers import GradientDescent
from alchina.preprocessors import Standardization
from alchina.utils import check_dataset_consistency, features_reshape


class AbstractLinearClassifier(ABC):
    """Abstract class for linear classifiers algorithms."""

    def __init__(self, *args, optimizer=None, standardize: bool = True, **kwargs):
        self.standardize = Standardization() if standardize else None
        self.optimizer = optimizer if optimizer else GradientDescent(*args, **kwargs)
        self.optimizer.build(self.cost, self.gradient)

        self.labels = None

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

        self.labels = np.unique(y)
        n_labels = np.size(self.labels)
        if n_labels < 2:
            raise InvalidInput("target must have at least two different classes")
        elif n_labels == 2:
            self.optimizer(X, y)
        else:
            self.optimizer(X, (y == self.labels).astype(int))

    def predict_probability(self, X):
        """Predict the probability of a target given features."""
        if self.parameters is None or self.labels is None:
            raise NotFitted("the model must be fitted before usage")

        X = features_reshape(X)
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
        if self.parameters is None or self.labels is None:
            raise NotFitted("the model must be fitted before usage")
        return accuracy_score(self.predict(X), y)


class LinearClassifier(AbstractLinearClassifier):
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


class RidgeClassifier(AbstractLinearClassifier):
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
