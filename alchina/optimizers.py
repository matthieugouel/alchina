"""Optimizers algorithms."""

import numpy as np

from abc import ABC, abstractmethod
from typing import Optional

from .utils import shuffle_dataset


class AbstractOptimizer(ABC):
    """Abstract class for optimizers algorithms."""

    def __init__(self, iterations: int = 100, history: bool = False):
        self.iterations = iterations

        self.function = None
        self.gradient = None

        self.is_history_enabled = history
        self.history: Optional[list] = [] if history else None

    def build(self, function, gradient):
        self.function = function
        self.gradient = gradient

    @abstractmethod
    def __call__(self):
        pass  # pragma: no cover


class GradientDescent(AbstractOptimizer):
    """Batch gradient descent."""

    def __init__(self, *args, learning_rate: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate

        self.parameters = None

    def __call__(self, X, y, *args, **kwargs):
        if not (self.function and self.gradient):
            raise ValueError("you must build the optimizer before calling it")

        m = y.shape[0]

        self.parameters = np.zeros((X.shape[1], 1))
        for _ in range(self.iterations):
            self.parameters = self.parameters - (
                self.learning_rate / m
            ) * self.gradient(X, y, self.parameters, *args, **kwargs)

            if self.is_history_enabled:
                self.history.append((1 / m) * self.function(X, y, self.parameters))

        return self.parameters


class SGD(AbstractOptimizer):
    """Stochastic gradient descent."""

    def __init__(
        self, *args, learning_rate: float = 0.01, iterations: int = 1, **kwargs
    ):
        kwargs.update({"iterations": iterations})
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate

        self.parameters = None

    def __call__(self, X, y, *args, **kwargs):
        if not (self.function and self.gradient):
            raise ValueError("you must build the optimizer before calling it")

        X, y = shuffle_dataset(X, y)

        self.parameters = np.zeros((X.shape[1], 1))
        for _ in range(self.iterations):
            for X_i, y_i in zip(X, y):
                self.parameters = self.parameters - self.learning_rate * self.gradient(
                    X_i[None, :], y_i[None, :], self.parameters, *args, **kwargs
                )

                if self.is_history_enabled:
                    self.history.append(self.function(X, y, self.parameters))

        return self.parameters


class MBGD(AbstractOptimizer):
    """Mini-batch gradient descent."""

    def __init__(
        self,
        *args,
        learning_rate: float = 0.01,
        batch_size: int = 10,
        iterations: int = 10,
        **kwargs,
    ):
        kwargs.update({"iterations": iterations})
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.parameters = None

    def __call__(self, X, y, *args, **kwargs):
        if not (self.function and self.gradient):
            raise ValueError("you must build the optimizer before calling it")

        X, y = shuffle_dataset(X, y)
        m = y.shape[0]

        self.parameters = np.zeros((X.shape[1], 1))
        for _ in range(self.iterations):
            for i in range(0, m, self.batch_size):
                self.parameters = self.parameters - (
                    self.learning_rate / m
                ) * self.gradient(
                    X[i : i + self.batch_size],
                    y[i : i + self.batch_size],
                    self.parameters,
                    *args,
                    **kwargs,
                )

            if self.is_history_enabled:
                self.history.append(
                    (1 / self.batch_size) * self.function(X, y, self.parameters)
                )

        return self.parameters
