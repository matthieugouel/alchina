"""Optimizers algorithms."""

import numpy as np

from abc import ABC, abstractmethod
from typing import Optional


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

        self.parameters = np.zeros((X.shape[1], 1))
        for _ in range(self.iterations):
            self.parameters = self.parameters - self.learning_rate * self.gradient(
                X, y, self.parameters, *args, **kwargs
            )

            if self.is_history_enabled:
                self.history.append(self.function(X, y, self.parameters))

        return self.parameters
