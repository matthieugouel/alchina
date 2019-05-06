"""Preprocessors."""

import numpy as np

from typing import Optional


class Normalization(object):
    """Rescale the data via a normalization.

    Produce:
    - Bring all values into the range [0, 1]
    """

    def __call__(self, X, axis: int = 0):
        min_x = np.amin(X, axis=axis)
        max_x = np.amax(X, axis=axis)
        return (X - min_x) / (max_x - min_x)


class Standardization(object):
    """Rescale the data via a standardization

    Produce:
    - mean(Xstandardized) = 0
    - std(Xstandardized) = 1
    """

    def __init__(self, mu: Optional[int] = None, sigma: Optional[int] = None):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, X, axis: int = 0):
        if self.mu is None or self.sigma is None:
            self.mu = np.mean(X, axis=axis)
            self.sigma = np.std(X, axis=axis)

            if not np.any(self.sigma):
                self.sigma = np.ones_like(self.sigma)

        return np.divide(X - self.mu, self.sigma)
