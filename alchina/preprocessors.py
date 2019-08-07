"""Preprocessors."""

import numpy as np

from typing import Optional

from .exceptions import InvalidInput, NotFitted
from .utils import features_reshape


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


class PCA(object):
    """Principal Component Analysis."""

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components

        self._covariance_matrix = None
        self._U_reduced = None
        self._mean = None

    def fit(self, X):
        """Train the model."""
        X = features_reshape(X)
        max_dimension = min(X.shape)

        if self.n_components is None:
            self.n_components = max_dimension
        elif self.n_components > max_dimension:
            raise InvalidInput(f"n_components must be lesser than {max_dimension}")

        self._covariance_matrix = np.cov(X.T)
        self._mean = np.mean(X, axis=0)

        U, _, _ = np.linalg.svd(self._covariance_matrix)
        self._U_reduced = U[:, : self.n_components]

    def transform(self, X):
        """Transform the input."""
        if self._U_reduced is None or self._mean is None:
            raise NotFitted("the model must be fitted before usage")

        return (X - self._mean).dot(self._U_reduced)

    def fit_transform(self, X):
        """Fit the model and transform the input."""
        self.fit(X)
        return self.transform(X)

    def score_samples(self, X):
        """Compute the log-likelihood of all samples."""
        if self._U_reduced is None or self._mean is None:
            raise NotFitted("the model must be fitted before usage")

        X = features_reshape(X)

        n_features = X.shape[1]
        precision = np.linalg.inv(self._covariance_matrix)
        residuals = X - self._mean

        return -(1 / 2) * (
            -np.log(np.linalg.det(precision))
            + np.sum((residuals * np.dot(residuals, precision)), axis=1)
            + n_features * np.log(2 * np.pi)
        )

    def score(self, X):
        """Compute the mean of log-likelihood of all samples."""
        return np.mean(self.score_samples(X))
