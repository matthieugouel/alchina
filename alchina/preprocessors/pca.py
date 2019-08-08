"""Principal Component Analysis."""

import numpy as np

from typing import Optional

from alchina.exceptions import InvalidInput, NotFitted
from alchina.utils import features_reshape


class PCA(object):
    """Principal Component Analysis algorithm."""

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
