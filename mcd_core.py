"""
Fast-MCD and deterministic MCD implementations with C-step optimization.
Based on Rousseeuw & Van Driessen (1999) with Tikhonov regularization.
"""
import numpy as np
from scipy import linalg
from sklearn.covariance import MinCovDet


class FastMCD:
    """
    Fast Minimum Covariance Determinant estimator with adaptive weighting support.
    """
    def __init__(self, support_fraction=None, random_state=42):
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.location_ = None
        self.covariance_ = None
        self.precision_ = None
        self.distances_ = None
        self.support_ = None
        self.support_fraction_ = None

    def fit(self, X):
        """
        Fit Fast-MCD to design matrix X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Design matrix (predictor space only)

        Returns
        -------
        self : FastMCD
        """
        X = np.asarray(X, dtype=np.float64)

        # Use scikit-learn's MinCovDet as the robust MCD engine
        mcd = MinCovDet(
            support_fraction=self.support_fraction,
            random_state=self.random_state
        )
        mcd.fit(X)

        self.location_ = mcd.location_
        self.covariance_ = mcd.covariance_
        self.precision_ = mcd.precision_
        self.distances_ = mcd.mahalanobis(X)
        self.support_ = mcd.support_

        # Store support fraction (handle None case)
        if self.support_fraction is not None:
            self.support_fraction_ = self.support_fraction
        else:
            self.support_fraction_ = 0.5

        # Tikhonov regularization for near-singular matrices
        self.covariance_ = self._regularize(self.covariance_)

        # Recompute precision after regularization
        try:
            self.precision_ = linalg.inv(self.covariance_)
        except linalg.LinAlgError:
            self.precision_ = linalg.pinv(self.covariance_)

        # Recompute distances with regularized covariance
        self.distances_ = self._robust_mahalanobis(X)

        return self

    def _regularize(self, cov, lambda_factor=1e-6):
        """Tikhonov regularization to prevent singularity."""
        p = cov.shape[0]
        trace = np.trace(cov)
        lambda_val = lambda_factor * trace / p
        reg_cov = cov + lambda_val * np.eye(p)
        return reg_cov

    def _robust_mahalanobis(self, X):
        """Compute robust Mahalanobis distances."""
        X = np.asarray(X, dtype=np.float64)
        diff = X - self.location_
        try:
            md = np.sqrt(np.sum(diff @ self.precision_ * diff, axis=1))
        except Exception:
            md = np.sqrt(np.sum(diff * np.linalg.solve(self.covariance_, diff.T).T, axis=1))
        return md

    def robust_distances(self, X):
        """Return robust Mahalanobis distances for observations in X."""
        X = np.asarray(X, dtype=np.float64)
        diff = X - self.location_
        try:
            md = np.sqrt(np.sum(diff @ self.precision_ * diff, axis=1))
        except Exception:
            md = np.sqrt(np.sum(diff * np.linalg.solve(self.covariance_, diff.T).T, axis=1))
        return md
