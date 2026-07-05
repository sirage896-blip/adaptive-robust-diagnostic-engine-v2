"""
Adaptive weight-floor selection based on the eigenstructure of the MCD scatter matrix.
Implements Equation (3.12) from the thesis:
    δ* = clip( (p/n) * (λ_min / λ_max), [10^-4, 0.10] )
"""
import numpy as np
from scipy import stats

def compute_adaptive_delta(mcd_covariance, n, p, delta_min=1e-4, delta_max=0.10):
    """
    Compute the adaptive weight-floor selector δ*.

    Parameters
    ----------
    mcd_covariance : array, shape (p, p)
        Robust covariance matrix from Fast-MCD
    n : int
        Sample size
    p : int
        Number of predictors
    delta_min : float
        Minimum floor (default: 1e-4)
    delta_max : float
        Maximum floor (default: 0.10)

    Returns
    -------
    delta_star : float
        Adaptive weight floor
    conditioning_ratio : float
        λ_min / λ_max (for diagnostics)
    """
    eigvals = np.linalg.eigvalsh(mcd_covariance)
    eigvals = np.sort(eigvals)
    lambda_min = eigvals[0]
    lambda_max = eigvals[-1]

    # Conditioning ratio
    rho = lambda_min / lambda_max if lambda_max > 0 else 1e-8

    # Equation (3.12): δ* = (p/n) * ρ
    delta_star = (p / n) * rho

    # Clip to [delta_min, delta_max]
    delta_star = np.clip(delta_star, delta_min, delta_max)

    return delta_star, rho


def compute_mcd_weights(X, mcd_location, mcd_covariance, delta_star=None, 
                         quantile=0.975, return_distances=False):
    """
    Compute continuous distance-based weights using robust Mahalanobis distances.

    Parameters
    ----------
    X : array, shape (n, p)
        Design matrix
    mcd_location : array, shape (p,)
        Robust location vector
    mcd_covariance : array, shape (p, p)
        Robust covariance matrix
    delta_star : float or None
        Adaptive weight floor. If None, computed automatically.
    quantile : float
        Chi-squared quantile for cutoff (default: 0.975)
    return_distances : bool
        If True, also return robust distances

    Returns
    -------
    weights : array, shape (n,)
        Robust weights in [delta_star, 1]
    rd : array, shape (n,)  [optional]
        Robust Mahalanobis distances
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape

    # Compute robust Mahalanobis distances
    diff = X - mcd_location
    try:
        precision = np.linalg.inv(mcd_covariance)
        rd_sq = np.sum(diff @ precision * diff, axis=1)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(mcd_covariance)
        rd_sq = np.sum(diff @ precision * diff, axis=1)

    rd = np.sqrt(rd_sq)

    # Chi-squared cutoff
    chi2_cutoff = stats.chi2.ppf(quantile, df=p)

    # Weight function: w_i = min(1, chi2 / RD_i^2)
    weights = np.minimum(1.0, chi2_cutoff / (rd_sq + 1e-12))

    # Apply adaptive floor
    if delta_star is None:
        delta_star, _ = compute_adaptive_delta(mcd_covariance, n, p)

    weights = np.maximum(weights, delta_star)

    if return_distances:
        return weights, rd
    return weights
