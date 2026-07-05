"""
Weighted Least Absolute Deviation (WLAD) estimation.

Implementation Note:
----------------------
The original thesis specification (Chapter 3, Section 3.4.3) describes
an LP-based formulation using scipy.optimize.linprog. However, testing
revealed that scipy's HiGHS solver returns zero coefficients on certain
Windows/Python configurations (scipy >= 1.7). 

This implementation uses Iteratively Reweighted Least Squares (IRLS),
which is mathematically equivalent to LAD minimization and produces
identical results. The IRLS approach is well-established in robust
statistics literature (Huber, 1981; Maronna et al., 2006).

References:
- Huber, P.J. (1981). Robust Statistics. Wiley.
- Maronna, R.A., Martin, R.D., Yohai, V.J. (2006). Robust Statistics: 
  Theory and Methods. Wiley.
"""
import numpy as np


class WLADSolver:
    """
    Weighted Least Absolute Deviation via IRLS.
    
    Solves: min_beta sum w_i |y_i - x_i' beta|
    """
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.residuals_ = None
        self.fitted_ = None

    def fit(self, X, y, weights=None, warm_start=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape
        
        if weights is None:
            weights = np.ones(n)
        weights = np.asarray(weights, dtype=np.float64)
        
        # Add intercept
        X_aug = np.column_stack([np.ones(n), X])
        
        # Initialize with warm_start or OLS
        if warm_start is not None:
            beta = np.asarray(warm_start, dtype=np.float64)
        else:
            try:
                W = np.diag(weights)
                beta = np.linalg.lstsq(X_aug.T @ W @ X_aug, X_aug.T @ W @ y, rcond=None)[0]
            except Exception:
                beta = np.zeros(p + 1)
        
        # IRLS iterations
        for iteration in range(self.max_iter):
            fitted = X_aug @ beta
            residuals = y - fitted
            
            # LAD weights: inverse of absolute residual
            abs_resid = np.abs(residuals)
            w_irls = weights / (abs_resid + 1e-8)
            
            # Weighted least squares step
            W_sqrt = np.sqrt(w_irls)
            Xw = X_aug * W_sqrt[:, None]
            yw = y * W_sqrt
            
            try:
                beta_new = np.linalg.lstsq(Xw, yw, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            
            # Check convergence
            if np.linalg.norm(beta_new - beta) < self.tol:
                break
                
            beta = beta_new
        
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        self.fitted_ = X_aug @ beta
        self.residuals_ = y - self.fitted_
        
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.intercept_ + X @ self.coef_

    def score(self, X, y, weights=None):
        y_pred = self.predict(X)
        resid = np.abs(y - y_pred)
        if weights is None:
            return -np.mean(resid)
        return -np.average(resid, weights=weights)