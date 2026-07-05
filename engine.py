"""
Adaptive Robust Diagnostic Engine
=================================
Unified four-stage pipeline:
  Stage 1: Fast-MCD robust leverage detection with adaptive weighting
  Stage 2: WLAD robust estimation
  Stage 3: LBEP diagnostic visualization
  Stage 4: BCa bootstrap inference

Based on:
  Kandeh, M. (2027). Closing the Finite-Sample Robustness Gap:
  An Adaptive Diagnostic Engine for Simultaneous Leverage and
  Heteroscedasticity in Linear Regression.
  MSc Thesis, University of The Gambia.
"""
import numpy as np
import warnings

from mcd_core import FastMCD
from adaptive_weights import compute_adaptive_delta, compute_mcd_weights
from wlad_solver import WLADSolver
from diagnostics import LBEPDiagnostic
from bca_bootstrap import BCABootstrap


class AdaptiveRobustDiagnosticEngine:
    """
    Four-stage Adaptive Robust Diagnostic Engine for linear regression
    under simultaneous leverage contamination and heteroscedasticity.

    Parameters
    ----------
    coverage : float
        Support fraction for Fast-MCD (default: 0.5 for 50% breakdown)
    alpha : float
        Significance level for confidence intervals (default: 0.05)
    n_bootstrap : int
        Number of bootstrap replications for LBEP and BCa (default: 1000)
    lowess_frac : float
        LOWESS span parameter for envelope smoothing (default: 0.3)
    random_state : int
        Random seed for reproducibility (default: 42)
    verbose : bool
        Print progress messages (default: True)
    """

    def __init__(self, coverage=0.5, alpha=0.05, n_bootstrap=1000,
                 lowess_frac=0.3, random_state=42, verbose=True):
        self.coverage = coverage
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.lowess_frac = lowess_frac
        self.random_state = random_state
        self.verbose = verbose

        # Results storage
        self.mcd_model_ = None
        self.wlad_model_ = None
        self.lbep_diagnostic_ = None
        self.bca_bootstrap_ = None

        self.weights_ = None
        self.delta_star_ = None
        self.conditioning_ratio_ = None
        self.robust_distances_ = None
        self.outlier_indices_ = None

    def fit(self, X, y):
        """
        Run the complete four-stage pipeline.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Design matrix (predictors)
        y : array-like, shape (n_samples,)
            Response vector

        Returns
        -------
        results : dict
            Dictionary containing all engine outputs
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape

        if self.verbose:
            print("=" * 60)
            print("ADAPTIVE ROBUST DIAGNOSTIC ENGINE")
            print("=" * 60)
            print(f"Sample size: n = {n}, Predictors: p = {p}")
            print()

        # Stage 1: Robust Leverage Detection (Fast-MCD)
        if self.verbose:
            print("[STAGE 1] Robust Leverage Detection (Fast-MCD)...")

        self.mcd_model_ = FastMCD(
            support_fraction=self.coverage,
            random_state=self.random_state
        )
        self.mcd_model_.fit(X)

        # Compute adaptive weight floor delta*
        self.delta_star_, self.conditioning_ratio_ = compute_adaptive_delta(
            self.mcd_model_.covariance_, n, p
        )

        if self.verbose:
            print(f"  Adaptive weight floor delta* = {self.delta_star_:.6f}")
            print(f"  Conditioning ratio rho = {self.conditioning_ratio_:.6f}")

        # Compute robust weights
        self.weights_, self.robust_distances_ = compute_mcd_weights(
            X,
            self.mcd_model_.location_,
            self.mcd_model_.covariance_,
            delta_star=self.delta_star_,
            return_distances=True
        )

        n_downweighted = np.sum(self.weights_ < 0.5)
        if self.verbose:
            print(f"  Observations with w < 0.5: {n_downweighted} ({100*n_downweighted/n:.1f}%)")
            print(f"  Observations at floor delta*: {np.sum(self.weights_ <= self.delta_star_ + 1e-10)}")
            print("  [STAGE 1] Complete.")
            print()

        # Stage 2: Robust Estimation (WLAD)
        if self.verbose:
            print("[STAGE 2] WLAD Robust Estimation...")

        # Warm start with OLS for faster convergence
        try:
            from sklearn.linear_model import LinearRegression
            ols = LinearRegression().fit(X, y)
            warm_start = np.concatenate([[ols.intercept_], ols.coef_])
        except Exception:
            warm_start = None

        self.wlad_model_ = WLADSolver()
        self.wlad_model_.fit(X, y, weights=self.weights_, warm_start=warm_start)

        if self.verbose:
            print(f"  Intercept: {self.wlad_model_.intercept_:.4f}")
            for j, coef in enumerate(self.wlad_model_.coef_):
                print(f"  beta{j+1}: {coef:.4f}")
            print("  [STAGE 2] Complete.")
            print()

        # Stage 3: LBEP Diagnostic Visualization
        if self.verbose:
            print("[STAGE 3] LBEP Diagnostic Visualization...")

        self.lbep_diagnostic_ = LBEPDiagnostic(
            n_bootstrap=self.n_bootstrap,
            frac=self.lowess_frac,
            alpha=self.alpha,
            random_state=self.random_state
        )
        self.lbep_diagnostic_.fit(X, y, self.wlad_model_, self.weights_)

        self.outlier_indices_ = self.lbep_diagnostic_.outlier_indices_

        if self.verbose:
            print(f"  Flagged outliers: {len(self.outlier_indices_)}")
            if len(self.outlier_indices_) > 0:
                print(f"  Outlier indices: {self.outlier_indices_}")
            print("  [STAGE 3] Complete.")
            print()

        # Stage 4: BCa Bootstrap Inference
        if self.verbose:
            print("[STAGE 4] BCa Bootstrap Inference...")
            print(f"  Bootstrap replications: B = {self.n_bootstrap}")
            print(f"  Jackknife acceleration: n = {n} leave-one-out fits")
            print("  (This may take a few minutes...)")

        self.bca_bootstrap_ = BCABootstrap(
            n_bootstrap=self.n_bootstrap,
            alpha=self.alpha,
            random_state=self.random_state
        )
        self.bca_bootstrap_.fit(X, y, self.wlad_model_, self.weights_)

        if self.verbose:
            print("  [STAGE 4] Complete.")
            print()
            print("=" * 60)
            print("ENGINE EXECUTION COMPLETE")
            print("=" * 60)
            print()

        # Compile results
        results = {
            'coefficients': self.wlad_model_.coef_,
            'intercept': self.wlad_model_.intercept_,
            'confidence_intervals': self.bca_bootstrap_.coef_intervals_,
            'intercept_interval': self.bca_bootstrap_.intercept_interval_,
            'outlier_indices': self.outlier_indices_,
            'adaptive_delta': self.delta_star_,
            'conditioning_ratio': self.conditioning_ratio_,
            'weights': self.weights_,
            'robust_distances': self.robust_distances_,
            'fitted_values': self.wlad_model_.fitted_,
            'residuals': self.wlad_model_.residuals_,
            'bca_z0': self.bca_bootstrap_.z0_,
            'bca_a': self.bca_bootstrap_.a_,
            'bootstrap_coefs': self.bca_bootstrap_.bootstrap_coefs_
        }

        return results

    def plot_lbep(self, save_path=None):
        """
        Generate and display the LBEP diagnostic plot.

        Parameters
        ----------
        save_path : str or None
            If provided, save figure to this path

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        if self.lbep_diagnostic_ is None:
            raise RuntimeError("Engine has not been fitted yet. Call fit() first.")

        return self.lbep_diagnostic_.plot(save_path=save_path)

    def summary(self, feature_names=None):
        """
        Print comprehensive summary of engine results.

        Parameters
        ----------
        feature_names : list or None
            Names of predictor variables
        """
        if self.wlad_model_ is None:
            raise RuntimeError("Engine has not been fitted yet. Call fit() first.")

        print("=" * 70)
        print("ADAPTIVE ROBUST DIAGNOSTIC ENGINE - RESULTS SUMMARY")
        print("=" * 70)
        print()
        print(f"Adaptive weight floor delta*: {self.delta_star_:.6f}")
        print(f"Conditioning ratio rho: {self.conditioning_ratio_:.6f}")
        print(f"Observations downweighted (w < 0.5): {np.sum(self.weights_ < 0.5)}")
        print(f"Observations at floor: {np.sum(self.weights_ <= self.delta_star_ + 1e-10)}")
        print()

        self.bca_bootstrap_.summary(feature_names=feature_names)

        print()
        print(f"Flagged outliers: {len(self.outlier_indices_)}")
        if len(self.outlier_indices_) > 0:
            print(f"Outlier indices: {self.outlier_indices_}")
        print("=" * 70)


if __name__ == "__main__":
    print("Adaptive Robust Diagnostic Engine")
    print("=================================")
    print()
    print("This module provides the AdaptiveRobustDiagnosticEngine class.")
    print("Import it in your script:")
    print()
    print("  from engine import AdaptiveRobustDiagnosticEngine")
    print()
    print("Example usage:")
    print("  engine = AdaptiveRobustDiagnosticEngine(n_bootstrap=1000)")
    print("  results = engine.fit(X, y)")
    print("  engine.summary()")
    print("  fig, ax = engine.plot_lbep()")
