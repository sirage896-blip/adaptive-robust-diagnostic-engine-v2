"""
BCa (Bias-Corrected and Accelerated) Bootstrap for WLAD inference.
Implements Stage 4 of the Adaptive Robust Diagnostic Engine.
Based on Efron (1987).
"""
import numpy as np
from scipy import stats
from wlad_solver import WLADSolver

class BCABootstrap:
    """
    Bias-Corrected and Accelerated bootstrap confidence intervals for WLAD.
    """
    def __init__(self, n_bootstrap=1000, alpha=0.05, random_state=42):
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state
        self.coef_intervals_ = None
        self.intercept_interval_ = None
        self.bootstrap_coefs_ = None
        self.z0_ = None
        self.a_ = None

    def fit(self, X, y, wlad_model, weights=None):
        """
        Compute BCa confidence intervals for WLAD coefficients.

        Parameters
        ----------
        X : array, shape (n, p)
            Design matrix
        y : array, shape (n,)
            Response vector
        wlad_model : WLADSolver
            Fitted WLAD model on full data
        weights : array, shape (n,) or None
            Robust weights. If None, all weights = 1.

        Returns
        -------
        self : BCABootstrap
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape

        if weights is None:
            weights = np.ones(n)
        weights = np.asarray(weights, dtype=np.float64)

        rng = np.random.RandomState(self.random_state)

        # Full-data estimates
        beta_full = np.concatenate([[wlad_model.intercept_], wlad_model.coef_])
        p_total = p + 1  # include intercept

        # Center residuals for residual bootstrap
        residuals = wlad_model.residuals_
        centered_residuals = residuals - np.median(residuals)

        # Step 1: Bootstrap replications
        bootstrap_coefs = np.zeros((self.n_bootstrap, p_total))

        for b in range(self.n_bootstrap):
            # Resample residuals with replacement
            boot_residuals = rng.choice(centered_residuals, size=n, replace=True)

            # Construct bootstrap response
            y_boot = wlad_model.fitted_ + boot_residuals

            # Refit WLAD
            try:
                wlad_boot = WLADSolver()
                wlad_boot.fit(X, y_boot, weights=weights, warm_start=beta_full)
                bootstrap_coefs[b, 0] = wlad_boot.intercept_
                bootstrap_coefs[b, 1:] = wlad_boot.coef_
            except Exception:
                # If bootstrap fails, use full-data estimate
                bootstrap_coefs[b] = beta_full

        self.bootstrap_coefs_ = bootstrap_coefs

        # Step 2: Bias-correction factor z0
        self.z0_ = self._compute_z0(bootstrap_coefs, beta_full)

        # Step 3: Acceleration factor a (jackknife)
        self.a_ = self._compute_acceleration(X, y, weights, beta_full)

        # Step 4: BCa interval endpoints
        self.coef_intervals_ = np.zeros((p, 2))
        self.intercept_interval_ = np.zeros(2)

        for j in range(p_total):
            boot_vals = bootstrap_coefs[:, j]
            z0_j = self.z0_[j]
            a_j = self.a_[j]

            # BCa percentile adjustment
            z_alpha = stats.norm.ppf(self.alpha / 2)
            z_1_alpha = stats.norm.ppf(1 - self.alpha / 2)

            # Adjusted percentiles
            alpha1 = stats.norm.cdf(z0_j + (z0_j + z_alpha) / (1 - a_j * (z0_j + z_alpha)))
            alpha2 = stats.norm.cdf(z0_j + (z0_j + z_1_alpha) / (1 - a_j * (z0_j + z_1_alpha)))

            # Clip to valid range
            alpha1 = np.clip(alpha1, 0.001, 0.999)
            alpha2 = np.clip(alpha2, 0.001, 0.999)

            ci_lower = np.percentile(boot_vals, alpha1 * 100)
            ci_upper = np.percentile(boot_vals, alpha2 * 100)

            if j == 0:
                self.intercept_interval_ = np.array([ci_lower, ci_upper])
            else:
                self.coef_intervals_[j-1] = [ci_lower, ci_upper]

        return self

    def _compute_z0(self, bootstrap_coefs, beta_full):
        """Compute bias-correction factor z0 for each parameter."""
        n_boot = bootstrap_coefs.shape[0]
        z0 = np.zeros(len(beta_full))

        for j in range(len(beta_full)):
            # Proportion of bootstrap estimates less than full-data estimate
            prop = np.mean(bootstrap_coefs[:, j] < beta_full[j])
            # Clip to avoid infinite z-scores
            prop = np.clip(prop, 1/n_boot, 1 - 1/n_boot)
            z0[j] = stats.norm.ppf(prop)

        return z0

    def _compute_acceleration(self, X, y, weights, beta_full):
        """
        Compute acceleration factor a via leave-one-out jackknife.
        This measures how rapidly standard error changes as observations are dropped.
        """
        n, p = X.shape
        p_total = p + 1

        # Leave-one-out estimates
        jackknife_coefs = np.zeros((n, p_total))

        for i in range(n):
            X_loo = np.delete(X, i, axis=0)
            y_loo = np.delete(y, i)
            w_loo = np.delete(weights, i)

            try:
                wlad_loo = WLADSolver()
                wlad_loo.fit(X_loo, y_loo, weights=w_loo, warm_start=beta_full)
                jackknife_coefs[i, 0] = wlad_loo.intercept_
                jackknife_coefs[i, 1:] = wlad_loo.coef_
            except Exception:
                jackknife_coefs[i] = beta_full

        # Mean of jackknife replicates
        jack_mean = np.mean(jackknife_coefs, axis=0)

        # Acceleration factor
        a = np.zeros(p_total)
        for j in range(p_total):
            diffs = jack_mean[j] - jackknife_coefs[:, j]
            numerator = np.sum(diffs**3)
            denominator = np.sum(diffs**2)

            if denominator > 1e-10:
                a[j] = numerator / (6 * denominator**1.5)
            else:
                a[j] = 0.0

        return a

    def summary(self, feature_names=None):
        """
        Print summary of BCa confidence intervals.

        Parameters
        ----------
        feature_names : list or None
            Names of predictors
        """
        print("=" * 70)
        print("BCa Bootstrap Confidence Intervals")
        print(f"Bootstrap replications: {self.n_bootstrap}")
        print(f"Confidence level: {100*(1-self.alpha):.1f}%")
        print(f"Bias-correction (median z0): {np.median(np.abs(self.z0_)):.4f}")
        print(f"Acceleration (median |a|): {np.median(np.abs(self.a_)):.4f}")
        print("=" * 70)
        print(f"{'Parameter':<20} {'Estimate':>12} {'CI Lower':>12} {'CI Upper':>12}")
        print("-" * 70)

        # Intercept
        est = self.bootstrap_coefs_[:, 0].mean()
        print(f"{'Intercept':<20} {est:>12.4f} {self.intercept_interval_[0]:>12.4f} {self.intercept_interval_[1]:>12.4f}")

        # Coefficients
        for j in range(self.coef_intervals_.shape[0]):
            name = feature_names[j] if feature_names and j < len(feature_names) else f"X{j+1}"
            est = self.bootstrap_coefs_[:, j+1].mean()
            print(f"{name:<20} {est:>12.4f} {self.coef_intervals_[j, 0]:>12.4f} {self.coef_intervals_[j, 1]:>12.4f}")

        print("=" * 70)
