with open('diagnostics.py', 'w') as f:
    f.write('''"""
LOWESS-Optimized Bootstrap Envelope Plot (LBEP) diagnostics.
Implements Stage 3 of the Adaptive Robust Diagnostic Engine.
"""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


class LBEPDiagnostic:
    """
    LOWESS-Optimized Bootstrap Envelope Plot for L1 regression residuals.
    """
    def __init__(self, n_bootstrap=1000, frac=0.3, alpha=0.05, random_state=42):
        self.n_bootstrap = n_bootstrap
        self.frac = frac
        self.alpha = alpha
        self.random_state = random_state
        self.lower_envelope_ = None
        self.upper_envelope_ = None
        self.fitted_values_ = None
        self.standardized_residuals_ = None
        self.outlier_indices_ = None

    def fit(self, X, y, wlad_model, weights, clean_threshold=0.5):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape
        rng = np.random.RandomState(self.random_state)

        fitted = wlad_model.fitted_
        residuals = wlad_model.residuals_

        mad = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
        if mad < 1e-10:
            mad = 1e-10
        standardized = np.abs(residuals) / mad

        self.fitted_values_ = fitted
        self.standardized_residuals_ = standardized

        clean_mask = weights > clean_threshold
        clean_indices = np.where(clean_mask)[0]
        n_clean = len(clean_indices)

        if n_clean < p + 5:
            clean_mask = weights > np.min(weights)
            clean_indices = np.where(clean_mask)[0]
            n_clean = len(clean_indices)

        sort_idx = np.argsort(fitted)
        fitted_sorted = fitted[sort_idx]

        bootstrap_std_residuals = np.zeros((self.n_bootstrap, n))

        from wlad_solver import WLADSolver

        for b in range(self.n_bootstrap):
            boot_idx = rng.choice(clean_indices, size=n_clean, replace=True)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]
            w_boot = weights[boot_idx]

            try:
                wlad_boot = WLADSolver()
                wlad_boot.fit(X_boot, y_boot, weights=w_boot,
                              warm_start=np.concatenate([[wlad_model.intercept_], wlad_model.coef_]))

                fitted_boot = wlad_boot.predict(X)
                resid_boot = y - fitted_boot
                
                # FIX: Compute MAD from bootstrap sample ONLY
                resid_boot_clean = resid_boot[boot_idx]
                mad_boot = np.median(np.abs(resid_boot_clean - np.median(resid_boot_clean))) * 1.4826
                if mad_boot < 1e-10:
                    mad_boot = 1e-10
                bootstrap_std_residuals[b] = np.abs(resid_boot) / mad_boot
            except Exception:
                bootstrap_std_residuals[b] = standardized

        lower_pct = self.alpha / 2 * 100
        upper_pct = (1 - self.alpha / 2) * 100

        raw_lower = np.percentile(bootstrap_std_residuals, lower_pct, axis=0)
        raw_upper = np.percentile(bootstrap_std_residuals, upper_pct, axis=0)

        raw_lower_sorted = raw_lower[sort_idx]
        raw_upper_sorted = raw_upper[sort_idx]

        lower_smooth_sorted = lowess(raw_lower_sorted, fitted_sorted, frac=self.frac, return_sorted=False)
        upper_smooth_sorted = lowess(raw_upper_sorted, fitted_sorted, frac=self.frac, return_sorted=False)

        lower_smooth = np.empty_like(lower_smooth_sorted)
        upper_smooth = np.empty_like(upper_smooth_sorted)
        lower_smooth[sort_idx] = lower_smooth_sorted
        upper_smooth[sort_idx] = upper_smooth_sorted

        self.lower_envelope_ = lower_smooth
        self.upper_envelope_ = upper_smooth

        self.outlier_indices_ = np.where(standardized > upper_smooth)[0]

        return self

    def plot(self, figsize=(10, 6), title="LBEP: LOWESS-Optimized Bootstrap Envelope Plot", save_path=None):
        fig, ax = plt.subplots(figsize=figsize)

        fitted = self.fitted_values_
        std_resid = self.standardized_residuals_

        sort_idx = np.argsort(fitted)
        fitted_sorted = fitted[sort_idx]
        lower_sorted = self.lower_envelope_[sort_idx]
        upper_sorted = self.upper_envelope_[sort_idx]

        outlier_mask = np.zeros(len(fitted), dtype=bool)
        if self.outlier_indices_ is not None and len(self.outlier_indices_) > 0:
            outlier_mask[self.outlier_indices_] = True

        ax.scatter(fitted[~outlier_mask], std_resid[~outlier_mask],
                   c="steelblue", alpha=0.6, s=50, edgecolors="navy", linewidth=0.5, label="Observations", zorder=3)

        if np.any(outlier_mask):
            ax.scatter(fitted[outlier_mask], std_resid[outlier_mask],
                       c="red", alpha=0.9, s=100, marker="X", edgecolors="darkred", linewidth=1.5,
                       label=f"Flagged Outliers (n={np.sum(outlier_mask)})", zorder=4)

        ax.plot(fitted_sorted, upper_sorted, "b-", linewidth=2.5,
                label=f"{100*(1-self.alpha/2):.1f}th Percentile Envelope (LOWESS)", zorder=2)
        ax.plot(fitted_sorted, lower_sorted, "gold", linestyle="--", linewidth=1.5,
                label=f"{100*self.alpha/2:.1f}th Percentile Envelope", zorder=2)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

        ax.set_xlabel("Fitted Values", fontsize=12)
        ax.set_ylabel("Standardized Absolute Residuals (|r|/MAD)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        ax.text(0.98, 0.02, f"LOWESS span = {self.frac}",
                transform=ax.transAxes, ha="right", va="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax
''')
print("diagnostics.py written successfully")