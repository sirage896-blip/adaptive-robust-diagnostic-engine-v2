import numpy as np
from engine import AdaptiveRobustDiagnosticEngine

np.random.seed(42)

n = 200
p = 2
mean = [50.0, 30.0]
cov = [[225.0, 45.0], [45.0, 144.0]]
X_clean = np.random.multivariate_normal(mean, cov, size=n)
beta_true = np.array([10.0, 0.5, -0.3])
y_clean = beta_true[0] + X_clean[:, 0]*beta_true[1] + X_clean[:, 1]*beta_true[2] + np.random.normal(0, 5, n)

n_outliers = 20
outlier_idx = np.random.choice(n, size=n_outliers, replace=False)
X = X_clean.copy()
y = y_clean.copy()
X[outlier_idx, 0] = np.random.normal(100, 5, n_outliers)
X[outlier_idx, 1] = np.random.normal(80, 4, n_outliers)
y[outlier_idx] += np.random.normal(25, 10, n_outliers)

engine = AdaptiveRobustDiagnosticEngine(
    coverage=0.5, alpha=0.05, n_bootstrap=200,
    lowess_frac=0.3, random_state=42, verbose=False
)
results = engine.fit(X, y)

# Get the fitted model
wlad = engine.wlad_model_

# Compute MAD on all residuals vs clean-only residuals
residuals = wlad.residuals_
mad_all = np.median(np.abs(residuals - np.median(residuals))) * 1.4826

clean_mask = results['weights'] > 0.5
clean_resid = residuals[clean_mask]
mad_clean = np.median(np.abs(clean_resid - np.median(clean_resid))) * 1.4826

std_all = np.abs(residuals) / mad_all
std_clean = np.abs(residuals) / mad_clean

print("=" * 60)
print("MAD DEBUG")
print("=" * 60)
print(f"MAD (all residuals):     {mad_all:.4f}")
print(f"MAD (clean only):        {mad_clean:.4f}")
print(f"Ratio (all/clean):       {mad_all/mad_clean:.4f}")
print()
print(f"Standardized residuals (all MAD):")
print(f"  Clean median: {np.median(std_all[clean_mask]):.4f}")
print(f"  Clean 97.5th: {np.percentile(std_all[clean_mask], 97.5):.4f}")
print(f"  Outlier min:  {std_all[outlier_idx].min():.4f}")
print()
print(f"Standardized residuals (clean MAD):")
print(f"  Clean median: {np.median(std_clean[clean_mask]):.4f}")
print(f"  Clean 97.5th: {np.percentile(std_clean[clean_mask], 97.5):.4f}")
print(f"  Outlier min:  {std_clean[outlier_idx].min():.4f}")