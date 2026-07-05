"""
Debug the LBEP bootstrap envelope computation.
"""
import numpy as np
from engine import AdaptiveRobustDiagnosticEngine

np.random.seed(42)

# Same data as before
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

# Access internal diagnostics
diag = engine.diagnostics_

# Check bootstrap envelope stats
boot_upper = diag.get('bootstrap_upper', None)
boot_lower = diag.get('bootstrap_lower', None)
s = diag.get('standardized_residuals', None)

print("=" * 60)
print("LBEP DEBUG OUTPUT")
print("=" * 60)
print(f"Number of observations: {n}")
print(f"Clean subset size (w > 0.5): {len(diag.get('clean_indices', []))}")
print(f"Standardized residuals (s_i) stats:")
print(f"  Min:  {s.min():.4f}")
print(f"  Med:  {np.median(s):.4f}")
print(f"  Max:  {s.max():.4f}")
print(f"  Mean: {s.mean():.4f}")
print(f"  Std:  {s.std():.4f}")
print()

print(f"Bootstrap UPPER envelope (97.5th pct) stats:")
print(f"  Min:  {boot_upper.min():.4f}")
print(f"  Med:  {np.median(boot_upper):.4f}")
print(f"  Max:  {boot_upper.max():.4f}")
print(f"  Mean: {boot_upper.mean():.4f}")
print()

print(f"Bootstrap LOWER envelope (2.5th pct) stats:")
print(f"  Min:  {boot_lower.min():.4f}")
print(f"  Med:  {np.median(boot_lower):.4f}")
print(f"  Max:  {boot_lower.max():.4f}")
print(f"  Mean: {boot_lower.mean():.4f}")
print()

# Compare a few specific observations
print("Sample observations (first 10 clean, first 5 outliers):")
clean_mask = np.ones(n, dtype=bool)
clean_mask[outlier_idx] = False
clean_idx = np.where(clean_mask)[0][:10]
dirty_idx = outlier_idx[:5]

print(f"\n{'Idx':>5} {'Type':>8} {'s_i':>8} {'Upper':>8} {'Lower':>8} {'Flagged?':>10}")
print("-" * 55)
for i in list(clean_idx) + list(dirty_idx):
    typ = "CLEAN" if i not in outlier_idx else "OUTLIER"
    flagged = "YES" if s[i] > boot_upper[i] else "NO"
    print(f"{i:5d} {typ:>8} {s[i]:8.3f} {boot_upper[i]:8.3f} {boot_lower[i]:8.3f} {flagged:>10}")