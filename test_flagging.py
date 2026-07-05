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

# Inject exactly 20 outliers
n_outliers = 20
outlier_idx = np.random.choice(n, size=n_outliers, replace=False)
X = X_clean.copy()
y = y_clean.copy()
X[outlier_idx, 0] = np.random.normal(100, 5, n_outliers)
X[outlier_idx, 1] = np.random.normal(80, 4, n_outliers)
y[outlier_idx] += np.random.normal(25, 10, n_outliers)

print(f"True outliers: {sorted(outlier_idx)}")

for B in [200, 500, 1000]:
    for frac in [0.2, 0.3, 0.4]:
        engine = AdaptiveRobustDiagnosticEngine(
            coverage=0.5, alpha=0.05, n_bootstrap=B,
            lowess_frac=frac, random_state=42, verbose=False
        )
        results = engine.fit(X, y)
        flagged = set(results['outlier_indices'])
        true_pos = len(flagged & set(outlier_idx))
        false_pos = len(flagged - set(outlier_idx))
        print(f"B={B:4d}, frac={frac:.1f} | Flagged={len(flagged):3d} | True pos={true_pos:2d} | False pos={false_pos:2d}")