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

print("Results keys:", list(results.keys()))

# If there's envelope data, print it
if 'diagnostics' in results:
    diag = results['diagnostics']
    print("Diagnostics keys:", list(diag.keys()))
    for k, v in diag.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
        else:
            print(f"{k}: {v}")