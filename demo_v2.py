"""
Adaptive Robust Diagnostic Engine - Demo v2
Properly scaled data to avoid numerical issues in LP solver.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for clean output
import matplotlib.pyplot as plt
from engine import AdaptiveRobustDiagnosticEngine

np.random.seed(42)

print("=" * 70)
print("ADAPTIVE ROBUST DIAGNOSTIC ENGINE - DEMO v2")
print("=" * 70)
print()

# Generate realistic synthetic data
n = 200
p = 2

# Clean predictors with good scale
mean = [50.0, 30.0]
cov = [[225.0, 45.0], [45.0, 144.0]]
X_clean = np.random.multivariate_normal(mean, cov, size=n)

# True parameters
beta_true = np.array([10.0, 0.5, -0.3])

# Clean response
y_clean = beta_true[0] + X_clean[:, 0] * beta_true[1] + X_clean[:, 1] * beta_true[2]
y_clean += np.random.normal(0, 5, n)

# Inject 10% outliers
n_outliers = int(0.10 * n)
outlier_idx = np.random.choice(n, size=n_outliers, replace=False)

X_contaminated = X_clean.copy()
y_contaminated = y_clean.copy()

X_contaminated[outlier_idx, 0] = np.random.normal(100, 5, n_outliers)
X_contaminated[outlier_idx, 1] = np.random.normal(80, 4, n_outliers)
y_contaminated[outlier_idx] += np.random.normal(25, 10, n_outliers)

print(f"Data: n={n}, p={p}")
print(f"True: intercept={beta_true[0]:.2f}, beta1={beta_true[1]:.2f}, beta2={beta_true[2]:.2f}")
print(f"Outliers: {n_outliers} ({100*n_outliers/n:.0f}%)")
print(f"X1 range: [{X_contaminated[:,0].min():.1f}, {X_contaminated[:,0].max():.1f}]")
print(f"X2 range: [{X_contaminated[:,1].min():.1f}, {X_contaminated[:,1].max():.1f}]")
print(f"y range: [{y_contaminated.min():.1f}, {y_contaminated.max():.1f}]")
print()

# Run engine
engine = AdaptiveRobustDiagnosticEngine(
    coverage=0.5,
    alpha=0.05,
    n_bootstrap=200,
    lowess_frac=0.3,
    random_state=42,
    verbose=True
)

results = engine.fit(X_contaminated, y_contaminated)

# Summary
feature_names = ["X1", "X2"]
engine.summary(feature_names=feature_names)

# Comparison
print()
print("TRUE vs ESTIMATED:")
print("-" * 60)
print(f"{'Param':<10} {'True':>8} {'WLAD':>8} {'CI Low':>10} {'CI High':>10}")
print(f"{'Intercept':<10} {beta_true[0]:>8.3f} {results['intercept']:>8.3f} "
      f"{results['intercept_interval'][0]:>10.3f} {results['intercept_interval'][1]:>10.3f}")
for j in range(p):
    print(f"{'X'+str(j+1):<10} {beta_true[j+1]:>8.3f} {results['coefficients'][j]:>8.3f} "
          f"{results['confidence_intervals'][j,0]:>10.3f} {results['confidence_intervals'][j,1]:>10.3f}")

# Save plot instead of showing
fig, ax = engine.plot_lbep(save_path="lbep_v2.png")
plt.close(fig)

print()
print("Plot saved to: lbep_v2.png")
print("=" * 70)
