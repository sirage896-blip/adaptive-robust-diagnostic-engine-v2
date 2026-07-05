"""
Demo: Adaptive Robust Diagnostic Engine
Generates synthetic data with outliers and runs the full 4-stage pipeline.
"""
import numpy as np
import matplotlib.pyplot as plt
from engine import AdaptiveRobustDiagnosticEngine

np.random.seed(42)

print("=" * 70)
print("ADAPTIVE ROBUST DIAGNOSTIC ENGINE - DEMO")
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

# Inject 10% leverage + vertical outliers
n_outliers = int(0.10 * n)
outlier_idx = np.random.choice(n, size=n_outliers, replace=False)

X_contaminated = X_clean.copy()
y_contaminated = y_clean.copy()

# Leverage contamination
X_contaminated[outlier_idx, 0] = np.random.normal(100, 5, n_outliers)
X_contaminated[outlier_idx, 1] = np.random.normal(80, 4, n_outliers)

# Vertical contamination
y_contaminated[outlier_idx] += np.random.normal(25, 10, n_outliers)

print(f"Generated data: n={n}, p={p}")
print(f"True parameters: intercept={beta_true[0]:.3f}, b1={beta_true[1]:.3f}, b2={beta_true[2]:.3f}")
print(f"Injected {n_outliers} contaminated observations ({100*n_outliers/n:.0f}%)")
print()

# Run engine
engine = AdaptiveRobustDiagnosticEngine(
    coverage=0.5,
    alpha=0.05,
    n_bootstrap=1000,
    lowess_frac=0.4,
    random_state=42,
    verbose=True
)

results = engine.fit(X_contaminated, y_contaminated)

# Print summary
feature_names = ["X1", "X2"]
engine.summary(feature_names=feature_names)

# Compare with true parameters
print()
print("TRUE vs ESTIMATED PARAMETERS:")
print("-" * 65)
print(f"{'Parameter':<12} {'True':>10} {'WLAD':>10} {'CI Lower':>12} {'CI Upper':>12}")
print(f"{'Intercept':<12} {beta_true[0]:>10.3f} {results['intercept']:>10.3f} "
      f"{results['intercept_interval'][0]:>12.3f} {results['intercept_interval'][1]:>12.3f}")
for j in range(p):
    print(f"{'X'+str(j+1):<12} {beta_true[j+1]:>10.3f} {results['coefficients'][j]:>10.3f} "
          f"{results['confidence_intervals'][j,0]:>12.3f} {results['confidence_intervals'][j,1]:>12.3f}")

# Plot LBEP
fig, ax = engine.plot_lbep(save_path="lbep_demo.png")
plt.show()

print()
print("=" * 70)
print("Demo complete! Plot saved to lbep_demo.png")
print("=" * 70)