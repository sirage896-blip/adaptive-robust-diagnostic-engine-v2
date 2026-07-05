"""
Improved Demo: Adaptive Robust Diagnostic Engine
Generates more realistic data with proper scale.
"""
import numpy as np
import matplotlib.pyplot as plt
from engine import AdaptiveRobustDiagnosticEngine

# Set random seed
np.random.seed(42)

print("=" * 70)
print("ADAPTIVE ROBUST DIAGNOSTIC ENGINE - IMPROVED DEMO")
print("=" * 70)
print()

# Generate more realistic synthetic data
n = 200
p = 2

# Clean predictors with reasonable scale
X_clean = np.random.multivariate_normal(
    mean=[50, 30],
    cov=[[100, 20], [20, 64]],
    size=n
)

# True parameters
beta_true = np.array([10.0, 0.5, -0.3])  # intercept + 2 slopes

# Clean response with moderate noise
y_clean = beta_true[0] + X_clean @ beta_true[1:] + np.random.normal(0, 5, n)

# Inject 10% leverage outliers
n_outliers = int(0.10 * n)
outlier_idx = np.random.choice(n, size=n_outliers, replace=False)
X_contaminated = X_clean.copy()
y_contaminated = y_clean.copy()

# Leverage contamination: extreme values in predictor space
X_contaminated[outlier_idx] = np.random.multivariate_normal(
    mean=[100, 80], cov=[[25, 0], [0, 16]], size=n_outliers
)
# Vertical contamination: large response errors
y_contaminated[outlier_idx] += np.random.normal(15, 10, n_outliers)

print(f"Generated data: n={n}, p={p}")
print(f"True parameters: intercept={beta_true[0]}, β1={beta_true[1]}, β2={beta_true[2]}")
print(f"Injected {n_outliers} contaminated observations ({100*n_outliers/n:.0f}%)")
print()

# Run the engine with fewer bootstrap replications for speed
engine = AdaptiveRobustDiagnosticEngine(
    coverage=0.5,
    alpha=0.05,
    n_bootstrap=200,  # Reduced for demo speed
    lowess_frac=0.3,
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
print("-" * 60)
print(f"{'Parameter':<15} {'True':>10} {'WLAD':>10} {'CI Lower':>12} {'CI Upper':>12}")
print(f"{'Intercept':<15} {beta_true[0]:>10.3f} {results['intercept']:>10.3f} "
      f"{results['intercept_interval'][0]:>12.3f} {results['intercept_interval'][1]:>12.3f}")
for j in range(p):
    print(f"{'X'+str(j+1):<15} {beta_true[j+1]:>10.3f} {results['coefficients'][j]:>10.3f} "
          f"{results['confidence_intervals'][j,0]:>12.3f} {results['confidence_intervals'][j,1]:>12.3f}")

# Plot LBEP
fig, ax = engine.plot_lbep(save_path="lbep_demo.png")
plt.show()

print()
print("Demo complete! Plot saved to lbep_demo.png")
print()
print("Key Results:")
print(f"  - Adaptive δ*: {results['adaptive_delta']:.6f}")
print(f"  - Conditioning ratio: {results['conditioning_ratio']:.6f}")
print(f"  - Downweighted observations: {np.sum(results['weights'] < 0.5)}")
print(f"  - Flagged outliers: {len(results['outlier_indices'])}")
