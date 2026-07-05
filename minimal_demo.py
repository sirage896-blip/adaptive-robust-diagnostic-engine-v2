"""
Minimal working example of the Adaptive Robust Diagnostic Engine.
"""
import numpy as np
from engine import AdaptiveRobustDiagnosticEngine

np.random.seed(42)

# Very simple data with clear scale
n = 100
X = np.column_stack([
    np.random.normal(100, 20, n),
    np.random.normal(50, 15, n)
])
y = 20 + 2.0 * X[:, 0] - 1.5 * X[:, 1] + np.random.normal(0, 10, n)

# Add a few outliers
X[:5, 0] = 200
y[:5] += 100

print("Data summary:")
print(f"  X shape: {X.shape}")
print(f"  y mean: {y.mean():.2f}, std: {y.std():.2f}")
print(f"  X1 mean: {X[:,0].mean():.2f}, std: {X[:,0].std():.2f}")
print(f"  X2 mean: {X[:,1].mean():.2f}, std: {X[:,1].std():.2f}")
print()

engine = AdaptiveRobustDiagnosticEngine(
    coverage=0.5,
    alpha=0.05,
    n_bootstrap=100,
    lowess_frac=0.3,
    random_state=42,
    verbose=True
)

results = engine.fit(X, y)

print()
print("RESULTS:")
print(f"  Intercept: {results['intercept']:.4f}")
print(f"  Beta 1:    {results['coefficients'][0]:.4f}")
print(f"  Beta 2:    {results['coefficients'][1]:.4f}")
print(f"  Delta*:    {results['adaptive_delta']:.6f}")
print(f"  Outliers:  {len(results['outlier_indices'])}")
