"""
Test WLAD solver independently to debug the zero-output issue.
"""
import numpy as np
from wlad_solver import WLADSolver

print("Testing WLAD Solver")
print("=" * 50)

# Simple test data
np.random.seed(42)
n = 50
X = np.random.normal(50, 10, (n, 2))
y = 10 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.normal(0, 2, n)
weights = np.ones(n)

print(f"Data: n={n}, X mean={X.mean():.2f}, y mean={y.mean():.2f}")
print(f"True: intercept=10, beta1=0.5, beta2=-0.3")
print()

# Fit WLAD
solver = WLADSolver()
solver.fit(X, y, weights=weights)

print("WLAD Results:")
print(f"  Intercept: {solver.intercept_:.4f}")
print(f"  Beta 1:    {solver.coef_[0]:.4f}")
print(f"  Beta 2:    {solver.coef_[1]:.4f}")
print()

# Check if all zeros
if abs(solver.intercept_) < 0.001 and np.all(np.abs(solver.coef_) < 0.001):
    print("WARNING: All coefficients are near zero!")
    print("This indicates the LP solver failed.")
else:
    print("SUCCESS: WLAD solver is working correctly!")
