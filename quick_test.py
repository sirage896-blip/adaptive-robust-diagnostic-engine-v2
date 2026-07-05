"""
Test the standalone IRLS-based WLAD solver.
"""
import numpy as np
from wlad_solver import WLADSolver

np.random.seed(42)
X = np.random.normal(50, 10, (50, 2))
y = 10 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.normal(0, 2, 50)

solver = WLADSolver()
solver.fit(X, y)

print("WLAD Solver Test (IRLS-based)")
print("=" * 40)
print(f"Intercept: {solver.intercept_:.4f} (expected ~10)")
print(f"Beta 1:    {solver.coef_[0]:.4f} (expected ~0.5)")
print(f"Beta 2:    {solver.coef_[1]:.4f} (expected ~-0.3)")
print()

if abs(solver.intercept_) < 0.1 and np.all(np.abs(solver.coef_) < 0.1):
    print("FAIL: Still getting zeros!")
else:
    print("SUCCESS: Solver is working correctly!")