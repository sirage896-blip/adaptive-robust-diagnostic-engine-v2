import numpy as np
from wlad_solver import WLADSolver
from statsmodels.nonparametric.smoothers_lowess import lowess

np.random.seed(42)

n = 200
p = 2
mean = [50.0, 30.0]
cov = [[225.0, 45.0], [45.0, 144.0]]
X = np.random.multivariate_normal(mean, cov, size=n)
beta_true = np.array([10.0, 0.5, -0.3])
y = beta_true[0] + X[:, 0]*beta_true[1] + X[:, 1]*beta_true[2] + np.random.normal(0, 5, n)

wlad = WLADSolver()
wlad.fit(X, y)

fitted = wlad.fitted_
residuals = wlad.residuals_
mad = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
standardized = np.abs(residuals) / mad

print("=" * 60)
print("STEP-BY-STEP ENVELOPE TRACE")
print("=" * 60)
print(f"Original standardized residuals:")
print(f"  Min: {standardized.min():.4f}, Med: {np.median(standardized):.4f}")
print(f"  Max: {standardized.max():.4f}, 97.5th: {np.percentile(standardized, 97.5):.4f}")
print()

# Bootstrap with B=10
rng = np.random.RandomState(42)
clean_indices = np.arange(n)
n_clean = n
bootstrap_std = np.zeros((10, n))

for b in range(10):
    boot_idx = rng.choice(clean_indices, size=n_clean, replace=True)
    X_boot = X[boot_idx]
    y_boot = y[boot_idx]
    w_boot = np.ones(n_clean)
    
    try:
        wlad_boot = WLADSolver()
        wlad_boot.fit(X_boot, y_boot, weights=w_boot,
                      warm_start=np.concatenate([[wlad.intercept_], wlad.coef_]))
        fitted_boot = wlad_boot.predict(X)
        resid_boot = y - fitted_boot
        mad_boot = np.median(np.abs(resid_boot - np.median(resid_boot))) * 1.4826
        if mad_boot < 1e-10:
            mad_boot = 1e-10
        bootstrap_std[b] = np.abs(resid_boot) / mad_boot
        print(f"Bootstrap {b}: SUCCESS, mad={mad_boot:.4f}, max_std={bootstrap_std[b].max():.4f}")
    except Exception as e:
        bootstrap_std[b] = standardized
        print(f"Bootstrap {b}: FAILED - {e}")

print()
raw_upper = np.percentile(bootstrap_std, 97.5, axis=0)
print(f"Raw upper percentile (before LOWESS):")
print(f"  Min: {raw_upper.min():.4f}, Med: {np.median(raw_upper):.4f}")
print(f"  Max: {raw_upper.max():.4f}")
print()

# Test LOWESS
upper_smooth = lowess(raw_upper, fitted, frac=0.3, return_sorted=False)
print(f"LOWESS result shape: {upper_smooth.shape}")
print(f"LOWESS result type: {type(upper_smooth)}")
print(f"LOWESS upper (after smoothing):")
print(f"  Min: {upper_smooth.min():.4f}, Med: {np.median(upper_smooth):.4f}")
print(f"  Max: {upper_smooth.max():.4f}")
print()

# Test with sorted data
sort_idx = np.argsort(fitted)
fitted_sorted = fitted[sort_idx]
raw_upper_sorted = raw_upper[sort_idx]
upper_smooth_sorted = lowess(raw_upper_sorted, fitted_sorted, frac=0.3, return_sorted=False)
print(f"LOWESS with sorted data:")
print(f"  Min: {upper_smooth_sorted.min():.4f}, Med: {np.median(upper_smooth_sorted):.4f}")
print(f"  Max: {upper_smooth_sorted.max():.4f}")