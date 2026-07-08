# Adaptive Robust Diagnostic Engine

**Closing the Finite-Sample Robustness Gap:**  
An Adaptive Diagnostic Engine for Simultaneous Leverage and Heteroscedasticity in Linear Regression

Based on the MSc thesis by **Muhammed Kandeh** (Mat# 22613547),  
University of The Gambia, Department of Mathematics, 2027.

---

## Overview

This engine provides a four-stage statistical pipeline that ensures robust parameter estimation and valid inference under simultaneous leverage contamination and heteroscedasticity:

| Stage | Component | Purpose |
|-------|-----------|---------|
| **1** | Fast-MCD + Adaptive Weighting | Robust leverage detection with data-adaptive weight floor δ* |
| **2** | WLAD | Weighted Least Absolute Deviation estimation |
| **3** | LBEP | LOWESS-Optimized Bootstrap Envelope Plot diagnostics |
| **4** | BCa Bootstrap | Bias-Corrected and Accelerated inference |

**Key Innovation:** The adaptive weight-floor selector δ* extends practical robustness from ~15% to over 35% contamination in finite samples.

---

## Requirements

- Python 3.10+
- NumPy, SciPy, scikit-learn, statsmodels, matplotlib, pandas

## Installation

```bash
git clone <repository-url>
cd adaptive-robust-diagnostic-engine
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```python
from engine import AdaptiveRobustDiagnosticEngine
import numpy as np

# Load your data
# X = your_design_matrix  # shape (n, p)
# y = your_response       # shape (n,)

# Initialize and run
engine = AdaptiveRobustDiagnosticEngine(
    coverage=0.5,        # Fast-MCD support fraction
    alpha=0.05,          # Significance level
    n_bootstrap=1000,    # Bootstrap replications
    lowess_frac=0.3,     # LOWESS smoothing span
    random_state=42      # Reproducibility seed
)

results = engine.fit(X, y)

# Access results
print(results['coefficients'])           # WLAD coefficients
print(results['confidence_intervals'])   # BCa 95% CIs
print(results['outlier_indices'])        # Flagged outliers
print(results['adaptive_delta'])           # Data-adaptive δ*

# Generate diagnostic plot
fig, ax = engine.plot_lbep(save_path="lbep_output.png")

# Full summary
engine.summary(feature_names=["X1", "X2", "X3"])
```

## Module Structure

```
adaptive-robust-diagnostic-engine/
├── engine.py              # Main pipeline orchestrator
├── mcd_core.py            # Fast-MCD implementation
├── adaptive_weights.py    # Adaptive δ* selector
├── wlad_solver.py         # WLAD linear programming solver
├── diagnostics.py         # LBEP visualization
├── bca_bootstrap.py       # BCa confidence intervals
├── demo.py                # Working example
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Computational Performance

| Stage | n=100 | n=500 | n=1000 |
|-------|-------|-------|--------|
| Fast-MCD + Adaptive | 0.5s | 2.2s | 5.4s |
| WLAD | 0.2s | 1.9s | 4.7s |
| LBEP (B=1000) | 12s | 45s | 99s |
| BCa Bootstrap | 19s | 52s | 112s |
| **Total (parallel)** | **8s** | **29s** | **63s** |

*Times on 6-core Intel i7, 16GB RAM. Parallelization reduces runtime significantly.*

## Reproducibility

- Fixed random seeds (42) ensure exact replication
- All Monte Carlo experiments in the thesis are reproducible
- Bootstrap methods may show minor Monte Carlo variation

## Citation

```bibtex
@mastersthesis{kandeh2027,
  author  = {Kandeh, Muhammed},
  title   = {Closing the Finite-Sample Robustness Gap: 
             An Adaptive Diagnostic Engine for Simultaneous Leverage 
             and Heteroscedasticity in Linear Regression},
  school  = {University of The Gambia},
  year    = {2027},
  type    = {MSc Thesis},
  department = {Department of Mathematics}
}
```

## License

MIT License

## Contact

For questions about methodology or implementation, please open an issue on the GitHub repository.
