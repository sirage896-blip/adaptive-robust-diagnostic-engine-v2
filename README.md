# Adaptive Robust Diagnostic Engine (ARDE)

> **Closing the Finite-Sample Robustness Gap: An Adaptive Diagnostic Engine for Simultaneous Leverage and Heteroscedasticity in Linear Regression**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [The Four-Stage Pipeline](#the-four-stage-pipeline)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Modules](#core-modules)
- [Theoretical Foundations](#theoretical-foundations)
- [Simulation Results](#simulation-results)
- [Real-World Applications](#real-world-applications)
- [API Reference](#api-reference)
- [Performance Benchmarks](#performance-benchmarks)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

The **Adaptive Robust Diagnostic Engine (ARDE)** is a unified, open-source statistical pipeline designed to address a critical gap in robust regression analysis: the simultaneous occurrence of **high-leverage outliers** and **heteroscedasticity** in linear regression models.

While classical Ordinary Least Squares (OLS) estimators achieve optimal properties under strict Gauss-Markov assumptions, real-world administrative data—particularly in developing nations—routinely violate these assumptions. The result is biased parameter estimates, invalid inference, and potentially harmful policy decisions.

### The Finite-Sample Gap Problem

Existing high-breakdown methods (e.g., MM-estimation) achieve a theoretical **50% asymptotic breakdown point**, but in practice their robustness collapses under **>15% contamination** due to fixed numerical floors in the weighting scheme. This thesis introduces a **data-adaptive weight-floor selector δ*** that extends practical robustness to **>35% contamination** while maintaining nominal 95% coverage up to 25% contamination.

**Key Innovation:** A closed-form, data-driven selector δ*(X, n, p) ∝ Cond(X)⁻²ᐟ³ · p/n that dynamically optimizes the finite-sample bias-variance trade-off based on the empirical geometry of your data.

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Fast-MCD Leverage Detection** | Affine-equivariant, high-breakdown outlier detection with adaptive weighting |
| ⚖️ **WLAD Robust Estimation** | Weighted Least Absolute Deviation via IRLS for simultaneous horizontal & vertical outlier resistance |
| 📊 **LBEP Diagnostics** | LOWESS-Optimized Bootstrap Envelope Plots with objective, data-driven outlier thresholds |
| 📈 **BCa Inference** | Bias-Corrected and Accelerated residual bootstrap for valid inference under L₁ sparsity |
| 🎯 **Adaptive δ*** | Data-driven weight-floor selector that closes the finite-sample robustness gap |
| ⚡ **Parallelized** | Embarrassingly parallel bootstrap and jackknife computation |
| 🌍 **Policy-Ready** | Validated on educational data from The Gambia and standard benchmarks (HBK, Brownlee) |

---

## The Four-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE ROBUST DIAGNOSTIC ENGINE                        │
│           Unified Framework for Simultaneous Leverage & Heteroscedasticity    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: (X, y) ──► Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4 ──► Output   │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  STAGE 1     │  │  STAGE 2     │  │  STAGE 3     │  │  STAGE 4     │    │
│  │  Robust      │  │  Weighted    │  │  LBEP        │  │  BCa         │    │
│  │  Leverage    │  │  LAD         │  │  Diagnostics │  │  Bootstrap   │    │
│  │  Detection   │  │  Estimation  │  │              │  │  Inference   │    │
│  │              │  │              │  │              │  │              │    │
│  │ • Fast-MCD   │  │ • IRLS Solver│  │ • Bootstrap  │  │ • Residual   │    │
│  │ • Adaptive   │  │ • L₁-Norm    │  │   Envelopes│  │   Bootstrap  │    │
│  │   δ* Selector│  │   Minimization│  │ • LOWESS    │  │ • Jackknife  │    │
│  │ • Robust     │  │ • Bounded    │  │   Smoothing │  │   Acceleration│   │
│  │   Mahalanobis│  │   Influence  │  │ • frac=0.4  │  │ • Bias-Corrected│  │
│  │   Distances  │  │              │  │              │  │   CIs        │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│        │                 │                 │                 │              │
│        ▼                 ▼                 ▼                 ▼              │
│   Breakdown Point    Efficiency        Diagnostics        Inference          │
│   ≈ 35% (Adaptive)   Near-nominal      Objective          Non-parametric    │
│                      coverage           thresholds         valid             │
│                                                                             │
│  Output: β̂_WLAD, Outliers ⊂ O, BCa CIs                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Robust Leverage Detection (Fast-MCD + Adaptive Weighting)

```python
# Core computation
(μ̂_MCD, Σ̂_MCD) ← FastMCD(X, h)                    # High-breakdown covariance
RDᵢ ← √[(xᵢ - μ̂_MCD)' Σ̂_MCD⁻¹ (xᵢ - μ̂_MCD)]      # Robust Mahalanobis distances
δ* ← max(10⁻⁴, min(0.10, (p/n) · (λ_min/λ_max)))  # Adaptive weight-floor (Eq. 3.12)
wᵢ ← max(δ*, min(1, χ²_p,₀.₉₇₅ / RDᵢ²))           # Distance-based weights
```

**Why it matters:** Classical Mahalanobis distances fail under masking/swamping. Fast-MCD achieves ~50% breakdown via concentration steps (C-steps), and our adaptive δ* selector optimizes the bias-variance trade-off empirically.

### Stage 2: Weighted Least Absolute Deviation (WLAD) Estimation

```python
β̂_WLAD ← arg min_β Σ wᵢ |yᵢ - xᵢ'β|              # Weighted L₁ objective
# Solved via IRLS: vᵢ = wᵢ / (|rᵢ| + ε), ε = 10⁻⁸
```

**Key properties:**
- Regression, scale, and affine equivariant (when weights from affine-equivariant MCD)
- Influence bounded linearly in response (via sgn(·)) and in predictor (via w(x₀))
- Theoretical breakdown point ≈ 50%; practical breakdown ≈ 35% with adaptive δ*

### Stage 3: LOWESS-Optimized Bootstrap Envelope Plot (LBEP)

```python
# Bootstrap from clean subset C = {i : wᵢ > 0.5}
For b = 1, ..., B:
    Resample C with replacement → refit WLAD → compute sᵢ⁽ᵇ⁾ = |rᵢ⁽ᵇ⁾|/MAD

Raw percentiles: êᵢ⁽²·⁵⁾, êᵢ⁽⁹⁷·⁵⁾ at each fitted value ŷᵢ
Optimized envelope: ẽᵢ ← LOWESS(ŷᵢ, êᵢ, frac=0.4)  # Suppresses L₁ sparsity sawtooth
Flag outliers: O ← {i : sᵢ > ẽᵢ⁽⁹⁷·⁵⁾}
```

**The Sparsity-Induced Sawtooth Phenomenon:** L₁ regression forces the fitted hyperplane through ≥ p design points, creating exact-zero residuals. Bootstrap resampling of these sparse residuals produces volatile "sawtooth" percentile boundaries. LOWESS smoothing extracts stable functional envelopes for objective outlier delineation.

### Stage 4: BCa Residual Bootstrap Inference

```python
# Full-sample residual resampling (NOT the clean subset—avoids selection bias)
r̃ᵢ ← rᵢ - median(r)                                # Centered residuals
For b = 1, ..., B:
    r*⁽ᵇ⁾ ~ {r̃₁, ..., r̃ₙ} with replacement
    y*⁽ᵇ⁾ ← Xβ̂_WLAD + r*⁽ᵇ⁾
    Refit WLAD → β̂*⁽ᵇ⁾

# BCa adjustment
ẑ₀ ← Φ⁻¹( (1/B) Σ I(β̂*⁽ᵇ⁾ < β̂_WLAD) )            # Bias correction
â ← jackknife_acceleration(β̂_₍ᵢ₎)                # Skewness correction
CI ← [Φ(ẑ₀ + (ẑ₀ + z_α/₂)/(1 - â(ẑ₀ + z_α/₂))),
      Φ(ẑ₀ + (ẑ₀ + z_₁₋α/₂)/(1 - â(ẑ₀ + z_₁₋α/₂)))]
```

**Why BCa over HCCM:** Naive application of HC4/HC5 to L₁ residuals produces coverage as low as 76% (clean) or 100% (leverage contamination) due to exact-zero residuals violating continuous-distribution assumptions. BCa bypasses density estimation entirely.

---

## Installation

### Prerequisites

- Python 3.10+
- 6+ CPU cores recommended for parallelization
- 16GB+ RAM for n > 500

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/sirage896-blip/adaptive-robust-diagnostic-engine-v2.git
cd adaptive-robust-diagnostic-engine-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
matplotlib>=3.4.0
pandas>=1.3.0
```

---

## Quick Start

### Basic Usage

```python
from engine import AdaptiveRobustDiagnosticEngine
import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv('your_data.csv')
X = data[['predictor1', 'predictor2', 'predictor3']].values
y = data['response'].values

# Initialize and run the full pipeline
engine = AdaptiveRobustDiagnosticEngine(
    coverage=0.5,           # MCD coverage parameter (h ≈ 0.5n)
    alpha=0.05,             # Significance level for CIs
    n_bootstrap=1000,       # Bootstrap replications (B ≥ 1000 recommended)
    lowess_frac=0.4,        # LOWESS span parameter (validated optimal)
    random_state=42         # Reproducibility
)

results = engine.fit(X, y)

# Access results
print("Robust coefficients:", results['coefficients'])
print("BCa 95% CIs:
", results['confidence_intervals'])
print("Outlier indices:", results['outlier_indices'])
print("Adaptive δ*:", results['adaptive_delta'])
print("Downweighted observations:", results['n_downweighted'])
```

### Interpreting the Adaptive Selector δ*

| δ* Value | Interpretation | Action |
|----------|---------------|--------|
| ≈ 0.01 | Data may be contaminated or ill-conditioned | Robust inference essential |
| ≈ 0.05 | Moderate contamination; useful robust check | Compare with OLS |
| ≈ 0.10 | Data appear relatively clean | Classical & robust should agree |

### Example: Gambian Educational Data

```python
# Reproduce Chapter 6 results
data = pd.read_csv('gambia_education/student_performance.csv')
X = data[['STR', 'PQT', 'SII', 'PSM']].values  # Predictors
y = data['performance_index'].values            # Response

engine = AdaptiveRobustDiagnosticEngine(
    n_bootstrap=1000,
    lowess_frac=0.4,
    random_state=42
)
results = engine.fit(X, y)

# Key findings (from thesis):
# STR: OLS = -1.84, WLAD = -0.92 (BCa CI: [-1.45, -0.38])
# PQT: OLS = 0.67,  WLAD = 0.41  (BCa CI: [0.12, 0.71])
# 23 high-leverage schools identified (4.8% of n=482)
# δ* = 0.067 (well-conditioned data)
```

---

## Core Modules

```
adaptive-robust-diagnostic-engine-v2/
├── engine.py              # Unified pipeline orchestration (Stages 1-4)
├── mcd_core.py            # Fast-MCD & deterministic MCD with C-steps
├── adaptive_weights.py    # δ* selector via eigenratio (Eq. 3.12)
├── wlad_solver.py         # WLAD estimation via IRLS
├── diagnostics.py         # LBEP visualization & bootstrap envelopes
├── bca_bootstrap.py       # BCa CI construction with jackknife
│
├── notebooks/             # Reproduce all figures and tables
│   ├── figure_3_1_influence_function.ipynb
│   ├── figure_3_2_delta_heatmap.ipynb
│   ├── table_5_1_bias_rmse.ipynb
│   ├── figure_5_4_coverage_curves.ipynb
│   └── ...
│
├── data/
│   ├── gambia_education/  # Anonymized MoBSE data (n=482 schools)
│   ├── hbk/               # Hawkins-Bradu-Kass benchmark (n=75)
│   └── brownlee/          # Brownlee Stack Loss (n=21)
│
├── scripts/               # Reproducibility scripts
│   ├── run_simulations.py
│   ├── run_benchmarks.py
│   └── run_gambia_analysis.py
│
├── tests/                 # Unit tests
├── requirements.txt
├── LICENSE (MIT)
└── README.md (this file)
```

---

## Theoretical Foundations

### Finite-Sample Breakdown Function

**Theorem 3.1** (Finite-Sample Breakdown Point):

Let ε*_MCD = (n - h + 1)/n be the Fast-MCD breakdown point. With adaptive floor δ*, the WLAD empirical breakdown point is:

```
ε̂*_n(δ*) = (n - h + 1)/n - κ̂_scale · g(δ*, p)
```

where `g(δ*, p) = δ* · tr(Σ̂_MCD) / λ_min(Σ̂_MCD)` is the penalty function, and `κ̂_scale` is a plug-in estimator from the MCD clean subset.

**Key insight:** As δ* → 0⁺, g(δ*, p) → 0, and the practical breakdown point approaches the theoretical 50% limit. The adaptive selector δ* minimizes this upper bound empirically.

### Adaptive Weight-Floor Selector

**Proposition 3.1** (Optimal Weight Floor):

```
δ* = max(10⁻⁴, min(0.10, (p/n) · (λ_min(Σ̂_MCD) / λ_max(Σ̂_MCD))))
     = max(10⁻⁴, min(0.10, (p/n) · Cond(Σ̂_MCD)⁻¹))
```

**Scaling:** δ* ∝ Cond(X)⁻²ᐟ³ · p/n (theoretical); implemented as ∝ Cond(X)⁻¹ · p/n for computational stability (correlation r = 0.98 with theoretical form).

### Influence Function

**Theorem 3.2** (Bounded Influence):

```
IF(x₀, y₀; T, F) = [E_F[w(x)f_{Y|X}(x'β₀)xx']]⁻¹ × w(x₀)sgn(y₀ - x₀'β₀)x₀
```

Bounded in **both** response (via sgn(·) ∈ [-1, 1]) and predictor (via w(x₀) = min(1, c/RD²(x₀)) ≥ δ* > 0).

### IRLS Convergence

**Theorem 3.5:** The IRLS algorithm converges superlinearly to the WLAD solution in a neighborhood of the optimum. The fixed point satisfies the L₁ subgradient optimality condition exactly as ε → 0⁺.

---

## Simulation Results

### Monte Carlo Design

- **Sample size:** n = 200
- **Predictors:** p = 2 (β = [2, 1.5, -0.8]')
- **Replications:** R = 1,000
- **Bootstrap:** B = 1,000
- **Contamination:** Clean, 10% vertical, 10% leverage, 10% combined
- **Heteroscedasticity:** Homoscedastic vs. quadratic (σ²ᵢ = σ²(1 + 0.5xᵢ₁²))

### Parameter Estimation (Table 5.1)

| Estimator | Clean (Bias) | Combined-Het (Bias) | Clean (RMSE) | Combined-Het (RMSE) |
|-----------|-------------|---------------------|-------------|---------------------|
| OLS | 0.0004 | 0.2433 | 0.0078 | 0.0423 |
| **WLAD-Adaptive** | **0.0004** | **0.0013** | **0.0095** | **0.0105** |

### Coverage Probabilities (Table 5.2)

| Method | Clean-Het | Leverage-Het | Combined-Het |
|--------|-----------|--------------|--------------|
| OLS-Classical | 0.723 | 0.312 | 0.245 |
| HC4-OLS | 0.761 | 1.000 | 1.000 |
| **WLAD-Adaptive-BCa** | **0.951** | **0.936** | **0.928** |

### Escalating Contamination (Table 5.5)

| Contamination | WLAD-Static (δ=0.05) | **WLAD-Adaptive (δ*)** | MM-Estimation |
|---------------|----------------------|------------------------|---------------|
| 10% | 0.925 | **0.941** | 0.912 |
| 20% | 0.874 | **0.931** | 0.851 |
| 30% | 0.802 | **0.915** | 0.734 |
| **35%** | 0.764 | **0.903** | 0.681 |

> **Result:** Adaptive selector extends usable coverage (>90%) from 15% (static) to 35% contamination—a **+20 percentage point** improvement.

### Type I Error Control (Table 5.10)

| Method | Clean Type I | Combined Type I | Clean Power | Combined Power |
|--------|-------------|-----------------|-------------|----------------|
| OLS-Classical | 0.050 | 0.782 | 0.942 | 0.234 |
| HC4-OLS | 0.052 | 0.781 | 0.941 | 0.045 |
| MM-Estimation | 0.051 | **0.158** | 0.935 | 0.845 |
| **WLAD-Adaptive** | **0.048** | **0.056** | **0.938** | **0.892** |

> **Result:** Under combined contamination, MM-estimation exhibits 15.8% Type I error (severe size distortion), while WLAD-Adaptive maintains nominal 5.6% control.

---

## Real-World Applications

### 1. Gambian Educational Data (Primary Application)

**Context:** Ministry of Basic and Secondary Education (MoBSE) resource allocation

| Variable | OLS Estimate | WLAD-Adaptive | Policy Impact |
|----------|-------------|---------------|---------------|
| Student-Teacher Ratio (STR) | -1.84*** (0.34) | **-0.92** (0.28) | OLS overestimates by **100%** |
| % Qualified Teachers (PQT) | 0.67** (0.21) | **0.41** (0.18) | OLS overestimates by **63%** |
| Infrastructure Index (SII) | 2.11*** (0.45) | **1.78*** (0.38) | Robust effect confirmed |

**Budget Implication:** A 5-point STR reduction policy budgeted at 18.4M GMD (OLS) requires only **9.2M GMD** (robust)—saving **9.2M GMD** while achieving the same outcome.

**Outlier Profile:** 23 high-leverage schools identified (8 elite urban, 15 rural deprivation), δ* = 0.067.

### 2. Hawkins-Bradu-Kass (HBK) Benchmark

- **Dataset:** n=75, p=3, 10 known leverage outliers (observations 1-10)
- **Result:** Fast-MCD TPR = 1.00, FPR = 0.00; δ* = 0.012
- **Validation:** All 10 leverage points clearly above 97.5th LBEP envelope; masking immunity confirmed

### 3. Brownlee Stack Loss Data

- **Dataset:** n=21, p=3, 4 known outliers (1, 3, 4, 21)
- **Result:** All 4 outliers identified; observation 21 = vertical outlier, {1,3,4} = leverage contamination
- **δ* = 0.008** (small sample, adaptive floor shrinks appropriately)

### 4. World Development Indicators (Macroeconomic)

- **Dataset:** Gambian GDP per capita, 1990-2023 (n=34)
- **Flagged years:** 1990, 1991, 1992 (post-civil war), 2003, 2004 (currency devaluation), 2023 (post-COVID)
- **Interpretation:** Structural breaks correctly identified as genuine economic events, not data errors

---

## API Reference

### `AdaptiveRobustDiagnosticEngine`

```python
class AdaptiveRobustDiagnosticEngine(
    coverage: float = 0.5,           # MCD coverage h/n (0.5 = maximal breakdown)
    alpha: float = 0.05,             # Significance level for BCa CIs
    n_bootstrap: int = 1000,         # Bootstrap replications (≥1000 for final)
    lowess_frac: float = 0.4,        # LOWESS span (0.3-0.5 stable)
    random_state: int = 42,          # Reproducibility seed
    max_iter_wlad: int = 100,        # IRLS max iterations
    tol_wlad: float = 1e-6,          # IRLS convergence tolerance
    ridge_lambda: float = 1e-8,      # Numerical regularization
    n_jobs: int = -1                 # Parallel cores (-1 = all available)
)
```

### Methods

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Run full 4-stage pipeline |
| `stage1_leverage_detection(X)` | Fast-MCD + adaptive weights |
| `stage2_wlad_estimation(X, y, weights)` | IRLS solver |
| `stage3_lbep_diagnostics(X, y, beta, weights)` | Bootstrap envelopes + LOWESS |
| `stage4_bca_inference(X, y, beta, residuals)` | BCa CIs |

### Output Dictionary

```python
{
    'coefficients': np.ndarray,           # β̂_WLAD (p,)
    'confidence_intervals': np.ndarray,     # BCa CIs (p, 2)
    'standard_errors': np.ndarray,          # Bootstrap SEs (p,)
    'outlier_indices': np.ndarray,          # Flagged outlier indices
    'adaptive_delta': float,               # δ* value
    'n_downweighted': int,                 # |{i : wᵢ < 0.5}|
    'weights': np.ndarray,                 # Final weights w (n,)
    'robust_distances': np.ndarray,        # RDᵢ (n,)
    'residuals': np.ndarray,               # Raw residuals r (n,)
    'standardized_residuals': np.ndarray,  # sᵢ = |rᵢ|/MAD (n,)
    'lbep_envelope': dict,                 # {'lower': ..., 'upper': ..., 'fitted': ...}
    'bca_factors': dict,                   # {'z0': ..., 'a': ...}
    'mcd_subset': np.ndarray,              # Clean subset indices C
    'condition_number': float,             # Cond(Σ̂_MCD)
    'convergence': bool                   # IRLS convergence flag
}
```

---

## Performance Benchmarks

### Runtime Scaling (6-core Intel i7-10750H, 16GB RAM, B=1000)

| Stage | n=100 | n=500 | n=1000 |
|-------|-------|-------|--------|
| Fast-MCD + δ* | 0.48s | 2.18s | 5.41s |
| WLAD (IRLS) | 0.15s | 0.89s | 2.10s |
| LBEP (B=1000) | 12.4s | 45.2s | 98.6s |
| BCa Bootstrap | 18.7s | 52.3s | 112.4s |
| **Total (parallel)** | **~12.5s** | **~22s** | **~48s** |

**Scaling:** Approximately O(n) for fixed p. Parallelization achieves 2.5-3× speedup on 6 cores.

### Memory Management

- Bootstrap matrices stored as 32-bit floats
- Chunked processing supported for n > 2000
- Peak RAM: ~2GB for n=500, p=8, B=1000

---

## Reproducibility

All analyses in the thesis are fully reproducible:

```bash
# Reproduce all figures and tables
jupyter notebooks/figure_3_1_influence_function.ipynb
jupyter notebooks/table_5_1_bias_rmse.ipynb
# ... etc

# Or run batch scripts
python scripts/run_simulations.py    # Chapter 5 simulations
python scripts/run_benchmarks.py      # HBK & Brownlee validation
python scripts/run_gambia_analysis.py # Chapter 6 application
```

**Fixed seeds:** Random state = 42 throughout all Monte Carlo experiments ensures exact replication.

**Monte Carlo variation:** Bootstrap-based methods (LBEP, BCa) may produce minor variation; reported results use B = 1000 replications.

---

## Citation

If you use this engine in your research, please cite:

```bibtex
@mastersthesis{kandeh2026adaptive,
  title={Closing the Finite-Sample Robustness Gap: An Adaptive Diagnostic Engine 
         for Simultaneous Leverage and Heteroscedasticity in Linear Regression},
  author={Kandeh, Muhammed},
  school={University of The Gambia, School of Arts and Science, Department of Mathematics},
  year={2026},
  type={Master of Science in Statistics and Data Science},
  note={Matriculation Number: 22613547. Supervisor: Dr. I. A. Baba}
}
```

---

## License

This project is licensed under the **MIT License**—see the [LICENSE](LICENSE) file for details.

---

## Contact

**Author:** Muhammed Kandeh  
**Email:** [mk22613547@utg.edu.gm](mailto:mk22613547@utg.edu.gm) | [sirage896@gmail.com](mailto:sirage896@gmail.com)  
**GitHub Issues:** [github.com/sirage896-blip/adaptive-robust-diagnostic-engine-v2/issues](https://github.com/sirage896-blip/adaptive-robust-diagnostic-engine-v2/issues)

**Supervisor:** Dr. I. A. Baba  
**Institution:** University of The Gambia, Department of Mathematics

---

## Acknowledgements

- **Ministry of Basic and Secondary Education (MoBSE), The Gambia** — for providing educational data
- **World Bank & UNESCO** — for comprehensive international education databases
- **Dr. I. A. Baba** — for guidance, expertise, and patience throughout this research
- **Faculty and staff, Department of Mathematics, UTG** — for academic foundation and resources

---

*"In an era of increasing reliance on evidence-based governance, robust statistical methods are not methodological luxuries but ethical necessities."* — Chapter 7, Concluding Remarks
