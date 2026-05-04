# Scientific Rigor and Statistical Validation Plan

This document outlines the statistical framework used to ensure the reproducibility and validity of the SpinnyBall research data.

## 1. Convergence Criteria (Monte Carlo)

To ensure publication-grade confidence, all results in the `RESEARCH_DATASET.md` must meet the following criteria:

- **Confidence Interval (CI):** Standardized to 95% confidence level.
- **CI Width (Margin of Error):** Targeted at $\le 3.7\%$ of the mean value.
- **Realization Count ($N$):**
    - **Exploratory Sweeps:** $N=100$ per point.
    - **Publication-Grade High-Res Sweeps:** $N=3,000$ to $N=8,000$ per point (dependent on variance).

## 2. Statistical Metrics

The following metrics are tracked for every sweep:

- **Survival Probability ($P_s$):** Fraction of realizations where $k_{eff} \ge 6,000$ N/m for the entire duration.
- **Cascade Index ($\chi$):** Average number of nodes failed per realization, normalized by the initial failure count.
- **Recovery Latency ($\tau_{rec}$):** Time taken for MPC to return libration to $\le 0.05$ rad after a quench event.
- **Stiffness Minimum ($k_{min}$):** The absolute minimum effective stiffness reached across all realizations in a sweep point.

## 3. Data Provenance and Reproducibility

- **Random Seeding:** All sweeps use deterministic seeds (e.g., `np.random.seed(42)`) to allow exact replication of results.
- **Parameter Standardization:** All simulations use the `canonical_values.py` parameter set unless explicitly noted in the sweep script.
- **Artifact Preservation:** All generated heatmaps and CSV files are stored with timestamps in the `sweep_results/` directory.

## 4. Verification Methodology

### Cross-Validation
Results from the `CascadeRunner` (Numba-accelerated Python) are cross-validated against the `MuJoCo` physics oracle for specific edge cases (e.g., $u = 15,000$ m/s) to ensure the simplified stiffness model remains accurate.

### Sensitivity Analysis
Key boundaries (T1 Latency, T3 Failure Rate) are subjected to sensitivity sweeps where secondary parameters (e.g., packet mass) are varied by $\pm 10\%$ to ensure the observed stability is not a result of "over-tuning" parameters.
