# Revision Experiments

This document describes the additional experiments and diagnostic workflows added during the major revision.

## 1. Multi-horizon forecasting

Script: `scripts/run_multi_horizon.py`

Config: `configs/multi_horizon.yaml`

Purpose: evaluate horizon sensitivity at short-term forecasting horizons such as 3, 7, and 14 days.

Example command:

```bash
python scripts/run_multi_horizon.py --config configs/multi_horizon.yaml --quick_test
```

## 2. Rolling-origin stress test

Script: `scripts/run_rolling_origin.py`

Config: `configs/rolling_origin.yaml`

Purpose: evaluate temporal robustness under rolling-origin splits.

Example command:

```bash
python scripts/run_rolling_origin.py --config configs/rolling_origin.yaml --quick_test
```

## 3. Probabilistic calibration diagnostics

Script: `scripts/run_calibration.py`

Config: `configs/calibration.yaml`

Metrics: PI90 coverage, NLL, CRPS, Winkler score, interval sharpness, and PIT diagnostics.

Example command:

```bash
python scripts/run_calibration.py --config configs/calibration.yaml --quick_test
```

## 4. Component-removal and ablation diagnostics

Script: `scripts/run_ablation.py`

Config: `configs/ablation.yaml`

Ablation settings:

- without graph message passing: `--ablation no_graph`
- without dual-branch transition: `--ablation no_dual_branch`
- without physically motivated regularization: `--ablation no_physical_regularization`
- without spatial coupling features: `--ablation no_spatial_features`

Example command:

```bash
python scripts/run_ablation.py --config configs/ablation.yaml --quick_test
```

## 5. Bootstrap confidence intervals

Script: `scripts/run_bootstrap_ci.py`

Config: `configs/bootstrap_ci.yaml`

Purpose: estimate bootstrap confidence intervals for selected metrics.

Example command:

```bash
python scripts/run_bootstrap_ci.py --config configs/bootstrap_ci.yaml --quick_test
```

## Important limitation

The public repository uses synthetic demonstration data only. The confidential field monitoring archive cannot be publicly released. The synthetic dataset supports workflow verification and code execution, but it does not reproduce the site-specific numerical results reported in the manuscript.
