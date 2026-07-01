# User Guide

## Installation

```bash
git clone https://github.com/liqiangqian/PG-SSM-CAGEO.git
cd PG-SSM-CAGEO
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick synthetic workflow

```bash
python scripts/run_preprocessing.py --config configs/pgssm_default.yaml --quick_test
python scripts/run_train_pgssm.py --config configs/pgssm_default.yaml --quick_test --epochs 1
python scripts/run_evaluate.py --config configs/pgssm_default.yaml --quick_test
```

## Major-revision diagnostics

```bash
python scripts/run_multi_horizon.py --config configs/multi_horizon.yaml --quick_test
python scripts/run_rolling_origin.py --config configs/rolling_origin.yaml --quick_test
python scripts/run_calibration.py --config configs/calibration.yaml --quick_test
python scripts/run_ablation.py --config configs/ablation.yaml --quick_test
python scripts/run_bootstrap_ci.py --config configs/bootstrap_ci.yaml --quick_test
```

## Output location

All quick-test outputs are written to `outputs/synthetic_quick_test/` or a subdirectory under it. These outputs are runtime artifacts and are not committed by default.

## Scope

The public workflow uses synthetic data only. It verifies code execution and the structure of the computational workflow. It does not reproduce confidential field-data metrics.
