#!/usr/bin/env bash
set -euo pipefail

python scripts/run_preprocessing.py --config configs/pgssm_default.yaml --quick_test
python scripts/run_train_pgssm.py --config configs/pgssm_default.yaml --quick_test --epochs 1
python scripts/run_evaluate.py --config configs/pgssm_default.yaml --quick_test
python scripts/run_multi_horizon.py --config configs/multi_horizon.yaml --quick_test
python scripts/run_rolling_origin.py --config configs/rolling_origin.yaml --quick_test
python scripts/run_calibration.py --config configs/calibration.yaml --quick_test
python scripts/run_ablation.py --config configs/ablation.yaml --quick_test
python scripts/run_bootstrap_ci.py --config configs/bootstrap_ci.yaml --quick_test
