# Quick Test on Synthetic Data

This example runs the public PG-SSM workflow using only synthetic demonstration data.

```bash
python scripts/run_preprocessing.py --config configs/pgssm_default.yaml --quick_test
python scripts/run_train_pgssm.py --config configs/pgssm_default.yaml --quick_test --epochs 1
python scripts/run_evaluate.py --config configs/pgssm_default.yaml --quick_test
```

Major-revision diagnostic workflows can be checked with:

```bash
bash examples/quick_test_commands.sh
```

All outputs are written under `outputs/synthetic_quick_test/`. The synthetic dataset is provided only for workflow verification and code execution. It is not field data and cannot reproduce the confidential site-specific numerical values or manuscript performance metrics.
