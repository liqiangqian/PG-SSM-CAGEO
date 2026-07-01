# PG-SSM-CAGEO: Physically Motivated Probabilistic Graph State-Space Framework for Short-Term Uranium Concentration Forecasting

This repository provides the source code and demonstration data for the manuscript:

**A Physically Motivated Probabilistic Graph State-Space Framework for Short-Term Uranium Concentration Forecasting in Five-Spot In-Situ Leaching Wellfield Units**

**Public repository:** [https://github.com/liqiangqian/PG-SSM-CAGEO](https://github.com/liqiangqian/PG-SSM-CAGEO)

## Overview

PG-SSM is a physically motivated graph state-space workflow for **short-term probabilistic forecasting** from sparse five-spot in-situ leaching (ISL) monitoring records. The implementation integrates (i) a **flow-modulated** five-spot graph affinity prior, (ii) **dual-branch** temporal encoding (TCN + LSTM) with gated fusion, (iii) **soft physical-plausibility** regularization terms, and (iv) a **Gaussian** predictive head with interval metrics.

The repository is intended for code execution and workflow verification. It provides a short-term forecasting workflow, not a full process simulator, not a site-calibrated groundwater model, and not a site-universal uranium concentration predictor.

See `docs/paper_alignment.md`, `docs/revision_experiments.md`, and `docs/reproducibility_note.md` for the mapping between this public workflow and the revised manuscript.

## Repository scope

This public repository contains demonstration-scale implementation files for:

- preprocessing of synthetic five-spot monitoring records;
- flow-modulated graph construction;
- PG-SSM training;
- deterministic evaluation;
- probabilistic calibration diagnostics;
- multi-horizon forecasting;
- rolling-origin stress testing;
- ablation / component-removal diagnostics;
- bootstrap confidence-interval estimation;
- synthetic demonstration datasets for workflow verification.

## Repository structure

```text
configs/        YAML configuration (`demo.yaml` full demo; `quick_test.yaml` fast smoke test)
data/           Synthetic demonstration data and data README files
src/            Preprocessing, graph construction, model, train, evaluate, metrics
examples/       Quick-test guide and command script
scripts/        Synthetic workflow and revision-experiment entry points
docs/           Data schema, user guide, reproducibility, and paper-alignment notes
outputs/        Runtime artifacts only (see note below; not meant for initial commits)
```

**Initial public checkout:** keep `outputs/` empty except `.gitignore` / `.gitkeep`. Do not commit generated `.pt` checkpoints or metrics JSON from local runs. Editors and reviewers regenerate them with `python examples/quick_test.py`.

## Installation

```bash
git clone https://github.com/liqiangqian/PG-SSM-CAGEO.git
cd PG-SSM-CAGEO
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start with synthetic data

```bash
python scripts/run_preprocessing.py --config configs/pgssm_default.yaml --quick_test
python scripts/run_train_pgssm.py --config configs/pgssm_default.yaml --quick_test --epochs 1
python scripts/run_evaluate.py --config configs/pgssm_default.yaml --quick_test
```

These commands use the synthetic dataset under `data/synthetic/` and write runtime files to `outputs/synthetic_quick_test/`.

The legacy quick-test driver is also retained:

```bash
python examples/quick_test.py
```

## Revision experiments

The major-revision workflows are available through dedicated scripts and configs:

```bash
python scripts/run_multi_horizon.py --config configs/multi_horizon.yaml --quick_test
python scripts/run_rolling_origin.py --config configs/rolling_origin.yaml --quick_test
python scripts/run_calibration.py --config configs/calibration.yaml --quick_test
python scripts/run_ablation.py --config configs/ablation.yaml --quick_test
python scripts/run_bootstrap_ci.py --config configs/bootstrap_ci.yaml --quick_test
```

See `docs/revision_experiments.md` for the purpose, scope, and limitations of these workflows.

Expected **test-set** outputs on the synthetic quick test include at least:

```text
RMSE
MAE
R2
PI90_coverage
```

Revision-experiment outputs are JSON or CSV files under `outputs/synthetic_quick_test/`. Generated checkpoints, processed arrays, and metrics files are runtime artifacts and are not intended for version control.

The quick tests are deliberately small. They verify preprocessing, graph construction, training, probabilistic inference, calibration helpers, ablation switches, and bootstrap utilities on synthetic data; they do not reproduce the manuscript's field-data RMSE, R2, PI90, CRPS, rolling-origin, bootstrap, ablation, or additional-archive tables.

## Main workflow (manual)

```bash
python scripts/generate_synthetic_demo.py   # optional: set PGSSM_QUICK_SYNTHETIC_N=750 for longer series
python src/train.py --config configs/demo.yaml
python src/evaluate.py --config configs/demo.yaml
```

## Data availability note

The confidential industrial monitoring archive used in the manuscript cannot be publicly released due to confidentiality restrictions. This repository provides a **synthetic** demonstration dataset with a similar general variable structure for workflow verification and code execution. The synthetic dataset is not intended to reproduce the confidential site-specific numerical values or the exact performance metrics reported in the manuscript.

The synthetic records use anonymized unit identifiers, anonymized well identifiers, and synthetic coordinates. They must not be interpreted as field data.

## Code availability

The source code is publicly available at [https://github.com/liqiangqian/PG-SSM-CAGEO](https://github.com/liqiangqian/PG-SSM-CAGEO) under the MIT License.

## Software requirements

Python 3.10+ recommended. Core dependencies are listed in `requirements.txt` (NumPy, pandas, scikit-learn, PyTorch, PyYAML, Matplotlib).

## License

This repository is released under the MIT License (see `LICENSE`).

## Citation

If you use this repository, please cite the associated manuscript after publication.

## Optional: GitHub Release (recommended)

A release is **not** required by the journal, but it gives editors a stable version pointer. On GitHub: **Releases → Draft a new release**, then use:

- **Tag:** `v1.0.0-cageo-submission`
- **Release title:** Initial public release for CAGEO submission
- **Release notes:**

```text
This release contains the source code, synthetic demonstration dataset, configuration files, documentation, and quick-test example for the physically motivated probabilistic PG-SSM framework submitted to Computers & Geosciences.
```

## Computer Code Availability (for the manuscript)

Use the section title **Computer Code Availability** in the main text. Suggested wording:

```text
The source code of PG-SSM is publicly available at https://github.com/liqiangqian/PG-SSM-CAGEO under the MIT License. The repository contains implementation files for preprocessing, flow-modulated graph construction, PG-SSM training, deterministic evaluation, probabilistic calibration, multi-horizon forecasting, rolling-origin stress testing, ablation analysis, and bootstrap confidence-interval estimation.

Because the original industrial monitoring archive is confidential, the repository provides a synthetic demonstration dataset with the same general variable structure for workflow verification and code execution. The synthetic dataset is not intended to reproduce the confidential site-specific numerical values or the exact performance metrics reported in the manuscript.
```
