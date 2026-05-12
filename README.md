# PG-SSM: Physics-Guided Graph State-Space Model

This repository provides the source code and demonstration data for the manuscript:

**A Physics-Guided Graph State-Space Model for Probabilistic Forecasting of Uranium Concentration Dynamics in In-Situ Leaching Wellfields**

## Overview

PG-SSM is a physics-guided graph state-space style framework for **probabilistic** forecasting of uranium concentration dynamics in structured in-situ leaching (ISL) wellfields. The implementation integrates (i) a **flow-aware** five-spot graph prior, (ii) **dual-branch** temporal encoding (TCN + LSTM) with gated fusion, (iii) **physics-guided** regularization terms, and (iv) a **Gaussian** predictive head with calibrated interval metrics.

See `docs/paper_alignment.md` for a concise mapping to the paper’s methods and what is intentionally out of scope on synthetic data.

## Repository structure

```text
configs/        YAML configuration (`demo.yaml` full demo; `quick_test.yaml` fast smoke test)
data/           Synthetic demonstration data and data README
src/            Preprocessing, graph construction, model, train, evaluate, metrics
examples/       Quick-test driver (uses `configs/quick_test.yaml` by default)
scripts/        Utility scripts (synthetic data generation)
docs/           Variable schema and paper-alignment notes
outputs/        Runtime artifacts only (see note below; not meant for initial commits)
```

**Initial public checkout:** keep `outputs/` empty except `.gitignore` / `.gitkeep`. Do not commit generated `.pt` checkpoints or metrics JSON from local runs. Editors and reviewers regenerate them with `python examples/quick_test.py`.

## Installation

Replace `https://github.com/liqiangqian/PG-SSM-CAGEO` below with your real repository URL before publishing.

```bash
git clone https://github.com/<your-username>/PG-SSM-CAGEO.git
cd PG-SSM-CAGEO
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick test

```bash
python examples/quick_test.py
```

This uses **`configs/quick_test.yaml`**: a smaller model, fewer epochs, a shorter synthetic series (via `PGSSM_QUICK_SYNTHETIC_N`), and capped CPU thread counts so a typical **CPU-only** environment finishes in a short time. It regenerates `data/synthetic_demo.csv`, runs training, then evaluation, and verifies the full pipeline.

**`configs/demo.yaml`** is the fuller demonstration configuration (closer capacity to the main `demo` experiment). It is intended for manual runs when you have more time or a GPU; it is **not** the default for `examples/quick_test.py`.

Expected **test-set** outputs (values depend on the synthetic draw) include at least:

```text
RMSE
MAE
R2
PI90_coverage
```

Quick-test artifacts (not for version control): `outputs/pgssm_quick_test.pt`, `outputs/metrics_quick_*.json`.

## Main workflow (manual)

```bash
python scripts/generate_synthetic_demo.py   # optional: set PGSSM_QUICK_SYNTHETIC_N=750 for longer series
python src/train.py --config configs/demo.yaml
python src/evaluate.py --config configs/demo.yaml
```

## Data availability note

The original industrial uranium ISL monitoring records are subject to operational confidentiality restrictions and **cannot** be publicly released. This repository provides a **synthetic** demonstration dataset with the **same variable structure** as the field records. The synthetic dataset is **not** intended to reproduce the numerical values reported in the manuscript; it is provided to demonstrate preprocessing, graph construction, training, inference, and evaluation.

## Software requirements

Python 3.10+ recommended. Core dependencies are listed in `requirements.txt` (NumPy, pandas, scikit-learn, PyTorch, PyYAML, Matplotlib).

## License

This repository is released under the MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the associated *Computers & Geosciences* article once it is available, and reference this public repository URL in the **Computer Code Availability** section as required by the journal.
