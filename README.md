# PG-SSM

Public companion repository for:

**A Physically Motivated Probabilistic Graph State-Space Framework for Short-Term Uranium Concentration Forecasting in Five-Spot In-Situ Leaching Wellfield Units**

## Overview

PG-SSM is a physically motivated probabilistic forecasting framework combining a topology-informed computational affinity prior, flow-modulated graph affinity, dual-timescale latent dynamics, soft plausibility regularization, leakage-safe preprocessing, and a Gaussian probabilistic output.

## Scientific scope

The receiving-row graph aggregates information from four injector roles toward one central extraction role. Its `alpha` and `beta` settings are computational modulation weights. The graph is not a calibrated hydraulic connectivity map, and PG-SSM is not a governing-equation transport, hydraulic-flow, or reactive-transport solver.

The latent state is a forecasting representation. The public implementation does not explicitly parameterize site-specific permeability, porosity, dispersion, hydraulic gradients, mineral composition, or reaction-rate constants.

## Repository structure

```text
configs/        Manuscript-aligned public demonstration configuration
data/           Independently generated synthetic mixed-frequency records
docs/           User guide, variable schema, alignment map, and data boundary
field_results/  Aggregate manuscript transcriptions plus synthetic run summary
scripts/        Generator, CPU demo, consistency checker, and safety scanner
src/            Preprocessing, stage logic, model, and evaluation modules
tests/          Causality, scoring, configuration, evidence, and safety tests
```

## Installation

The pinned public environment is Python 3.9.13, PyTorch 2.2.2, NumPy 1.26.4, pandas 2.3.2, scikit-learn 1.6.1, SciPy 1.13.1, PyArrow 21.0.0, and Matplotlib 3.8.4. The verified public workflow executes on CPU; CUDA is not required.

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

From the repository root:

```bash
python scripts/generate_synthetic_demo.py
python scripts/run_demo.py --config configs/manuscript_demo.json --epochs 2
python scripts/check_manuscript_consistency.py
python scripts/scan_sensitive_artifacts.py
python -m unittest discover -s tests -v
```

The demo writes `field_results/synthetic_results.json`. Its values are demonstration outputs for code execution and workflow verification only.

## Synthetic demonstration data

`data/synthetic_demo.csv` is generated forward in time from generic smooth processes, step changes, lagged operational forcing, and heteroscedastic noise. It contains a daily calendar index, daily operational variables, mixed-frequency assays, and an explicit synthetic marker.

The synthetic data are intended for code execution, schema checks, dimensional checks, causal preprocessing verification, and workflow verification. They are not intended to reproduce site-specific numerical values, confidential trajectories, industrial operating records, or exact field-derived manuscript metrics.

## Leakage-safe preprocessing

The public pipeline retains an assay-observation mask and days-since-assay feature. Historical assay inputs use causal last-observation-carried-forward only: no backward filling, future-target interpolation, or future assay is permitted. Normalization is fitted on training rows only, sample partition is determined by target day, histories may cross an earlier partition boundary, and scoring is restricted to assay-observed target days.

Flow variables use training-only min-max scaling and enter the multiplicative graph modulation directly. Other model features use training-only z-score scaling. The synthetic reference configuration follows the manuscript-selected architecture settings: hidden size 64, dropout 0.10, additive slow/fast fusion, and Adam learning rate `1e-3`.

## Model architecture

```text
multi-well inputs
-> receiving-row dynamic graph encoder
-> slow branch + fast branch
-> state fusion
-> Gaussian output head
-> predictive mean + predictive log-variance
```

The slow branch represents delayed concentration memory and cumulative process response. The fast branch represents short-term flow and hydrochemical disturbances.

The Gaussian head represents aggregate predictive uncertainty through learned predictive log-variance; process noise covariance `Q` and observation variance are not separately parameterized. The 90% interval is `mu +/- 1.645 sigma`. Training and probabilistic scoring use the untruncated Gaussian. Optional non-negative lower-bound clipping is restricted to operational visualization.

Soft plausibility regularization comprises non-negativity, rate consistency, and ramp-up monotonicity. The trend moving-average window is `M = 7` days and the ramp-up persistence is `k = 3` days; the thresholds are `tau_Q = 0.60`, `tau_s = 0.0`, `eta_y = 0.04`, and `Delta_max = 0.80`. Causal stages are Rising, Peak-transition, Quasi-steady, and Declining. The stage loss is applied to ramp-up samples only. Peak-transition behavior is evaluated through subgroup coverage, residual diagnostics, and rate/stage consistency rather than a standalone scalar timing metric.

## Evaluation

- Deterministic: RMSE, MAE, R², MASE.
- Probabilistic: interval coverage, mean interval width, NLL, CRPS, Winkler score, sharpness, PIT, and multi-level calibration.
- Physical consistency: Negative prediction rate, Rate violation, and Stage violation.

Baseline selection is recorded in `configs/manuscript_demo.json`. In particular, TFT uses Validation NLL / RMSE, whereas DeepAR uses Validation NLL / CRPS. The other listed baselines use their specified validation RMSE criteria.

## Reproducibility configuration

The default public configuration uses `L = 28`, `H = 7`, primary seed `43`, and graph weights `alpha = beta = 1.0`. Repeated-seed stability for seeds 41–45 is reported only for PG-SSM, LSTM, TCN, and N-BEATS.

Gaussian NLL uses the complete untruncated Gaussian expression, including `log(2*pi)`. MASE uses the training-period 7-day naive scaling denominator.

## Manuscript evidence boundary

The repository supports code reproducibility, schema and dimensional checks, workflow verification, and consistency checking of released aggregate evidence.

Independent recomputation of confidential field-derived Tables 2–4, Figs. 5–7, residual diagnostics, and field sensitivity analyses requires authorized access to the protected industrial inputs and frozen field-level result bundle. Public synthetic data are not manuscript field data, and running the demo does not reproduce exact manuscript results.

See [DATA_BOUNDARY.md](DATA_BOUNDARY.md) for the full public/not-public inventory.

## Data availability

The public repository provides source code, configuration, pinned environment specifications, synthetic demonstration data, aggregate manuscript-evidence summaries, documentation, tests, and consistency utilities. Raw hydrogeochemical monitoring records and industrial operational logs remain confidential under the applicable data-use restrictions.

## License

MIT License. See [LICENSE](LICENSE).

## Citation

Citation information will be updated after publication. No DOI has been assigned in this repository.
