# A Physically Motivated Probabilistic Graph State-Space Framework for Seven-Day Uranium Concentration Forecasting in a Five-Spot In-Situ Leaching Wellfield

This repository accompanies the Computers & Geosciences manuscript:

**A Physically Motivated Probabilistic Graph State-Space Framework for Seven-Day Uranium Concentration Forecasting in a Five-Spot In-Situ Leaching Wellfield**

Manuscript number: `CAGEO-D-26-00782R1`.

The manuscript develops and evaluates PG-SSM as a methodological forecasting framework. PG-SSM is a physically motivated probabilistic graph state-space framework for seven-day uranium-concentration forecasting in a five-spot ISL wellfield.

## Scientific scope

PG-SSM uses:

1. a receiving-row five-spot graph
2. operation-conditioned multi-well information aggregation
3. dual-timescale latent dynamics
4. physically motivated soft plausibility regularization
5. a probabilistic seven-day endpoint output

The graph is a computational information-aggregation prior. It is not a hydraulic-flow solution, a reactive-transport simulator, or a validated mechanistic connectivity map. The soft penalties provide rising-stage and endpoint plausibility guidance; they are not conservation laws or a symmetric three-stage physical law.

Seven-day persistence is the primary matched operational no-change point-forecast control. It is not the research subject of the manuscript.

## Locked field evidence

The public files record the manuscript-locked aggregate results for 73 scored seven-day endpoints.

| Model | RMSE (mg/L) | MAE (mg/L) | R² | MASE | PI90 |
|---|---:|---:|---:|---:|---|
| PG-SSM | 0.2284 | 0.1712 | 0.2275 | 0.5093 | 89.0% (65/73) |
| Persistence | 0.2588 | 0.1960 | −0.1622 | 0.5831 | point-forecast control only |

Additional locked PG-SSM scores: CRPS 0.1732, NLL 0.4381, mean PI90 width 1.1824, Winkler90 1.4237.

Persistence is reported only as a point-forecast control. Persistence CRPS, NLL, PI90, interval width, and Winkler scores are not official manuscript evidence.

The rolling-origin audit shows temporal heterogeneity rather than uniform superiority: PG-SSM RMSE 0.2412 / 0.2826 / 0.1694 versus persistence 0.4411 / 0.2077 / 0.2892 in folds 1–3.

## Reproducibility boundary

**Public**

- code
- configuration
- synthetic data
- aggregate manuscript evidence
- workflow documentation

**Protected**

- row-level industrial monitoring data
- exact site coordinates
- operational records
- field-level row predictions

The public repository supports code inspection, execution, dimensional checks, and verification of the reported aggregate evidence chain. Reproduction of the confidential field analysis requires authorized access to the protected industrial inputs.

## Repository structure

```text
configs/          Locked field-analysis configuration
data/             Deterministic synthetic five-well dataset
docs/             User guide, variable schema, and data-boundary notes
field_analysis/   Authorized field-analysis scripts; require protected data
field_results/    Aggregate manuscript evidence and synthetic execution check
scripts/          Synthetic generator, quick test, and consistency checker
src/              PG-SSM architecture and loss implementation
archive/          Superseded development artifacts excluded from the evidence chain
```

Legacy development outputs, including earlier revision metrics and unmatched external-model trials, are excluded from the manuscript evidence chain.

## Installation

Python 3.9.13 was used for the verified synthetic run.

```bash
git clone https://github.com/liqiangqian/PG-SSM-CAGEO.git
cd PG-SSM-CAGEO
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Public commands

```bash
python scripts/generate_synthetic_data.py
python scripts/run_synthetic_example.py
python scripts/check_manuscript_consistency.py
```

The synthetic example writes `field_results/synthetic_results.json`. A successful run reports 120/33/33 train/validation/test windows and 9,538 trainable parameters. Synthetic scores are an execution and dimensional check only. They are not field-performance scores and do not reproduce manuscript RMSE.

The consistency checker verifies that README, configuration, manifest, and aggregate evidence files remain aligned with the locked manuscript record.

## Authorized field workflow

Place the protected daily record and well-role file in a directory that is not part of this public repository, then set:

```bash
export PGSSM_FIELD_DATA_DIR=/protected/path/to/field_data
export PGSSM_OUTPUT_DIR=/protected/path/to/output_root
```

On Windows PowerShell:

```powershell
$env:PGSSM_FIELD_DATA_DIR = "D:\protected\field_data"
$env:PGSSM_OUTPUT_DIR = "D:\protected\field_results"
```

Then run the locked protocol documented in `field_analysis/README.md`:

```bash
python field_analysis/execute_r3_experiments.py
python field_analysis/rolling_origin_audit.py
python field_analysis/inference_parameter_sweep.py
python field_analysis/finalize_analysis_results.py
```

Expected source SHA-256 values are stored in `configs/field_analysis_configuration.json` and `field_results/analysis_manifest_final.json`. Authorized outputs must remain in the protected output directory. They must not be copied into the public `field_results/` tree.

## License

MIT License. See `LICENSE`.

## Citation

Please cite the associated Computers & Geosciences manuscript when available.
