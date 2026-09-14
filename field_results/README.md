# Manuscript evidence

This directory is the public aggregate evidence chain for Computers & Geosciences manuscript CAGEO-D-26-00782R1.

Every file here records **manuscript-locked aggregate results**. None of these files is a row-level field prediction file, and none was fabricated by inventing 73 confidential endpoints.

## Official files

| File | Role |
|---|---|
| `manuscript_primary_metrics.csv` | Locked point and probabilistic scores for PG-SSM and persistence |
| `manuscript_calibration.csv` | Multi-level empirical coverage for PG-SSM |
| `manuscript_stage_coverage.csv` | Stage-wise PI90 coverage |
| `manuscript_rolling_origin.csv` | Three-fold rolling-origin RMSE |
| `manuscript_bootstrap.csv` | Paired seven-day moving-block bootstrap contrasts |
| `manuscript_sensitivity_summary.csv` | Inference-stage coefficient-sensitivity bounds |
| `manuscript_component_variants.csv` | Within-framework component and graph variants |
| `analysis_manifest_final.json` | Locked protocol, dates, and evaluation rules |
| `provenance.json` | Source-hash and evidence-file mapping |
| `synthetic_results.json` | Public synthetic execution check only |

## What is not here

- Confidential industrial rows and dated field predictions
- Exact site coordinates
- Persistence CRPS / NLL / PI90 / width / Winkler as official manuscript evidence
- Ridge / Gaussian-GRU / TCN outputs as official matched benchmarks

Superseded development artifacts are stored under `archive/` and are excluded from the manuscript evidence chain.

## Synthetic file

`synthetic_results.json` is produced by `scripts/run_synthetic_example.py`. It checks execution, schema, and tensor dimensions. It does not reproduce field RMSE, MAE, coverage, or any other manuscript field metric.
