# Authorized field-analysis workflow

These scripts contain the locked-protocol analysis logic. They do not include confidential field rows.

The official public evidence remains the aggregate files in `field_results/`. Authorized reruns write to `PGSSM_OUTPUT_DIR` and must not overwrite the public manuscript evidence unless a later protocol-identical confirmation is approved.

## Required environment

```bash
export PGSSM_FIELD_DATA_DIR=/protected/path/to/field_data
export PGSSM_OUTPUT_DIR=/protected/path/to/output_root
```

The data directory must contain:

- `five_wells_timeseries_clean.parquet`
- `five_wells_info.csv`

`five_wells_info.csv` must list `well_id` in receiving-row order: extraction well first, then four injectors. Well identifiers remain in the protected files.

## Official locked commands

```bash
python field_analysis/execute_r3_experiments.py
python field_analysis/rolling_origin_audit.py
python field_analysis/inference_parameter_sweep.py
python field_analysis/finalize_analysis_results.py
```

These commands implement:

- 28-day history and 7-day endpoint
- target-date sample assignment
- training-only normalization
- one post-validation train+validation refit
- fixed test parameters
- no test-period model updating

Seed 11 is the locked manuscript seed.

## Not part of the official workflow

- `--expanding` is accepted only as a deprecated alias for the locked target-date protocol. It does not start a test-period parameter update.
- `--strict` is a superseded partition check and is not the manuscript protocol.
- Ridge / Gaussian-GRU / TCN scripts are archived under `archive/legacy_workflow/` and are excluded from the manuscript evidence chain.

## Synthetic reminder

The public synthetic example is an execution check only. It does not replace this authorized field workflow.
