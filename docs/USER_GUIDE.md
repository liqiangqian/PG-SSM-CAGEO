# User guide

## Public synthetic check

The public quick test reads `data/synthetic_five_well.npz`. It contains `x` with shape `(time, well, variable)`, a one-dimensional extraction-well target, and five well distances. Node 0 is the extraction well and nodes 1–4 are injectors.

This example is only an execution, schema, and dimensional check. It is not a field-performance reproduction, not a manuscript RMSE reproduction, and not an external validation set.

```bash
python scripts/generate_synthetic_data.py
python scripts/run_synthetic_example.py
python scripts/check_manuscript_consistency.py
```

The synthetic command writes `field_results/synthetic_results.json`. The expected architecture size is 9,538 trainable parameters.

## Authorized field inputs

Authorized field execution expects `five_wells_timeseries_clean.parquet` and `five_wells_info.csv` in `PGSSM_FIELD_DATA_DIR`. The parquet file must provide five wells and the six variables listed in `VARIABLE_SCHEMA.md`. The role file must provide `well_id` in receiving-row order: extraction well first, then four injectors. Exact well identifiers stay in the protected files and are not published here.

## Official protocol

- History length: 28 days.
- Endpoint horizon: 7 days.
- Sample membership is determined by target date.
- Forecast origin is the target date minus 7 days.
- Predictors use only the 28 origin-available days.
- Normalization uses training-partition statistics only.
- Model selection uses training and validation only.
- Final fitting is one post-validation train+validation refit.
- Test parameters remain fixed; there is no test-period updating.
- Locked seed: 11.
- Locked graph coefficients: `sigma_d = 120 m`, `alpha = 0.60`, `beta = 0.25`.
- Loss weights: negative 0.08, rate 0.05, rising-stage 0.03.

## Outputs

Public aggregate manuscript evidence lives in `field_results/`. Authorized field scripts write private CSV and JSON files under `PGSSM_OUTPUT_DIR`. Those private files must not replace the locked public aggregate evidence unless the author later confirms a full protocol-identical rerun.
