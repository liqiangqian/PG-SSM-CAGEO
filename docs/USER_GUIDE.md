# User guide

## Inputs

The public quick test reads `data/synthetic_five_well.npz`. It contains `x` with shape `(time, well, variable)`, a one-dimensional extraction-well target and five well distances. Node 0 is the extraction well and nodes 1–4 are injectors.

Authorized field execution expects `five_wells_timeseries_clean.parquet` and `five_wells_info.csv`. The parquet file must provide five wells and the six variables listed in `VARIABLE_SCHEMA.md`; the coordinate file must provide `well_id`, `X` and `Y`.

## Outputs

The synthetic command writes a JSON file containing window counts, RMSE, MASE, mean normalized incoming weight and parameter count. Field scripts write CSV and JSON audit files under the directory selected by `PGSSM_OUTPUT_DIR`.

## Fixed conventions

- History length: 28 days.
- Endpoint horizon: 7 days.
- Receiving node: extraction well at index 0.
- Incoming injector edges are normalized jointly with a unit centre self-loop.
- Training-only normalization is used.
- Seed 11 is the prespecified cross-architecture comparison.
- Two terminal target dates are masked in the field evaluation.

## Expected behaviour

The quick test checks execution, tensor dimensions and architecture size. Its synthetic accuracy is not expected to equal the confidential field result.
