# Public data boundary

## Public

- source code for preprocessing, graph construction, PG-SSM, training, and evaluation;
- manuscript-aligned public configuration;
- pinned environment specifications;
- independently generated synthetic demonstration data;
- aggregate manuscript-evidence summaries;
- schema and workflow documentation;
- regression, consistency, and safety tests.

## Not public

- raw hydrogeochemical monitoring records;
- industrial operational logs;
- exact site, block, or production-zone information;
- actual well identifiers or coordinates;
- row-level field observations or predictions;
- the protected frozen field-result bundle;
- private adapters, source paths, hashes, or metadata identifying protected inputs.

## Interpretation

Synthetic data are not field data. They support execution, schema, dimensional, causal-preprocessing, and workflow checks only. They must not be used to reconstruct real industrial records or presented as an independent recomputation of confidential field-derived manuscript results.

The CSV/JSON evidence in `field_results/` is aggregate manuscript transcription. It contains no target-level records.
