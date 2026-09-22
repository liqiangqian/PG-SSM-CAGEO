# User guide

## Run the public workflow

Create the synthetic mixed-frequency data:

```bash
python scripts/generate_synthetic_demo.py
```

Run a short CPU training and evaluation check:

```bash
python scripts/run_demo.py --config configs/manuscript_demo.json --epochs 2
```

Validate configuration/evidence and the public data boundary:

```bash
python scripts/check_manuscript_consistency.py
python scripts/scan_sensitive_artifacts.py
python -m unittest discover -s tests -v
```

## What the demo verifies

- daily five-role input structure;
- mixed-frequency assay masking;
- causal LOCF and days-since-assay;
- training-only normalization and target-day partitioning;
- receiving-row graph message passing;
- slow/fast branch fusion and Gaussian output;
- CPU training, untruncated probabilistic scoring, and consistency metrics.

## What the demo does not verify

The demo does not reproduce site-specific values, confidential trajectories, exact industrial operating records, or field-derived manuscript metrics. Its `synthetic_results.json` output must not be substituted for aggregate manuscript evidence.

## Configuration

`configs/manuscript_demo.json` contains the final public defaults and candidate grids. Geometry is supplied to the model at runtime; there are no embedded site-derived distances.
