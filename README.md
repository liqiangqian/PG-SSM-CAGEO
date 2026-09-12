# PG-SSM-CAGEO revision companion

This repository accompanies the manuscript:

**When a Graph Forecast Does Not Surpass Persistence: A Provenance-Traceable Five-Spot Uranium In-Situ Leaching Case Study**

It exposes the corrected receiving-row graph, the 9,538-parameter endpoint architecture, original-scale soft penalties, deterministic synthetic data, aggregate field results and the field-analysis workflow used in the revision.

## Scientific scope

The revised study reports a negative result. In the locked 73-endpoint test, PG-SSM did not outperform seven-day persistence and its 90% predictive interval was undercovered. The graph is a computational information-aggregation prior. It is not a hydraulic-flow solution, reactive-transport simulator or validated mechanistic model.

The industrial field rows and exact site coordinates cannot be redistributed. The public synthetic example verifies execution and dimensional consistency. Aggregate field outputs and source hashes support provenance inspection, but the confidential field result cannot be independently replayed without authorized access to the source data.

## Repository structure

```text
configs/          Locked field-analysis configuration without field rows
data/             Deterministic synthetic five-well dataset
docs/             User guide, variable schema and data-limit explanation
field_analysis/   Portable field-analysis scripts; require authorized data
field_results/    Aggregate field outputs and synthetic quick-test result
scripts/          Synthetic-data generator and executable quick test
src/              Corrected PG-SSM architecture and loss implementation
```

## Installation

Python 3.9.13 was used for the verified synthetic run.

```bash
git clone https://github.com/liqiangqian/PG-SSM-CAGEO.git
cd PG-SSM-CAGEO
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick test

```bash
python scripts/generate_synthetic_data.py
python scripts/run_synthetic_example.py
```

The second command writes `field_results/synthetic_results.json`. A successful run reports 120/33/33 train/validation/test windows and 9,538 trainable parameters. Synthetic scores are not manuscript field scores.

## Field workflow for authorized data holders

Place `five_wells_timeseries_clean.parquet` and `five_wells_info.csv` in a protected directory, then set:

```bash
export PGSSM_FIELD_DATA_DIR=/protected/path/to/field_data
export PGSSM_OUTPUT_DIR=/protected/path/to/output_root
```

On Windows PowerShell:

```powershell
$env:PGSSM_FIELD_DATA_DIR = "D:\protected\field_data"
$env:PGSSM_OUTPUT_DIR = "D:\protected\field_results"
```

Run the scripts in the order documented in `field_analysis/README.md`. The expected source SHA-256 is stored in `configs/field_analysis_configuration.json` and `field_results/analysis_manifest_final.json`.

## Reported field evidence

`field_results/` contains aggregate metrics, calibration, block-bootstrap, stage, rolling-origin, matched-baseline and coefficient-sensitivity outputs. Row-level field observations and predictions are excluded because the data-owner agreement does not permit public redistribution.

## License

MIT License. See `LICENSE`.

## Citation

Please cite the associated Computers & Geosciences manuscript when available.
