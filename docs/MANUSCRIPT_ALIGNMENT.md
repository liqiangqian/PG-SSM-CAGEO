# Final manuscript alignment

| Manuscript item | Public repository implementation |
|---|---|
| Title and scientific scope | `README.md` |
| `L = 28`, `H = 7`, seed `43` | `configs/manuscript_demo.json` |
| Seeds 41–45 scope | config and `field_results/manuscript_seed_stability.csv` |
| Causal assay handling | `src/preprocessing.py` and `tests/test_preprocessing.py` |
| Observation mask and days-since-assay | `src/preprocessing.py` |
| Target-day partitioning | `src/preprocessing.py` |
| Training-only normalization | `src/preprocessing.py` |
| Receiving-row dynamic graph | `src/pgssm_model.py` |
| Training-only min-max flow scaling and z-score scaling for other features | `src/preprocessing.py` |
| `alpha = beta = 1.0` and candidate grids | `configs/manuscript_demo.json` |
| Hidden size 64, dropout 0.10, additive slow/fast fusion, and Gaussian head | `configs/manuscript_demo.json` and `src/pgssm_model.py` |
| Ramp-up-only stage regularization and complete Gaussian NLL | `src/pgssm_model.py` |
| Four causal stages | `src/stages.py` and `tests/test_stage_causality.py` |
| Untruncated Gaussian scoring | `src/evaluation.py` and `tests/test_gaussian_scoring.py` |
| Deterministic/probabilistic/consistency metrics | `src/evaluation.py` |
| Tables 2–4 aggregate evidence | `field_results/manuscript_table*.csv` |
| Calibration and stage coverage | `field_results/manuscript_calibration.csv` and `manuscript_stage_coverage.csv` |
| Sensitivity and repeated-seed summaries | `field_results/manuscript_sensitivity.csv` and `manuscript_seed_stability.csv` |
| Public confidentiality boundary | `DATA_BOUNDARY.md` |
| Automated evidence check | `scripts/check_manuscript_consistency.py` |
| Automated safety check | `scripts/scan_sensitive_artifacts.py` |

The implementation mapping does not expose protected records and does not claim that the synthetic demo recomputes field-derived results.
