# Repository–manuscript reconciliation report

Manuscript: `CAGEO-D-26-00782R1`  
Repository: https://github.com/liqiangqian/PG-SSM-CAGEO  
Release name: `v2.0-final-r2`  
Report date: 2026-09-14

## 1. README modification

**Before.** The public README used the superseded title *When a Graph Forecast Does Not Surpass Persistence...* and described a negative result in which PG-SSM did not outperform seven-day persistence.

**After.** The README uses the locked manuscript title:

*A Physically Motivated Probabilistic Graph State-Space Framework for Seven-Day Uranium Concentration Forecasting in a Five-Spot In-Situ Leaching Wellfield*

It presents PG-SSM as a methodological Research Article framework, reports the locked headline scores (PG-SSM RMSE 0.2284 vs persistence 0.2588; MAE 0.1712 vs 0.1960; PI90 89.0%, 65/73), and states the public/protected reproducibility boundary.

## 2. Identified legacy files

- `README.md` (previous negative-result narrative)
- `field_results/final_metrics_valid.csv` (full PG-SSM RMSE ≈ 0.4156)
- `field_results/final_calibration_valid.csv`
- `field_results/final_stage_metrics.csv`
- `field_results/final_paired_block_bootstrap.csv` (sign-reversed contrasts)
- `field_results/final_diagnostics.json`
- `field_results/analysis_manifest_final.json` (`expanding refit` terminology)
- `field_results/inference_parameter_sweep.csv` (superseded RMSE grid)
- `field_results/rolling_origin_metrics.csv` (superseded PG-SSM fold RMSEs)
- `field_results/matched_baselines.csv`
- `field_results/matched_baselines_summary.csv`
- `field_analysis/matched_baselines_final.py`
- `field_analysis/generate_final_figures.py`
- root `MANIFEST.json` (stale inventory including `__pycache__`)

## 3. Files moved into archive

| Destination | Contents |
|---|---|
| `archive/obsolete_field_results/` | Superseded aggregate CSVs/JSON, including the 0.4156-family metrics |
| `archive/development_baselines/` | Ridge / Gaussian-GRU / TCN development outputs |
| `archive/legacy_workflow/` | `matched_baselines_final.py`, `generate_final_figures.py` |
| `archive/legacy_protocol/` | Previous negative-result README |
| `archive/MATCHED_BASELINE_RECONCILIATION.md` | CASE B decision record |

Well identifiers were removed from archived public files before release.

## 4. Final `field_results/` file list

- `README.md`
- `manuscript_primary_metrics.csv`
- `manuscript_calibration.csv`
- `manuscript_stage_coverage.csv`
- `manuscript_rolling_origin.csv`
- `manuscript_bootstrap.csv`
- `manuscript_sensitivity_summary.csv`
- `manuscript_component_variants.csv`
- `analysis_manifest_final.json`
- `provenance.json`
- `synthetic_results.json`

All manuscript CSVs are labelled `aggregate_manuscript_result`. No 73-row confidential prediction file was invented.

## 5. Manifest protocol

`field_results/analysis_manifest_final.json` now records:

- dates 2023-06-06 to 2024-10-13
- 496 calendar positions
- L = 28, H = 7
- train 347 / validation 74 / test target dates 75 / scored endpoints 73
- missing dates 2024-10-12 and 2024-10-13
- target-date sample assignment
- forecast origin = target date minus 7 days
- 28 origin-available history days
- training-only normalization
- training+validation model selection
- one train+validation refit after selection
- no test update; fixed test parameters

`expanding refit` and `expanding_origin_73_endpoints` were removed from the official evidence chain.

## 6. Configs consistency

`configs/field_analysis_configuration.json` now locks:

- L = 28, H = 7
- `sigma_d = 120`, `alpha = 0.60`, `beta = 0.25`
- loss weights 0.08 / 0.05 / 0.03 (negative / rate / rising-stage)
- Adam, lr 0.002, weight decay 0.0001, clip 2.0, epochs 300, patience 40
- locked seed 11
- one post-validation refit and fixed test parameters

The code alias `lambda_stage` is documented as the rising-stage plausibility weight.

## 7. Scripts consistency

Official public commands exist and are referenced by the README:

- `scripts/generate_synthetic_data.py`
- `scripts/run_synthetic_example.py`
- `scripts/check_manuscript_consistency.py`

Official authorized commands:

- `field_analysis/execute_r3_experiments.py`
- `field_analysis/rolling_origin_audit.py`
- `field_analysis/inference_parameter_sweep.py`
- `field_analysis/finalize_analysis_results.py`

The locked field script now uses target-date windows, one train+validation refit, fixed test parameters, and anonymized well-role loading. `--expanding` is only a deprecated warning alias and does not start test-period updating.

## 8. Matched baseline status

**CASE B / RESOLVED.**

Ridge / Gaussian-GRU / TCN outputs are development/legacy artifacts from the superseded 0.4156-family revision. They are not official FINAL PAPER matched benchmarks. This matches Reviewer 3 Comment 17: external model families are methodological positioning, not a completed same-protocol benchmark. Persistence remains the only official matched operational point-forecast control.

## 9. Synthetic-data boundary

The synthetic example remains. README, docs, script docstrings, and `synthetic_results.json` state that it is an execution / schema / dimensional check only. It is not used for field-performance reproduction, manuscript RMSE reproduction, or external validation.

## 10. Repository URL

https://github.com/liqiangqian/PG-SSM-CAGEO

## 11. Final commit SHA

`25e3c7d6571ed172efcd2f871feb320a3237e2f4`

## 12. Consistency checker result

```text
python scripts/check_manuscript_consistency.py
MANUSCRIPT CONSISTENCY: PASS
```

The checker verified 73 endpoints, L=28, H=7, locked PG-SSM and persistence scores, PI90 65/73, sensitivity 0.2214–0.2768, rolling-origin values, bootstrap values, missing dates, and the absence of the forbidden README phrases.

## Gate checklist

| Gate | Result |
|---|---|
| README vs Manuscript | PASS |
| Protocol vs Manuscript | PASS |
| Primary metrics | PASS |
| Calibration | PASS |
| Rolling origin | PASS |
| Bootstrap | PASS |
| Sensitivity | PASS |
| Matched baseline status | RESOLVED |
| Legacy outputs | ARCHIVED |
| Protected-data boundary | PASS |
| Repository reproducibility | PASS |
| Reviewer-facing consistency | PASS |

## Final statement

FINAL REPOSITORY–MANUSCRIPT CONSISTENCY: PASS

READY TO TAG FINAL RELEASE
