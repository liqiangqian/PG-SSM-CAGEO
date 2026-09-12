# Field-analysis workflow

These scripts contain the analysis logic used for the revision. They do not include confidential field rows.

Set `PGSSM_FIELD_DATA_DIR` to a directory containing the verified parquet and coordinate files. Set `PGSSM_OUTPUT_DIR` to a protected output root. Then run:

```bash
python field_analysis/execute_r3_experiments.py --expanding
python field_analysis/execute_r3_experiments.py
python field_analysis/matched_baselines_final.py
python field_analysis/rolling_origin_audit.py
python field_analysis/inference_parameter_sweep.py
python field_analysis/finalize_analysis_results.py
python field_analysis/finalize_analysis_results.py --strict
```

To rebuild figures from authorized field outputs, optionally set `PGSSM_FIGURE_DIR` and run `python field_analysis/generate_final_figures.py`.

Seed 11 is the prespecified cross-architecture comparison. Seeds 23 and 47 are retained only as neural optimization diagnostics. The public aggregate files are provided for provenance comparison.
