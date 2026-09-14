# Matched baseline reconciliation

**NOT USED IN CURRENT MANUSCRIPT**

Decision: **CASE B — development / legacy outputs**.

Status: excluded from the manuscript evidence chain.

## Why these files are not official manuscript evidence

The locked paper uses seven-day persistence as the primary matched operational no-change point-forecast control. Reviewer 3 Comment 17 is answered by methodological positioning of external model families, not by a completed same-protocol Ridge / Gaussian-GRU / TCN benchmark.

The archived files `matched_baselines.csv` and `matched_baselines_summary.csv` report:

- Gaussian-GRU RMSE about 0.50–0.51
- TCN RMSE about 0.52–0.69
- Ridge RMSE about 0.91

Those scores belong to the earlier revision output family whose full PG-SSM RMSE was about 0.4156. They do not correspond to the locked manuscript PG-SSM RMSE of 0.2284. They therefore cannot be treated as FINAL PAPER matched quantitative benchmarks.

## Protocol check

The archived script `archive/legacy_workflow/matched_baselines_final.py` uses a 28-day history and a 7-day endpoint, but it is tied to the superseded public result files rather than to a confirmed protocol-identical rerun of the locked 73-endpoint record. No author-confirmed same-population, one-refit, fixed-test output was available for Ridge / GRU / TCN under the final paper protocol.

## Action taken

- Moved `field_results/matched_baselines.csv` and `matched_baselines_summary.csv` to `archive/development_baselines/`
- Moved `field_analysis/matched_baselines_final.py` to `archive/legacy_workflow/`
- Left the files available for provenance
- Did not add them to the official public evidence chain
- Did not change the paper to absorb these numbers

If a later authorized rerun produces protocol-identical matched baselines, they should be reviewed as a paper-integration candidate before any public promotion.
