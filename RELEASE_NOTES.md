# Release notes: v2.0-final-r2

This release synchronizes the public repository with the locked CAGEO-D-26-00782R1 manuscript.

## Changes

- Synchronized the repository with the final R2 manuscript
- Corrected the manuscript title and scientific scope
- Reconciled the endpoint protocol: 28-day history, 7-day horizon, 73 scored endpoints, one post-validation train+validation refit, and fixed test parameters
- Reconciled aggregate manuscript metrics, calibration, rolling-origin, bootstrap, and sensitivity evidence
- Clarified the public/protected evidence boundary
- Archived superseded development artifacts, including earlier revision outputs and unmatched external-model trials
- Updated reproducibility documentation and added `scripts/check_manuscript_consistency.py`

## Evidence rule

Public `field_results/` files are aggregate manuscript results. They are not invented row-level field predictions.

## Archived material

Previous internal revision artifacts remain in `archive/` for provenance and are excluded from the manuscript evidence chain.
