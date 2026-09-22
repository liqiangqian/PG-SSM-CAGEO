# Final-Manuscript Public Repository Alignment Design

## Purpose

Align the public `PG-SSM-CAGEO` repository with the final submission versions of the manuscript, Supplementary Information, and Response to Reviewers while preserving the industrial-data confidentiality boundary.

The repository must support code reproducibility, workflow verification, a synthetic demonstration, and consistency checking of released aggregate evidence. It must not claim or imply independent recomputation of confidential field-derived results from public synthetic data.

## Authoritative scientific baseline

The final submission documents are the sole scientific baseline. The public repository must use the following title and positioning:

> A Physically Motivated Probabilistic Graph State-Space Framework for Short-Term Uranium Concentration Forecasting in Five-Spot In-Situ Leaching Wellfield Units

Required concepts:

- physically motivated probabilistic forecasting;
- topology-informed computational affinity prior;
- flow-modulated graph affinity;
- dual-timescale or dual-branch latent dynamics;
- soft plausibility regularization;
- leakage-safe preprocessing;
- Gaussian probabilistic output.

The repository must not describe PG-SSM as a governing-equation, hydraulic-flow, reactive-transport, or universal uranium-prediction solver. The latent state is a forecasting representation, not a directly measured hydrogeochemical state.

## Public-data boundary

### Public

- source code for synthetic-data generation, preprocessing, graph construction, PG-SSM, training, and evaluation;
- manuscript-aligned configuration files;
- version-pinned environment specifications;
- synthetic demonstration data generated independently of the field archive;
- non-row-level aggregate manuscript-evidence summaries;
- schema, workflow, and usage documentation;
- regression, leakage-safety, and sensitive-artifact tests;
- consistency-check utilities.

### Not public

- raw hydrogeochemical monitoring records;
- industrial operational logs;
- exact site, block, production-zone, or well identifiers;
- actual coordinates or site-derived geometry;
- row-level field observations or predictions;
- protected frozen field-result bundles;
- private adapters, paths, filenames, hashes, or metadata that identify protected sources.

The synthetic data must be described as synthetic in filenames, metadata, code, documentation, and generated output. It is for execution, schema, dimensional, causal-preprocessing, and workflow checks only. It must not be tuned to reproduce field RMSE, R², PI90, trajectories, operating records, or exact field metrics.

## Repository boundary cleanup

The current public tree contains obsolete aggregate evidence, stale scientific narratives, a non-verifiable authorized-field adapter, protected-source hashes, and fixed distances that may be field-derived. The aligned public tree will:

1. remove obsolete aggregate result files and archived legacy scientific narratives from the current branch;
2. remove the public `field_analysis/` adapter because it depends on protected data and exposes internal field schema details that cannot be publicly verified;
3. retain no protected-source hashes or site-derived distances in public configuration or provenance;
4. preserve Git history without rewriting, cleaning, or force pushing;
5. record historical-artifact uncertainty as a manual author check.

Removal from the current branch is recoverable from Git history and is not a history rewrite.

## Target repository structure

```text
configs/
  manuscript_demo.json
data/
  synthetic_demo.csv
docs/
  DATA_BOUNDARY.md
  MANUSCRIPT_ALIGNMENT.md
  USER_GUIDE.md
  VARIABLE_SCHEMA.md
field_results/
  README.md
  manuscript_table2_deterministic.csv
  manuscript_table3_probabilistic.csv
  manuscript_table4_ablation.csv
  manuscript_calibration.csv
  manuscript_stage_coverage.csv
  manuscript_sensitivity.csv
  manuscript_seed_stability.csv
  evidence_manifest.json
  synthetic_results.json
scripts/
  generate_synthetic_demo.py
  run_demo.py
  check_manuscript_consistency.py
  scan_sensitive_artifacts.py
src/
  pgssm_model.py
  preprocessing.py
  stages.py
  evaluation.py
tests/
  test_config_alignment.py
  test_preprocessing.py
  test_stage_causality.py
  test_gaussian_scoring.py
  test_model_smoke.py
  test_evidence_consistency.py
  test_no_sensitive_artifacts.py
README.md
DATA_BOUNDARY.md
RELEASE_NOTES.md
requirements.txt
LICENSE
```

`docs/DATA_BOUNDARY.md` may link to the root `DATA_BOUNDARY.md`; the root file is canonical.

## Synthetic demonstration design

The generator will create a deterministic, generic five-well demonstration with four synthetic injector roles and one central synthetic extraction role. It will not use actual identifiers, coordinates, dates, distances, source hashes, or copied field rows.

The generated daily table will contain:

- a synthetic daily calendar index;
- generic `well_role` values;
- daily injection and extraction flows;
- bounded synthetic pH and dissolved-oxygen signals;
- forward-generated latent uranium concentration;
- mixed-frequency uranium assay observations with missing values on non-assay dates.

Generation rules will use smooth seasonal signals, synthetic step changes, random perturbations, lagged injection-to-extraction response, a smooth latent process, and heteroscedastic noise. The random seed is dedicated to synthetic-data generation and is not presented as a manuscript model-selection seed.

Preprocessing will derive, rather than pre-bake:

- assay-observation mask;
- causal last-observation-carried-forward concentration;
- days-since-assay;
- chronological target-date splits;
- normalization statistics from training rows only;
- samples whose history may cross an earlier partition boundary while ending at the forecast origin;
- target eligibility restricted to assay-observed target dates.

Backward filling, future-target interpolation, and future-assay use are prohibited.

## Manuscript demo configuration

The default public configuration will contain:

- input window `L = 28` days;
- forecast horizon `H = 7` days;
- primary seed `43`;
- repeated-seed stability list `[41, 42, 43, 44, 45]` for PG-SSM, LSTM, TCN, and N-BEATS only;
- `alpha = 1.0` with candidates `[0.5, 1.0, 1.5]`;
- `beta = 1.0` with candidates `[0.5, 1.0, 1.5]`;
- distance-scale candidates `[0.5, 1.0, 2.0]` times the synthetic median inter-well distance;
- `eta_y = 0.04` with candidates `[0.02, 0.04, 0.06]`;
- `tau_Q = 0.60` with candidates `[0.55, 0.60, 0.65]`;
- `Delta_max = 0.80` with candidates `[0.60, 0.80, 1.00]`;
- `lambda_nonneg = 1.0`;
- `lambda_rate = 0.10` with candidates `[0.05, 0.10, 0.20]`;
- `lambda_stage = 0.20` with candidates `[0.10, 0.20, 0.30]`;
- seven-day moving-average trend window;
- three-day ramp-up persistence;
- CPU as the verified public execution device.

The configuration will not contain protected dates, source hashes, real coordinates, site distances, or run-specific CPU, RAM, or operating-system claims.

## Graph and model architecture

The public model will preserve the receiving-row convention:

- node `0` is the central extraction or receiving node;
- nodes `1–4` are synthetic injector nodes;
- injector-to-extraction information is represented by `A[0, j, t]`;
- the graph uses a distance-decay geometric prior, flow modulation, self-loop augmentation, and row normalization;
- `alpha` and `beta` are computational modulation weights, not hydraulic coefficients.

The forward path is:

```text
multi-well inputs
-> graph encoder
-> slow branch + fast branch
-> state fusion
-> Gaussian output head
-> predictive mean + predictive log-variance
```

The slow branch represents delayed concentration memory and cumulative process response. The fast branch represents short-term flow and hydrochemical disturbances. No peak-aware, peak-correction, peak-shift, or peak-displacement component will exist.

The public `PGSSM` class will accept synthetic distances or affinities as input. It must not contain hard-coded site-derived distances.

## Gaussian uncertainty and soft regularization

The Gaussian head returns predictive mean and log-variance. Training uses untruncated Gaussian NLL. Aggregate predictive uncertainty is represented through learned log-variance; process covariance `Q` and observation variance are not separately parameterized.

The 90% interval is `mu ± 1.645 sigma`. Optional operational display clipping may use `max(0, lower_bound)`, but raw NLL, CRPS, PIT, calibration, and other probabilistic scores must use the untruncated Gaussian distribution and unclipped interval parameters.

Soft regularization names are:

- non-negativity;
- rate consistency;
- stage-consistent monotonicity.

The repository must use `soft regularization` or `soft plausibility regularization`; it must not use `Physics loss`, `Physics constraint`, `nophys`, or `w/o Physics` in active code or public outputs.

## Causal stage identification

The stage module will assign:

- Rising / ramp-up;
- Peak-transition;
- Quasi-steady;
- Declining.

All assignments use observations available at or before the forecast origin. Peak-transition is a past-only trend-change rule following ramp-up; it cannot search for future local maxima or use future assay targets. Validation selects thresholds; test execution receives frozen thresholds.

Peak-transition behavior is evaluated using subgroup coverage, residual diagnostics, and rate/stage consistency. A standalone Peak Timing Error is not an active primary metric.

## Evaluation

The public evaluation module will implement:

- deterministic: RMSE, MAE, R², MASE;
- probabilistic: PI coverage, mean interval width, Gaussian NLL, Gaussian CRPS, Winkler score, sharpness, PIT, and multi-level calibration;
- physical-consistency: negative prediction rate, rate violation, and stage violation.

The evaluation API will clearly separate statistical scoring from optional visualization clipping.

Ablation labels are locked to:

- `PG-SSM`;
- `w/o spatial coupling`;
- `w/o Graph`;
- `w/o soft regularization`;
- `w/o Dual-branch`;
- `w/o Graph + soft regularization`;
- `w/o Graph + Dual-branch`.

Baseline-selection documentation will distinguish TFT (`Validation NLL / RMSE`) from DeepAR (`Validation NLL / CRPS`); all other listed baselines use their manuscript-specified validation criteria.

## Aggregate manuscript evidence

Aggregate evidence files will be transcribed from the final submission documents and checked by exact-value tests. They may include final table-level values such as:

- primary `n = 38`, `H = 7`, `L = 28`;
- PG-SSM RMSE `0.455`, MAE `0.357`, R² `0.933`, MASE `0.867`;
- PI90 `34/38 (89.5%)`;
- central interval coverage `50.0%, 63.2%, 76.3%, 89.5%, 100.0%`;
- stage-wise PI90 `8/9`, `7/8`, `10/11`, `9/10`;
- final ablation names and aggregate metrics;
- final one-factor sensitivity and seed-stability summaries.

Evidence files must contain no dates, observations, predictions, residual rows, per-target intervals, actual identifiers, or coordinates. The manifest will label them as transcription-level aggregate manuscript evidence, not public recomputation of field results.

## Documentation

The README will contain:

- overview and scientific scope;
- explicit non-solver boundary;
- repository structure;
- pinned installation instructions for Linux/macOS and Windows activation;
- commands that exist and are verified;
- synthetic-data purpose and exclusions;
- leakage-safe preprocessing;
- model architecture;
- evaluation metrics;
- reproducibility configuration and seeds;
- manuscript-evidence boundary;
- data availability;
- MIT license;
- a citation placeholder without a fabricated DOI.

`DATA_BOUNDARY.md` will provide a concise public/not-public inventory and state `synthetic != field data`.

`docs/MANUSCRIPT_ALIGNMENT.md` will map manuscript items to concrete configuration, source, script, test, and aggregate-evidence files without exposing protected rows.

## Testing strategy

Behavior changes will be developed test-first. Tests will use standard-library `unittest` so the pinned scientific environment does not require a new public dependency.

Required regression coverage:

- no future assay is used;
- target partition follows target date;
- normalization uses training rows only;
- LOCF is causal and never backward-filled;
- only assay-observed targets are scored;
- stage labels are past-only;
- peak-transition does not use future maxima;
- thresholds remain frozen after validation;
- Gaussian NLL is untruncated;
- display clipping cannot change probabilistic scores;
- model forward pass returns finite mean/log-variance with correct shapes;
- custom synthetic geometry is used instead of hard-coded field distances;
- config schema contains every locked final setting;
- aggregate evidence matches the final documents;
- forbidden legacy terminology is absent from active public files;
- generic secret, absolute-path, field-filename, identifier, coordinate-column, and sensitive-extension patterns are rejected.

The verification gate will run:

1. import/dependency check;
2. synthetic generation;
3. preprocessing tests;
4. model forward pass;
5. CPU training smoke test;
6. evaluation smoke test;
7. complete unit-test suite;
8. sensitive-artifact scan;
9. README command replay;
10. config/evidence consistency check;
11. `git diff --check`.

## Git and release handling

- Work is limited to the isolated local clone of `PG-SSM-CAGEO`.
- No reset, clean, force push, history deletion, or tag movement is permitted.
- The implementation will be committed as `refactor: align public workflow with final PG-SSM manuscript` if all required verification passes.
- A normal push of the current branch is allowed only after the final sensitive-data scan and Git status review.
- The stale `v2.0-final-r2` tag will not be overwritten or moved automatically; final release/tag selection remains an author check.
- If later evidence shows that reachable history contains real industrial rows or actual coordinates, normal push stops and the issue is reported without destructive history rewriting.

## Success criteria

The update is complete only when:

1. active public documentation, configuration, aggregate evidence, and tests match the final submission;
2. the public demo is demonstrably causal and synthetic;
3. model and evaluation behavior match the stated architecture and uncertainty boundary;
4. no active public file contains old R1/n=73 evidence, protected-source hashes, hard-coded site geometry, row-level field results, or forbidden legacy terminology;
5. all verification commands pass with recorded output;
6. the resulting commit contains only repository-alignment changes.
