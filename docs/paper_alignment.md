# Alignment with the revised manuscript (PG-SSM)

**Repository:** [https://github.com/liqiangqian/PG-SSM-CAGEO](https://github.com/liqiangqian/PG-SSM-CAGEO)

This public repository implements the **same core methodological components** described in the associated manuscript *A Physically Motivated Probabilistic Graph State-Space Framework for Short-Term Uranium Concentration Forecasting in Five-Spot In-Situ Leaching Wellfield Units* (submitted work; not a published journal article at the time of this repository snapshot):

1. **Flow-aware topology affinity prior**
   Row-normalized adjacency over the five-spot graph (central extraction well + four injection wells). Injection-side edge weights can be modulated by recent injection-flow proxies (`graph.flow_aware` in `configs/demo.yaml`). This graph is a computational affinity prior, not a solved hydraulic-flow field.

2. **Dual-branch temporal encoder**
   A dilated **TCN** processes fast hydrodynamic drivers; an **LSTM** encodes slower chemical covariates together with low-dimensional static attributes (placeholder coordinates in the demo).

3. **Gated fusion**
   A sigmoid gate fuses fast and slow latent trajectories before graph readout (same role as the parallel gating / dual-timescale mechanism in the paper).

4. **Graph readout**
   Learned aggregation over injector neighbors using the topology prior weights.

5. **Probabilistic output**
   Gaussian predictive head (mean + log-variance). Metrics include **90% predictive interval coverage** after mapping uncertainty back to physical units via the target scaler.

6. **Soft physical-plausibility regularization (training loss)**
   Soft penalties aligned with the revised supplementary specification: `delta_max` (rate surrogate on batch-ordered means), `lambda_mass`, `lambda_mono`, and `lambda_smooth` (hidden-state temporal smoothness). Defaults match the stable operating point documented in Supplementary Table S3 (`delta_max=0.20`, `lambda_mass=lambda_mono=0.01`, `alpha=0.2`).

## Relation to the major-revision experiments

The revised manuscript and Supplementary Information report field-data analyses that cannot be reproduced from the public synthetic dataset, including multi-horizon evaluation, rolling-origin stress tests, probabilistic calibration diagnostics, residual diagnostics, physical-consistency diagnostics, component-removal/ablation studies, bootstrap confidence intervals, and additional anonymized archive boundary tests.

This repository supports the same computational workflow on a synthetic demonstration dataset:

| Manuscript item | Public repository support | Scope on synthetic data |
| --- | --- | --- |
| Preprocessing and chronological splitting | `src/preprocessing.py`, `configs/*.yaml` | Executable workflow check |
| Five-spot graph construction | `src/graph_builder.py` | Demonstrates topology and flow-aware affinity |
| PG-SSM training | `src/train.py`, `src/model.py` | Demonstrates model fitting and validation checkpointing |
| Probabilistic output | `src/model.py`, `src/metrics.py` | Demonstrates Gaussian mean/log-variance and PI90 coverage |
| Evaluation script | `src/evaluate.py` | Reports deterministic metrics and PI90 on synthetic test data |
| Quick-test example | `examples/quick_test.py` | CPU-oriented smoke test for editors/reviewers |

The synthetic quick test is therefore a workflow-verification example. It should not be interpreted as independent multi-site validation, a reproduction of the confidential field-data tables, or evidence that PG-SSM is a site-universal uranium concentration predictor.

## What is *not* reproduced here

- **Confidential field time series** and unit-identifying metadata.
- **Exact tabulated metrics** from the main text and Supplementary Information, which were obtained on confidential field archives, larger training budgets, and manuscript-specific data curation (gap handling threshold *k*=21 days, rolling-origin windows, bootstrap resampling, and additional-archive screening).
- **Reactive-transport simulation** or site-calibrated hydraulic-flow solution. The public code implements the short-term forecasting framework, not a governing-equation solver.

The **synthetic** CSV is provided solely so editors and reviewers can execute the pipeline: preprocessing → graph construction → training → evaluation.

For major-revision workflow entry points, see `docs/revision_experiments.md`. For reproducibility limitations, see `docs/reproducibility_note.md`.
