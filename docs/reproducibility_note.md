# Reproducibility Note

This repository is designed for transparent workflow verification of the public PG-SSM implementation.

The confidential industrial monitoring archive used in the manuscript cannot be publicly released. The synthetic dataset in this repository has a similar general variable structure, but it does not contain real industrial measurements, real well identifiers, real coordinates, real operating logs, or site-specific uranium concentration records.

The synthetic dataset cannot reproduce the exact values or performance metrics reported in the manuscript. Its purpose is to let editors, reviewers, and readers execute the computational workflow: preprocessing, graph construction, model training, evaluation, calibration diagnostics, ablation checks, rolling-origin checks, multi-horizon checks, and bootstrap confidence-interval estimation.

Independent field data are required for external validation. The public synthetic workflow should not be interpreted as independent multi-site validation.

The PG-SSM implementation is physically motivated, but it is not a full advection-dispersion-reaction solver. The graph adjacency should be interpreted as a computational affinity prior, not a solved hydraulic-flow field.
