# Alignment with the manuscript (PG-SSM)

**Repository:** [https://github.com/liqiangqian/PG-SSM-CAGEO](https://github.com/liqiangqian/PG-SSM-CAGEO)

This public repository implements the **same methodological components** described in *A Physics-Guided Graph State-Space Model for Probabilistic Forecasting of Uranium Concentration Dynamics in In-Situ Leaching Wellfields* (Computers & Geosciences submission):

1. **Flow-aware topology prior**  
   Row-normalized adjacency over the five-spot graph (central producer + four injectors). Injection-side edge weights can be modulated by recent injection-flow proxies (`graph.flow_aware` in `configs/demo.yaml`).

2. **Dual-branch temporal encoder**  
   A dilated **TCN** processes fast hydrodynamic drivers; an **LSTM** encodes slower chemical covariates together with low-dimensional static attributes (placeholder coordinates in the demo).

3. **Gated fusion**  
   A sigmoid gate fuses fast and slow latent trajectories before graph readout (same role as the parallel gating / dual-timescale mechanism in the paper).

4. **Graph readout**  
   Learned aggregation over injector neighbors using the topology prior weights.

5. **Probabilistic output**  
   Gaussian predictive head (mean + log-variance). Metrics include **90% predictive interval coverage** after mapping uncertainty back to physical units via the target scaler.

6. **Physics-guided regularization (training loss)**  
   Soft penalties aligned with the revised supplementary specification: `delta_max` (rate surrogate on batch-ordered means), `lambda_mass`, `lambda_mono`, and `lambda_smooth` (hidden-state temporal smoothness). Defaults match the stable operating point documented in Supplementary Table S3 (`delta_max=0.20`, `lambda_mass=lambda_mono=0.01`, `alpha=0.2`).

## What is *not* reproduced here

- **Confidential field time series** and unit-identifying metadata.  
- **Exact tabulated metrics** from the main text, which were obtained on the proprietary dataset, larger training budgets, and manuscript-specific data curation (gap handling threshold *k*=21 days, etc.).

The **synthetic** CSV is provided solely so editors and reviewers can execute the pipeline: preprocessing → graph construction → training → evaluation.
