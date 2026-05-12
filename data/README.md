# Demonstration data

This folder contains a **synthetic** demonstration dataset for running the PG-SSM workflow end-to-end.

The original industrial uranium in-situ leaching (ISL) monitoring records used in the manuscript **cannot** be publicly released because they are subject to operational confidentiality restrictions. The synthetic dataset follows the **same variable schema** as the field records: injection-flow proxies for four injection wells, a production-flow proxy, chemical indicators (pH and dissolved oxygen) on injectors, and production-well uranium concentration as the forecasting target.

The synthetic data are used **only** for demonstrating code execution and workflow integrity. They **do not** reproduce confidential field measurements or the exact numerical results reported in the manuscript (tables and figures).

To regenerate the CSV (optional):

```bash
python scripts/generate_synthetic_demo.py
```
