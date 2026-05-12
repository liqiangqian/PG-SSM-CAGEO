# Demonstration data

This folder contains a **synthetic** demonstration dataset for running the PG-SSM workflow end-to-end.

The original industrial uranium in-situ leaching (ISL) monitoring records used in the manuscript **cannot** be publicly released because they are subject to operational confidentiality restrictions. The synthetic dataset follows the **same variable schema** as the field records: injection-flow proxies for four injection wells, a production-flow proxy, chemical indicators (pH and dissolved oxygen) on injectors, and production-well uranium concentration as the forecasting target.

The synthetic data are used **only** for demonstrating code execution and workflow integrity. They **do not** reproduce confidential field measurements or the exact numerical results reported in the manuscript (tables and figures).

To regenerate the CSV (optional):

```bash
python scripts/generate_synthetic_demo.py
```

**Public repository:** [https://github.com/liqiangqian/PG-SSM-CAGEO](https://github.com/liqiangqian/PG-SSM-CAGEO)

Note: `examples/quick_test.py` sets a shorter series by default (`PGSSM_QUICK_SYNTHETIC_N=420`) for a fast smoke test. For a longer CSV in the repository default style, run `generate_synthetic_demo.py` with `PGSSM_QUICK_SYNTHETIC_N=750` (or unset the variable; the script default is 750).
