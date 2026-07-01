# Data Schema

The public synthetic dataset is stored in `data/synthetic/synthetic_five_spot.csv`.

| Column | Description |
| --- | --- |
| `date` | Synthetic daily timestamp |
| `unit_id` | Anonymized synthetic unit identifier, e.g. `U_SYN_001` |
| `well_id` | Anonymized well identifier, e.g. `W0` to `W4` |
| `well_role` | `extraction` for the central well or `injection` for surrounding wells |
| `anonymized_x` | Synthetic local x-coordinate |
| `anonymized_y` | Synthetic local y-coordinate |
| `uranium_concentration` | Synthetic central-well target value; injection-well rows are not field observations |
| `injection_flow_proxy` | Synthetic injection-flow proxy |
| `extraction_flow_proxy` | Synthetic extraction-flow proxy |
| `hydrochemical_proxy_1` | Synthetic hydrochemical covariate |
| `hydrochemical_proxy_2` | Synthetic hydrochemical covariate |
| `stage_label` | Synthetic operational stage label |

The preprocessing script converts the long-format file into a wide synthetic table under `outputs/synthetic_quick_test/processed/` for model training.

This dataset is synthetic and is provided only for workflow verification. It must not be interpreted as field data.
