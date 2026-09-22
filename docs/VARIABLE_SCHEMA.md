# Synthetic variable schema

`data/synthetic_demo.csv` is a long-format synthetic table with five rows per calendar day.

| Field | Meaning |
|---|---|
| `day` | Zero-based synthetic daily calendar index |
| `well_role` | Generic central extraction or injector role |
| `injection_flow` | Synthetic daily injection flow; zero for the central role |
| `extraction_flow` | Synthetic daily extraction flow; zero for injector roles |
| `ph` | Bounded synthetic pH process |
| `dissolved_oxygen` | Bounded synthetic dissolved-oxygen process |
| `uranium_assay` | Mixed-frequency synthetic assay; blank when not observed |
| `synthetic_record` | Explicit marker that the row is synthetic |

Preprocessing derives `assay_observed`, `uranium_locf`, and `days_since_assay`. These derived fields are intentionally not pre-baked into the CSV so the causal transformation remains testable.

The table contains no actual identifiers, coordinates, site labels, or copied field rows.
