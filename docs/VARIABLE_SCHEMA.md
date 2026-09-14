# Variable schema

Each daily well record contains six variables in this order:

| Index | Field label | Role |
|---:|---|---|
| 0 | `U/mg/l` | Uranium concentration; extraction-well target |
| 1 | `Q日抽` | Daily flow feature used as an operational proxy |
| 2 | `Q瞬时` | Instantaneous flow feature |
| 3 | `Q累计` | Cumulative flow feature |
| 4 | `U/㎏` | Uranium mass feature |
| 5 | `工作频率` | Operating-frequency feature |

The public synthetic dataset preserves this tensor layout. It contains no field measurements, no site coordinates, and no well identifiers. Synthetic values are used only for execution, schema, and dimensional checks.
