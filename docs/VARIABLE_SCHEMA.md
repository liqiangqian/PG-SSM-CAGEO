# Variable schema

Each daily well record contains six variables in this order:

| Index | Field label | Role |
|---:|---|---|
| 0 | `U/mg/l` | Uranium concentration; centre-well target |
| 1 | `Q日抽` | Daily flow feature |
| 2 | `Q瞬时` | Instantaneous flow feature |
| 3 | `Q累计` | Cumulative flow feature |
| 4 | `U/㎏` | Uranium mass feature |
| 5 | `工作频率` | Operating-frequency feature |

The public synthetic dataset preserves this tensor layout but contains no field measurements or site coordinates.
