# Variable schema

| Variable | Description | Role |
| --- | --- | --- |
| `date` | Daily timestamp | Temporal index |
| `inj1_flow`–`inj4_flow` | Injection flow proxies for four injection wells | Hydrodynamic drivers (fast branch) |
| `prod_flow` | Production flow proxy | Hydrodynamic sink / mass-flux context (fast branch) |
| `inj1_pH`–`inj4_pH` | Injection acidity indicators | Chemical covariates (slow branch) |
| `inj1_DO`–`inj4_DO` | Dissolved oxygen indicators | Oxidant-related covariates (slow branch) |
| `uranium_concentration` | Uranium concentration at the central production well | Forecasting target |

The demonstration setting follows a **five-spot** ISL unit (four injectors and one central producer). The default configuration uses a **28-day** input window and a **7-day** forecast horizon (target aligned to the day at horizon length after the window end), consistent with the manuscript’s structured forecasting setup.
