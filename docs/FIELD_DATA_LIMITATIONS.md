# Field-data limitations

The monitoring archive and exact coordinates are owned by an industrial data provider and cannot be publicly redistributed. This restriction covers row-level observations, dated row-level predictions that reproduce observations, and exact site geometry.

The repository therefore provides:

- a deterministic synthetic dataset with the same tensor dimensions;
- the corrected model and loss implementation;
- portable field-analysis scripts for authorized data holders;
- the locked configuration and cryptographic source hash;
- aggregate, non-row-level results used in the manuscript.

Qualified researchers may request controlled access from the corresponding author. Release remains subject to permission from the data owner. The public materials support code execution and provenance inspection; they do not permit independent replay of confidential field metrics without the protected source rows.
