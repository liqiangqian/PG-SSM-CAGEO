# Final Manuscript Repository Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the obsolete R1 public workflow with a confidential-data-safe, manuscript-aligned PG-SSM synthetic demonstration, aggregate-evidence package, tests, and submission-grade documentation.

**Architecture:** The public workflow is split into causal preprocessing, past-only stage assignment, a receiving-row graph/dual-branch Gaussian model, statistical evaluation, and a small orchestration script. Final field-derived values live only in row-free aggregate evidence files, while executable outputs are explicitly synthetic and are checked separately.

**Tech Stack:** Python 3.9.13, PyTorch 2.2.2, NumPy 1.26.4, pandas 2.3.2, scikit-learn 1.6.1, SciPy 1.13.1, PyArrow 21.0.0, Matplotlib 3.8.4, standard-library `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-22-final-manuscript-alignment-design.md`

## Global Constraints

- Never add raw field data, actual identifiers, coordinates, row-level field predictions, protected hashes, or protected bundle metadata.
- Synthetic data and results must say `synthetic` and must not be tuned to field metrics.
- Locked defaults: `L=28`, `H=7`, seed `43`, alpha/beta `1.0`, repeated seeds `41–45` only for PG-SSM/LSTM/TCN/N-BEATS.
- Scoring uses the untruncated Gaussian; lower-bound clipping is visualization-only.
- Stage labels use only observations available at or before forecast origin; validation-frozen thresholds are immutable at test time.
- Active terminology excludes `w/o Physics`, `Physics loss`, `Physics constraint`, peak correction, and PTE as a primary metric.
- Keep exact dependency pins; CPU is the only verified public execution claim.
- Do not rewrite Git history, force push, move the old tag, or publish without a passing sensitive-artifact scan.

## Review Focus

- A target on a non-assay date must be omitted even when causal LOCF supplies an input value; Task 2 pins this.
- A validation/test sample may use earlier historical rows but its partition must be determined only by target date; Task 2 pins this.
- A future perturbation must not change an earlier stage label, including peak-transition; Task 3 pins this.
- Display clipping must not alter NLL, CRPS, PIT, calibration, or raw interval coverage; Task 5 pins this.
- Public scans must reject private-path, secret, coordinate-column, protected-filename, and sensitive-extension patterns without embedding real sensitive terms; Task 8 pins this.

---

### Task 1: Locked public configuration

**Files:**
- Create: `configs/manuscript_demo.json`
- Create: `tests/test_config_alignment.py`
- Delete: `configs/field_analysis_configuration.json`

**Interfaces:**
- Produces: JSON configuration consumed by generators, preprocessing, model, demo, and consistency checks.

- [ ] **Step 1: Write the failing configuration tests**

```python
class ConfigAlignmentTests(unittest.TestCase):
    def test_locked_defaults_and_candidate_grids(self):
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual((cfg["history_days"], cfg["horizon_days"]), (28, 7))
        self.assertEqual(cfg["primary_seed"], 43)
        self.assertEqual(cfg["repeated_seed_stability"]["seeds"], [41, 42, 43, 44, 45])
        self.assertEqual(cfg["graph"]["alpha"], 1.0)
        self.assertEqual(cfg["graph"]["beta"], 1.0)

    def test_no_protected_config_metadata(self):
        text = CONFIG.read_text(encoding="utf-8").lower()
        for forbidden in ("sha256", "latitude", "longitude", "date_start", "actual_well"):
            self.assertNotIn(forbidden, text)
```

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.test_config_alignment -v`
Expected: FAIL because `configs/manuscript_demo.json` does not exist.

- [ ] **Step 3: Add the exact manuscript-demo configuration**

Create JSON sections for task, graph, stages, regularization, optimizer, split, environment, baseline selection, and synthetic generation using every locked value from the spec.

- [ ] **Step 4: Run GREEN and the full suite**

Run: `python -m unittest tests.test_config_alignment -v && python -m unittest discover -s tests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs tests/test_config_alignment.py
git commit -m "test: lock final manuscript configuration"
```

### Task 2: Synthetic mixed-frequency data and causal preprocessing

**Files:**
- Create: `src/preprocessing.py`
- Create: `scripts/generate_synthetic_demo.py`
- Create: `tests/test_preprocessing.py`
- Generate: `data/synthetic_demo.csv`
- Delete: `data/synthetic_five_well.npz`
- Delete: `scripts/generate_synthetic_data.py`

**Interfaces:**
- Produces: `causal_assay_features(frame) -> DataFrame`, `fit_train_normalizer(frame, train_end) -> Normalizer`, and `build_endpoint_samples(frame, config) -> dict[str, SampleSet]`.
- Sample tensors have shape `(samples, 28, 5, features)` and contain mask/days-since channels.

- [ ] **Step 1: Write failing causal preprocessing tests**

```python
def test_causal_forward_fill_never_uses_future_assay(self):
    frame = fixture_with_assays(day0=1.0, day3=9.0)
    out = causal_assay_features(frame)
    self.assertEqual(out.loc[1, "uranium_locf"], 1.0)
    self.assertEqual(out.loc[2, "uranium_locf"], 1.0)

def test_target_partitioning_and_observed_target_scoring_only(self):
    sets = build_endpoint_samples(self.frame, self.config)
    self.assertTrue(all(s.target_observed for s in sets["test"].metadata))
    self.assertTrue(all(s.target_date >= self.test_start for s in sets["test"].metadata))

def test_train_only_normalization_is_unchanged_by_test_outlier(self):
    first = fit_train_normalizer(self.frame, self.train_end)
    changed = self.frame.copy()
    changed.loc[changed.date > self.train_end, "extraction_flow"] = 1e9
    self.assertEqual(first, fit_train_normalizer(changed, self.train_end))
```

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.test_preprocessing -v`
Expected: FAIL because `src.preprocessing` is missing.

- [ ] **Step 3: Implement causal feature construction and sample building**

Use grouped `ffill` only, derive `assay_observed` and `days_since_assay`, fit means/stds on training dates only, permit histories to cross prior split boundaries, assign by target date, and filter targets on the explicit assay mask.

- [ ] **Step 4: Implement deterministic forward synthetic generator**

Generate a generic daily five-well CSV with `day`, `well_role`, `injection_flow`, `extraction_flow`, `ph`, `dissolved_oxygen`, `uranium_assay`, and metadata-safe synthetic geometry fields. Make assay schedules mixed-frequency and retain blank non-assay values.

- [ ] **Step 5: Run GREEN, generate data, and run the full suite**

Run: `python -m unittest tests.test_preprocessing -v; python scripts/generate_synthetic_demo.py; python -m unittest discover -s tests -v`
Expected: all tests PASS and `data/synthetic_demo.csv` is regenerated deterministically.

- [ ] **Step 6: Commit**

```bash
git add src/preprocessing.py scripts/generate_synthetic_demo.py tests/test_preprocessing.py data/synthetic_demo.csv
git add -u data scripts
git commit -m "feat: add causal synthetic preprocessing workflow"
```

### Task 3: Past-only four-stage assignment

**Files:**
- Create: `src/stages.py`
- Create: `tests/test_stage_causality.py`

**Interfaces:**
- Produces: immutable `StageThresholds`, `fit_stage_thresholds(validation_history, candidates)`, and `assign_stage(history, flows, thresholds) -> str`.

- [ ] **Step 1: Write failing causality and frozen-threshold tests**

```python
def test_future_values_cannot_change_origin_stage(self):
    prefix = np.array([1.0, 1.02, 1.07, 1.13, 1.18, 1.20, 1.19])
    a = assign_stage(prefix, self.flows, self.thresholds)
    b = assign_stage(np.r_[prefix, 99.0, -99.0][:len(prefix)], self.flows, self.thresholds)
    self.assertEqual(a, b)

def test_thresholds_are_frozen(self):
    thresholds = fit_stage_thresholds(self.validation, self.candidates)
    with self.assertRaises(FrozenInstanceError):
        thresholds.eta_y = 9.0
```

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.test_stage_causality -v`
Expected: FAIL because `src.stages` is missing.

- [ ] **Step 3: Implement the causal trend-change classifier**

Use only the last seven historical LOCF values and origin-available flow ratio. Implement Rising, Peak-transition, Quasi-steady, and Declining without searching for future maxima.

- [ ] **Step 4: Run GREEN and the full suite**

Run: `python -m unittest tests.test_stage_causality -v; python -m unittest discover -s tests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stages.py tests/test_stage_causality.py
git commit -m "feat: add causal four-stage identification"
```

### Task 4: Receiving-row PG-SSM and soft regularization

**Files:**
- Modify: `src/pgssm_model.py`
- Create: `tests/test_model_smoke.py`

**Interfaces:**
- Produces: `build_receiving_row_affinity(distances, injection_flow, extraction_flow, alpha, beta, scale)`, `PGSSM.forward(x, flows) -> (mean, log_variance, affinity)`, and `pgssm_loss(...)`.

- [ ] **Step 1: Write failing graph/model tests**

```python
def test_receiving_row_uses_custom_geometry_and_normalizes_rows(self):
    a = build_receiving_row_affinity(self.distances, self.injection, self.extraction, 1.0, 1.0, 1.0)
    self.assertTrue(torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-6))
    self.assertGreater(a[0, 1], a[0, 4])

def test_forward_returns_finite_gaussian_parameters(self):
    mean, log_variance, affinity = self.model(self.x)
    self.assertEqual(mean.shape, (3,))
    self.assertTrue(torch.isfinite(mean).all() and torch.isfinite(log_variance).all())
```

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.test_model_smoke -v`
Expected: FAIL because the manuscript-aligned graph function/interface is absent.

- [ ] **Step 3: Replace the old architecture**

Implement dynamic receiving-row affinities with self loops and row normalization, a graph encoder, separate slow and fast GRU branches, fusion, and a two-output Gaussian head. Implement non-negativity, rate-consistency, and stage-consistent monotonicity terms with manuscript names and weights.

- [ ] **Step 4: Run GREEN and the full suite**

Run: `python -m unittest tests.test_model_smoke -v; python -m unittest discover -s tests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pgssm_model.py tests/test_model_smoke.py
git commit -m "refactor: align PG-SSM architecture and regularization"
```

### Task 5: Deterministic, Gaussian, and consistency evaluation

**Files:**
- Create: `src/evaluation.py`
- Create: `tests/test_gaussian_scoring.py`

**Interfaces:**
- Produces: `deterministic_metrics`, `gaussian_metrics`, `calibration_table`, `physical_consistency`, and `display_interval`.

- [ ] **Step 1: Write failing metric-boundary tests**

```python
def test_nll_uses_untruncated_gaussian(self):
    score = gaussian_metrics(np.array([-1.0]), np.array([0.0]), np.array([0.0]))
    self.assertAlmostEqual(score["nll"], 0.5 * (1.0 + np.log(2 * np.pi)))

def test_display_clipping_does_not_change_scores(self):
    raw = gaussian_metrics(self.y, self.mu, self.log_var)
    _ = display_interval(self.mu, self.log_var, clip_lower=True)
    self.assertEqual(raw, gaussian_metrics(self.y, self.mu, self.log_var))
```

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.test_gaussian_scoring -v`
Expected: FAIL because `src.evaluation` is missing.

- [ ] **Step 3: Implement all manuscript metrics**

Implement RMSE, MAE, R², MASE, coverage, width, Gaussian NLL/CRPS, Winkler score, sharpness, PIT, multi-level calibration, negative prediction rate, rate violation, and stage violation. Keep display clipping in a separate function.

- [ ] **Step 4: Run GREEN and the full suite**

Run: `python -m unittest tests.test_gaussian_scoring -v; python -m unittest discover -s tests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation.py tests/test_gaussian_scoring.py
git commit -m "feat: add manuscript-aligned evaluation metrics"
```

### Task 6: End-to-end CPU synthetic demo

**Files:**
- Create: `scripts/run_demo.py`
- Create: `tests/test_demo_smoke.py`
- Replace: `field_results/synthetic_results.json`
- Delete: `scripts/run_synthetic_example.py`

**Interfaces:**
- Produces: `run_demo(config_path, epochs, output_path) -> dict` and a CLI accepting `--config`, `--epochs`, and `--output`.

- [ ] **Step 1: Write the failing end-to-end smoke test**

```python
def test_cpu_demo_runs_and_marks_output_synthetic(self):
    result = run_demo(CONFIG, epochs=1, output_path=self.output)
    self.assertTrue(result["synthetic_only"])
    self.assertEqual(result["device"], "CPU")
    self.assertEqual((result["history_days"], result["horizon_days"]), (28, 7))
    self.assertGreater(result["scored_test_targets"], 0)
```

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.test_demo_smoke -v`
Expected: FAIL because `scripts.run_demo` is missing.

- [ ] **Step 3: Implement the orchestration and minimal training loop**

Load config/CSV, build causal samples, train on CPU with seed 43, select on validation only, score assay-observed test targets, and write clearly synthetic metrics without comparing them to field values.

- [ ] **Step 4: Run GREEN and regenerate the checked synthetic result**

Run: `python -m unittest tests.test_demo_smoke -v; python scripts/run_demo.py --config configs/manuscript_demo.json --epochs 2; python -m unittest discover -s tests -v`
Expected: PASS and `field_results/synthetic_results.json` contains `synthetic_only: true`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_demo.py tests/test_demo_smoke.py field_results/synthetic_results.json
git add -u scripts
git commit -m "feat: add end-to-end CPU synthetic demo"
```

### Task 7: Final aggregate manuscript evidence and consistency checker

**Files:**
- Replace: `field_results/README.md`
- Create: `field_results/manuscript_table2_deterministic.csv`
- Create: `field_results/manuscript_table3_probabilistic.csv`
- Create: `field_results/manuscript_table4_ablation.csv`
- Replace: `field_results/manuscript_calibration.csv`
- Replace: `field_results/manuscript_stage_coverage.csv`
- Create: `field_results/manuscript_sensitivity.csv`
- Create: `field_results/manuscript_seed_stability.csv`
- Create: `field_results/evidence_manifest.json`
- Replace: `scripts/check_manuscript_consistency.py`
- Create: `tests/test_evidence_consistency.py`
- Delete: obsolete current evidence files not named above.

**Interfaces:**
- Produces: aggregate-only CSV/JSON evidence and `check_repository() -> list[str]`.

- [ ] **Step 1: Write failing exact-value and row-granularity tests**

```python
def test_primary_and_calibration_values_match_final_documents(self):
    self.assertEqual(self.primary["PG-SSM"]["RMSE"], "0.455")
    self.assertEqual(self.primary["PG-SSM"]["R2"], "0.933")
    self.assertEqual(self.calibration["90"]["hits"], "34")

def test_no_row_level_fields_exist(self):
    blob = "\n".join(p.read_text(encoding="utf-8") for p in EVIDENCE.glob("*"))
    for field in ('"date"', '"y_true"', '"y_pred"', '"coordinate"'):
        self.assertNotIn(field, blob.lower())
```

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.test_evidence_consistency -v`
Expected: FAIL because the current evidence is the obsolete n=73 record.

- [ ] **Step 3: Transcribe only final aggregate evidence**

Use the final DOCX-derived Table 2–4, calibration, stage-coverage, sensitivity, and repeated-seed values. Include n=38, PG-SSM RMSE 0.455, R² 0.933, PI90 34/38, final ablation labels, central-interval coverage, and four stage groups. Do not include dates or row predictions.

- [ ] **Step 4: Replace consistency checker**

Check config, title, evidence, terminology, file allowlist, data-boundary statements, and absence of old n=73/R1 values.

- [ ] **Step 5: Run GREEN and the checker/full suite**

Run: `python -m unittest tests.test_evidence_consistency -v; python scripts/check_manuscript_consistency.py; python -m unittest discover -s tests -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add field_results scripts/check_manuscript_consistency.py tests/test_evidence_consistency.py
git commit -m "data: align aggregate evidence with final manuscript"
```

### Task 8: Public documentation, cleanup, and sensitive-artifact gate

**Files:**
- Replace: `README.md`
- Create: `DATA_BOUNDARY.md`
- Create: `docs/DATA_BOUNDARY.md`
- Create: `docs/MANUSCRIPT_ALIGNMENT.md`
- Replace: `docs/USER_GUIDE.md`
- Replace: `docs/VARIABLE_SCHEMA.md`
- Replace: `RELEASE_NOTES.md`
- Modify: `.gitignore`
- Create: `scripts/scan_sensitive_artifacts.py`
- Create: `tests/test_no_sensitive_artifacts.py`
- Delete: `archive/`, `field_analysis/`, `docs/FIELD_DATA_LIMITATIONS.md`, and obsolete reconciliation report from the current tree.

**Interfaces:**
- Produces: `scan_repository(root) -> list[Finding]`, CLI exit 0 only when clean, and complete public documentation.

- [ ] **Step 1: Write failing scan and terminology tests**

```python
def test_scanner_rejects_generic_sensitive_patterns(self):
    (self.root / "leak.txt").write_text("password=example-value", encoding="utf-8")
    self.assertTrue(scan_repository(self.root))

def test_active_public_tree_has_no_legacy_terms_or_protected_adapter(self):
    findings = scan_repository(ROOT)
    self.assertEqual(findings, [])
    self.assertFalse((ROOT / "field_analysis").exists())
```

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.test_no_sensitive_artifacts -v`
Expected: FAIL because the scanner is missing and protected hashes/old paths remain.

- [ ] **Step 3: Implement the generic scanner and ignore policy**

Scan tracked public files for secrets, absolute paths, private URLs/IPs, sensitive extensions, coordinate-like headers, protected filenames, forbidden terms, caches, and unexpected binary/data files. Exclude Git internals and the design/plan historical-analysis documents from active-terminology checks while still scanning them for secrets/paths.

- [ ] **Step 4: Rewrite documentation and remove obsolete current-tree material**

Document exact title/scope, public/not-public lists, synthetic purpose, causal preprocessing, architecture, metrics, baseline tuning distinctions, seeds, pinned CPU environment, aggregate evidence boundary, MIT license, and citation placeholder. Remove obsolete archive and public field adapter without rewriting history.

- [ ] **Step 5: Run GREEN and replay every README command**

Run: `python -m unittest tests.test_no_sensitive_artifacts -v; python scripts/generate_synthetic_demo.py; python scripts/run_demo.py --config configs/manuscript_demo.json --epochs 2; python scripts/check_manuscript_consistency.py; python scripts/scan_sensitive_artifacts.py; python -m unittest discover -s tests -v`
Expected: every command exits 0.

- [ ] **Step 6: Commit**

```bash
git add README.md DATA_BOUNDARY.md docs RELEASE_NOTES.md .gitignore scripts/scan_sensitive_artifacts.py tests/test_no_sensitive_artifacts.py
git add -u archive field_analysis docs REPOSITORY_MANUSCRIPT_RECONCILIATION_REPORT.md
git commit -m "docs: publish final safe repository boundary"
```

### Task 9: Final verification, review, and release commit

**Files:**
- Modify only files required by verified Critical/Important review findings.

**Interfaces:**
- Consumes the complete repository and produces a verified branch ready for normal push.

- [ ] **Step 1: Run the complete verification gate**

Run: `python -c "import numpy,pandas,torch,sklearn,scipy,pyarrow,matplotlib; print('dependency import PASS')"`

Run: `python scripts/generate_synthetic_demo.py`

Run: `python scripts/run_demo.py --config configs/manuscript_demo.json --epochs 2`

Run: `python scripts/check_manuscript_consistency.py`

Run: `python scripts/scan_sensitive_artifacts.py`

Run: `python -m unittest discover -s tests -v`

Run: `git diff --check`

Expected: all commands exit 0 with no warnings that invalidate the public workflow.

- [ ] **Step 2: Inspect release state**

Run: `git diff --stat 8b5b4a7..HEAD; git status --short; git log --oneline --decorate -10; git tag --list`
Expected: only final-alignment changes, clean/understood generated changes, and unchanged `v2.0-final-r2`.

- [ ] **Step 3: Conduct whole-branch review and test any fixes RED→GREEN**

Review against the spec, plan Review Focus, and ledger rulings. For every Critical/Important finding, add a reproducing test, observe RED, fix minimally, observe GREEN, and rerun the suite.

- [ ] **Step 4: Create the requested aggregate implementation commit if needed**

```bash
git add -A
git commit -m "refactor: align public workflow with final PG-SSM manuscript"
```

- [ ] **Step 5: Normal push only after a final clean scan**

Run: `git push origin main`
Expected: ordinary fast-forward push succeeds. Do not force, rewrite history, or move tags.
