# Project 6 Step 5D — Development-only EL structural training (v3)

This patch extends the **last working freeze-fixed** version of `scripts/experiment_el_structural_mowl.py`. It supports the requested **50-epoch development experiment without running the outer final fit**. It retains the existing `--diagnose-only` and normal full-training behavior.

## Included files (replace only these)

- `scripts/experiment_el_structural_mowl.py` — existing experiment with new `--development-only` mode.
- `scripts/test_experiment_el_structural_mowl.py` — 32 offline tests, including development-only isolation.

The archive does **not** modify or include `scripts/experiment_joint_rail_mowl.py`, `scripts/run_all.py`, original ontology, imports, or mOWL input split artifacts. **Do not** overwrite those files.

## Install

Open PowerShell in the Project 6 root and ensure `.venv-mowl` is active. Adjust ZIP location if needed.

```powershell
cd "C:\Users\cappa\Ontology-Tradecraft\projects\project-6"
Copy-Item .\scripts\experiment_el_structural_mowl.py .\scripts\experiment_el_structural_mowl_pre_dev_only.py
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\Project6_Step5_EL_Development_Only.zip" -DestinationPath . -Force
```

## Run optional offline regression tests

```powershell
python -m unittest discover -s scripts -p "test_experiment_el_structural_mowl.py" -v
```

## Run the 50-epoch development-only experiment

```powershell
python -u scripts/experiment_el_structural_mowl.py `
  --project-root . `
  --development-only `
  --seed 42 `
  --configs 1 `
  --tune-epochs 50 `
  --max-inferred 50 `
  --controls 20 `
  --threshold 0.70
```

This runs **one development training fit of 50 epochs**, not a 50-epoch development fit followed by a final fit. `--final-epochs` is ignored in this mode. It does not evaluate the outer 167 holdout pairs and does not create or overwrite the outer training graph, outer model vectors, or `reports/step5_el_structural_comparison.json`. It will update the development graph, its metadata, and `reports/step5_el_structure_diagnostic.json`.

**New output:** `reports/step5_el_structural_development.json`.

```powershell
$r = Get-Content .\reports\step5_el_structural_development.json -Raw | ConvertFrom-Json
$r.status
$r.outer_holdout_evaluated
$r.chosen_dev_result.epochs
$r.chosen_dev_result.mean_cosine
$r.chosen_dev_result.positive_control_auc
$r.chosen_dev_result.positive_control_gap
$r.chosen_dev_result.train_positive_mean_cosine
```

Compare against the previous 20-epoch development result: mean cosine **0.127652**, development AUC **0.691964**, development control gap **0.081704**, development training mean cosine **0.378328**. For a useful learning signal, development cosine **and** discrimination should increase together; only training cosine rising does not justify much longer training.

If this looks promising, repeat the development-only command with `--tune-epochs 100`. The current implementation **restarts from scratch each run** (no training checkpoint resumption), so 100 is not an incremental addition of 50.

## Evaluation safeguards

- Outer 80/20 holdout is *not evaluated or used for model selection* by `--development-only`.
- Training and validation split provenance hashes are checked before and after preparation/training.
- mOWL EL normal-form counts are computed for the development training graph. GCI1, GCI2, or GCI3 must be nonzero.
- Separate development report makes no claim that the 0.70 full-holdout threshold was met.
- Original source ontology, imports, Project 6 `run_all.py`, Step 5A/B reports, and outer artifact outputs are not written by this mode.
- OWL entailment-level leakage has **not** been independently certified; held-out child filtering and named transitive-closure checks are safeguards, not a proof.
- Source label/definition features may contain semantic hints, and sampled control pairs are not necessarily false under open-world semantics.

## Local verification

32 offline unit/regression tests passed. These include checks that `--development-only` only trains once when `--configs 1`, never calls outer normalization, never overwrites previous outer comparison/vector files, and exits clearly when no structural EL axiom survives. Actual mOWL/JVM training requires your configured Windows environment and was **not** executed by the authoring environment.
