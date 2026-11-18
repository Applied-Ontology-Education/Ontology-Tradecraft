# Project 6 — Independent cosine-aware embedding experiment

This is a **targeted research experiment for Step 5**, not a patch that changes the successfully validated 709-triple ontology. It compares:

1. **mOWL ELEmbeddings baseline:** the original 80/20 Step 5B trained class-center vectors, re-scored independently on the exact saved validation set;
2. **Cosine-aware PyTorch model:** a separate embedding table, trained **only on training subclass relationships** and sampled unasserted control pairs using a logistic contrastive cosine objective.

The experimental model is **not mOWL ELEmbeddings**; its higher cosine (if any) does not establish an improvement to ELEmbeddings' native geometric OWL scoring. It is a comparison designed to determine whether a cosine-aligned objective is a better fit for your professor's required metric. Both models use the **same immutable Step 5A 80/20 split**. The experiment does not make any LLM calls, rerun Steps 1–4, or alter source or candidate ontology files. It does not change the `run_all.py` success/failure status.

## Install

From `C:\Users\cappa\Ontology-Tradecraft\projects\project-6` in an activated `.venv-mowl` PowerShell terminal:

```powershell
Expand-Archive `
  -LiteralPath "$env:USERPROFILE\Downloads\Project6_Step5_Cosine_Contrastive_Experiment.zip" `
  -DestinationPath . `
  -Force
```

Only two new files are added:

- `scripts/compare_rail_cosine.py`
- `scripts/test_compare_rail_cosine.py`

The previously installed 80/20 Step 5A and Step 5B must be run first, so these files must already exist:

- `generated/mowl_rail_train.ttl`
- `generated/mowl_rail_valid.ttl`
- `generated/mowl_rail_class_vectors.npz`
- `reports/step5_split_report.json`
- `reports/step5_cosine_scores.json`
- `reports/step5_model_manifest.json`
- `src/module.ttl` and `generated/rail_axioms_refined.ttl`

The program verifies the SHA-256 hashes of the split files, source and candidate against the reports and baseline model manifest; it rejects a stale or accidentally changed model.

## Offline unit tests

```powershell
python -m unittest discover -s scripts -p "test_compare_rail_cosine.py" -v
```

## Train and compare, using the same 80/20 split

```powershell
python -u scripts/compare_rail_cosine.py --project-root . --epochs 300 --dim 50 --seed 42 --threshold 0.70
```

This makes **no external API request**. It saves:

- `reports/step5_cosine_comparison.json`: exact-pair baseline and experimental scores, rail/source held-out means, unasserted-control means, discrimination AUC, threshold status, complete training provenance;
- `generated/cosine_contrastive_class_vectors.npz`: separate experimental class embeddings. Never overwrites the mOWL `.npz`.

Inspect results:

```powershell
$r = Get-Content .\reports\step5_cosine_comparison.json -Raw | ConvertFrom-Json
$r.status
$r.baseline_mowl | Select-Object heldout_rail_mean_cosine, heldout_reference_mean_cosine, rail_control_auc
$r.experiment_cosine_aware | Select-Object heldout_rail_mean_cosine, heldout_reference_mean_cosine, rail_control_mean_cosine, rail_control_auc, rail_positive_minus_control
$r.experiment_cosine_aware.rail_pair_scores | Format-Table child, parent, cosine
```

`--strict` makes the command exit with code **1** unless the *experimental* held-out rail cosine mean is **at least 0.70**, rail versus unasserted-control discrimination AUC is **at least 0.70**, and the rail mean exceeds the control mean by **at least 0.15**. Default mode always writes the report and does not return failure simply for a low cosine.

```powershell
python -u scripts/compare_rail_cosine.py --project-root . --strict
```

## Scientific safeguards and limitations

- All **two rail candidate holdouts** plus the **165 source/import holdouts** remain out of positive training. They are not reintroduced to training to inflate scores.
- Known validation positives are *only excluded* from negative sampling so they are not accidentally labeled as negatives; they are never positive training targets. This masking is disclosed because it uses validation-label knowledge.
- The negative controls are **unasserted relationships**, **not provably false OWL axioms**. Under OWL's open-world semantics, lack of a statement does not establish negation.
- The cosine-based model uses a symmetric similarity function, so it cannot encode subclass direction or replace OWL reasoning.
- The experiment uses a fixed hyperparameter configuration. Repeated tuning based on the same held-out validation set would bias evaluation and require a fresh test split.
- **Two rail holdouts are a very small sample.** A high rail mean alone is not convincing without discrimination controls and the larger reference-set results.
- Raw cosine is not ELEmbeddings' native ball-inclusion axiom scoring metric.
- The earlier full Step 8 execution remains valid for **SHACL and scoped HermiT**, but that does not imply the separate 0.70 cosine criterion was met.
- This experiment **cannot guarantee 0.70**. Its failure is an authentic result, not a bug to be suppressed. It will preserve the original Step 5B baseline and show both outcomes.

**After running, share `reports/step5_cosine_comparison.json`.** We can then determine whether the cosine-aware objective improves generalization, and only integrate a verified model or scoring change into `run_all.py` if it is justified by the results.
