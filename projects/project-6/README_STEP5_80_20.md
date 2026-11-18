# Project 6 — Step 5: independent 80/20 subclass split

## Why this patch exists

Previously the split held out 165 of 823 source/import subclass pairs (approximately 20%),
but only **one** of the nine rail candidate subclass pairs. This patch also holds out
20% of the nine rail candidate subclass pairs: **two**. This yields expected totals
of 665 training and 167 validation pairs for the reported current project dataset
(658+7 training, 165+2 validation). It is an **80/20 partition of named subclass
axioms**, not 80/20 of all ontology RDF triples or OWL expressions.

The 20% held-out rail subset must retain **both endpoints** in other training
subclass relations, and be unreachable through retained training paths. Otherwise
class centers could be random/cold-start embeddings, or a held-out pair could be
entailed from retained edges. Fails explicitly if the requested rail holdout
cannot be constructed from a different design. The selection is deterministic
with `seed=42`.

For the current 9-pair rail subclass design, the two held-out edges are:
- rail_passenger_role -> AgentOntology/passenger
- train -> BFO material entity

No changes to the ontology, Step 2–4, Step 5C, Step 6–10, or run_all.py are
necessary; Step 5C already handles any number of independently scored pairs.

## Install (PowerShell, from the project-6 root)

```powershell
Copy-Item .\scripts\prepare_rail_mowl.py .\scripts\prepare_rail_mowl_before_8020.py
Copy-Item .\scripts\evaluate_rail_mowl.py .\scripts\evaluate_rail_mowl_before_8020.py
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\Project6_Step5_80_20_Rail_Holdout.zip" -DestinationPath . -Force
```

The ZIP replaces only these four files:
- `scripts/prepare_rail_mowl.py`
- `scripts/evaluate_rail_mowl.py`
- `scripts/test_rail_mowl.py`
- `scripts/test_refine_rail_hybrid.py`

The two test files are updated for the two independently scored rail edges.

## Test and evaluate without rerunning Grok

```powershell
python -m unittest discover -s scripts -p "test_rail_mowl.py" -v
python -m unittest discover -s scripts -p "test_refine_rail_hybrid.py" -v
python -u scripts/prepare_rail_mowl.py --project-root . --seed 42
python -u scripts/evaluate_rail_mowl.py --project-root . --epochs 200 --dim 50 --seed 42
```

**Important:** Do not pass `--reuse-vectors` on the first run; the training
partition has changed and cached vectors from the older split are stale.
The last command runs mOWL training, but **does not make a Grok API request**.
It updates `reports/step5_cosine_scores.json`, `reports/step5_summary.json`,
`reports/step5_hybrid_review.json`, and the split and vector manifests.

Inspect results:

```powershell
$s = Get-Content .\reports\step5_split_report.json -Raw | ConvertFrom-Json
$s | Select-Object source_import_train_pairs, source_import_holdout_pairs, rail_candidate_training_pairs, rail_candidate_holdout_pairs, train_axiom_pairs, valid_axiom_pairs, overall_validation_fraction
$c = Get-Content .\reports\step5_cosine_scores.json -Raw | ConvertFrom-Json
$c | Select-Object heldout_rail_pairs, mean_heldout_rail_cosine, cosine_threshold, cosine_requirement_met
$c.heldout_rail_pair_scores | Format-Table child, parent, cosine
```

To subsequently regenerate the complete 13-stage workflow from scratch:

```powershell
python -u scripts/run_all.py --manifest templates/passenger_manifest_template.xlsx --flight-document FL-20261117-AA101=FPDOC-AA101
```

That run makes new live Grok calls. It is **not necessary** merely to inspect
this Step 5 split.

## Scope and academic integrity

This patch increases independent rail subclass holdouts from 1 to 2, but it
does not change ELEmbeddings' training loss or cosine calculation and cannot
guarantee a mean cosine >= 0.70. Only nine named candidate subclass edges
exist; most have a single training link and cannot be independently held out
without cold-start embeddings. The 165 source/import validation examples are
reference scores, **not** additional rail examples.

`PASS_HYBRID_REVIEW_ONLY` still means the mOWL/LLM stage executed, not that the
cosine 0.70 requirement is satisfied. The report now includes
`mean_heldout_rail_cosine`, `cosine_threshold` and `cosine_requirement_met`.
The pipeline does **not** fail-fast on this threshold in this limited split
update. The user's separate `audit_step5_cosine.py --strict` can explicitly
fail on the threshold; true full-pipeline gating would require editing the
runner and assessing whether this is the assignment's intended metric.

mOWL itself was not installed in the test environment. We validated the split,
vector scoring, report arithmetic, SHA/provenance, and downstream Step 5C
integration using controlled offline fixtures, not a live mOWL training run.
