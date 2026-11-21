# Project 6: Stage 1 rail-only holdout feasibility diagnostic

**Purpose:** Identify whether 4 folds × 2 newly added rail classes can be tested without training/validation leakage using the *current* mOWL named-subclass-only training projection. **No training or ontology edits.**

The diagnostic uses your existing:
- `src/module.ttl` (421 source triples)
- `generated/module_augmented.ttl` (484 augmented triples)
- `imports/` (local reference ontology context, loaded if present)

It does not need `sentence-transformers`, an API key, Java, ROBOT, or PyTorch. It only uses Python's standard library and the existing `rdflib` package.

## Run in the restored Project 6 PowerShell terminal

From `C:\Users\cappa\Ontology-Tradecraft\projects\project-6`:

```powershell
Expand-Archive "$env:USERPROFILE\Downloads\Project6_Rail_Holdout_Diagnostic.zip" -DestinationPath . -Force
python -u scripts\diagnose_rail_holdouts.py --project-root .
```

This writes only **`reports/rail_holdout_feasibility.json`**. `src/module.ttl`, `generated/module_augmented.ttl`, prior reports, and all pipeline scripts are untouched. Do **not** rerun `run_all.py` just for this diagnostic.

Look at `summary`, `folds`, `per_class`, and `imports_loaded` in the generated JSON. Upload that report for the next iterative step.

## Method and limits

- First verify 421 original triples, 484 merged triples, 8 added rail classes, 4 object properties, 2 data properties, and 63 rail TBox triples.
- Of the nine added explicit subclass relationships, choose one primary relation per new rail class. For `rail_passenger_role`, the more specific `passenger` parent is the primary test relation; the broader BFO Role assertion remains possible training context.
- Assign all eight classes to a deterministic four-fold split with two held-out primary edges per fold (seed 42).
- For each held-out class, check whether the remaining named-subclass graph provides its own parent anchor and whether the held-out relation is reachable transitively (leakage).
- **Important:** labels, definitions, object-property domain/range constraints, and richer OWL EL expressions are **not** consumed by the existing Stage 5A training projection. They are counted/described for future options, but not treated as evidence already used by mOWL.
- The audit performs **no cosine calculation** and makes no 0.70 claim. It assesses whether a fair test is viable *before* training.

## Offline verification performed before delivery

Two unit tests passed, and the diagnostic ran successfully on an earlier user-supplied 421-triple source and a 484-triple augmented copy. That copy has a different byte hash from the November 18 final run, so **the definitive result comes from the report you run in your restored project directory**.
