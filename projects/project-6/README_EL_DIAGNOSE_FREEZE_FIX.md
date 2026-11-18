# Project 6 Step 5D — EL structural diagnostic freeze fix (v2)

## Why the previous `--diagnose-only` command could appear frozen

The old `--diagnose-only` invoked `prepare()`, which **built and serialized** two large ontology training graphs, called `rdflib.compare.to_canonical_graph()` across the whole graph, and sometimes performed full RDF blank-node graph-isomorphism checks. Whole-graph blank-node canonicalization may become very slow for graphs containing repeated anonymous class expressions. Those operations were unnecessary for a syntax-only diagnostic.

This fix removes those operations from `--diagnose-only`. It also replaces whole-graph canonicalization during *full training preparation* with deterministic structural-expression serialization and bounded round-trip signature checks. It retains the exact existing named 80/20 pairs and excludes the same held-out child classes from structural axioms.

## Files to replace

- `scripts/experiment_el_structural_mowl.py` — replacement script
- `scripts/test_experiment_el_structural_mowl.py` — updated tests (adds four freeze regression tests)

**Do not replace** `scripts/experiment_joint_rail_mowl.py`, `scripts/prepare_rail_mowl.py`, `scripts/evaluate_rail_mowl.py`, `scripts/run_all.py`, `src/module.ttl`, or any of the original imported ontologies.

## Install in PowerShell

From `C:\Users\cappa\Ontology-Tradecraft\projects\project-6`:

```powershell
cd "C:\Users\cappa\Ontology-Tradecraft\projects\project-6"
Test-Path .\scripts\experiment_el_structural_mowl.py
Copy-Item .\scripts\experiment_el_structural_mowl.py .\scripts\experiment_el_structural_mowl_before_freeze_fix.py
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\Project6_Step5_EL_Diagnostic_Freeze_Fix.zip" -DestinationPath . -Force
```

If the ZIP was downloaded to a different location, change the `-LiteralPath` argument.

## Run regression tests

```powershell
python -m unittest discover -s scripts -p "test_experiment_el_structural_mowl.py" -v
```

## Run the fast, bounded diagnostic

```powershell
python -u scripts/experiment_el_structural_mowl.py --project-root . --diagnose-only --diagnose-timeout-seconds 120
```

You should see printed stages (and progress for each import):

```text
[EL structural HH:MM:SS] Starting bounded diagnostic (timeout=120s)
[EL structural HH:MM:SS] 1/5 Validate original 80/20 split and SHA-256 provenance
[EL structural HH:MM:SS] 2/5 Create internal development split (no graph serialization)
[EL structural HH:MM:SS] 3/5 Inspect EL expressions in original source, refined and imports
[EL structural HH:MM:SS] Parsing 1/8: source (... bytes)
...
[EL structural HH:MM:SS] 5/5 Diagnostic complete: reports/step5_el_structure_diagnostic.json
```

The report is `reports/step5_el_structure_diagnostic.json`. In this *diagnose-only* mode, **no mOWL training** runs, **no training TTLs or vector files are created**, and **no mOWL normalized GCI counts are claimed**. The diagnostic only audits the input graphs and counts supported, post-filter EL expressions.

If a parser or upstream input validation truly stalls for 120 seconds, the controller forcibly kills the diagnostic child process and returns exit code **124**, printing the last completed stage. Other exceptions return exit code **2** with their reason.

## Run the full structural training only after diagnostic succeeds

```powershell
python -u scripts/experiment_el_structural_mowl.py `
  --project-root . `
  --seed 42 `
  --configs 4 `
  --tune-epochs 250 `
  --final-epochs 700 `
  --threshold 0.70
```

The full experiment **still takes significant CPU time** for multiple mOWL training sessions; only the diagnostic is bounded by the `--diagnose-timeout-seconds` option. Full preparation now prints progress and uses a stable, non-canonicalizing serializer. The experiment still needs actual local Java and mOWL to train.

## Validation and limitations

- **27 offline tests passed** on the revised code, including: no training graph writes in diagnose-only mode, deterministic intersection serialization, no whole-graph canonicalization, and explicit timeout/kill behavior.
- A synthetic 500-complex-axiom serialization check completed in approximately **0.16 seconds** on the test environment (not a claim about your project's processing time).
- The fix was not run against your complete live Project 6 imports/mOWL environment. It is not a guarantee that those sources have no other bottlenecks; the timeout and progress logging now expose them.
- This is a syntax-level EL audit; a reasoner is still needed to establish semantic nonleakage of anonymous GCIs.
- Your original source ontology, imports, baseline outputs, existing 80/20 split, and full pipeline are not edited by these scripts.
