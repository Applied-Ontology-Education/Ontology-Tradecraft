# Project 6 — Step 6 Recovery: Windows deadline + stage diagnostics

This package **replaces only** `scripts/validate_rail_reasoner.py` and adds the new offline test `scripts/test_step6_timeout_fix.py`. It leaves `src/module.ttl`, imports, refined candidate, Steps 1–5C, and reports unchanged until Step 6 is run. Back up the original Step 6 script before extraction.

## Why the original could run for hours

Two concrete issues in Step 6 v1:

1. Ontology import canonicalization (`rdflib.compare.to_canonical_graph`) and the subsequent *whole-graph* `isomorphic` checks were outside the `--timeout` guarding ROBOT. Preparation could therefore run indefinitely from the perspective of the script.
2. Python's subprocess standard-output pipes may remain open through child Java processes on Windows. Killing the direct ROBOT wrapper after a timeout does not always close inherited pipes, so `subprocess.run(..., capture_output=True, timeout=...)` may still hang while draining output.

The original status cannot be diagnosed conclusively without its last output files, so this package makes both failure modes observable.

## v2 changes

- Prints **timestamped progress** while verifying, parsing each import, canonicalizing each graph, serializing, and invoking each reasoner.
- Builds the baseline only once and adds the 63 candidate triples for the second ontology; avoids canonicalizing all imports twice.
- Replaces expensive `isomorphic` checks with **limited, explicit** parsing checks: triplet count and named subclass set preservation. This is faster but **not** a complete graph-isomorphism check; full semantic verification remains with ROBOT. The original source and candidate SHA-256 checks remain.
- Captures ROBOT output in logs **as it runs**, not in stdout/stderr pipes. On a timeout, terminates the **Windows process tree** using `taskkill /T /F`.
- Adds a **parent watchdog** limiting the **entire step**, including preprocessing; it kills the Python/ROBOT/Java descendant process tree. This is independent of the per-ROBOT timeout.
- Adds `--prepare-only` to test the slow import/graph phase without launching Java.
- A timeout or error **cannot** produce a successful reasoner status.
- No candidate merge, automatic semantic approval, ontology edit, or named individual generation.

## Installation (PowerShell in projects/project-6)

Stop the previously hung Step 6 invocation first (`Ctrl+C`). If Java remains running, identify the process before stopping it.

```powershell
Copy-Item .\scripts\validate_rail_reasoner.py .\scripts\validate_rail_reasoner_step6_v1_backup.py
```

Extract the ZIP into `projects/project-6` and **replace** `scripts/validate_rail_reasoner.py` when prompted. The second file `scripts/test_step6_timeout_fix.py` is new. Keep the v1 backup separate from discoverable test names.

Run the original Step 6 regression tests:

```powershell
python -m unittest discover -s scripts -p "test_validate_rail_reasoner.py" -v
```

Run the new timeout checks:

```powershell
python -m unittest discover -s scripts -p "test_step6_timeout_fix.py" -v
```

## Stage A: preparation-only diagnostic (3-minute absolute limit)

```powershell
python -u scripts/validate_rail_reasoner.py --prepare-only --hard-timeout 180
```

Expected **if** preparation succeeds:

```
STEP 6 — PASS_PREPARATION_ONLY_UNREASONED
```

Creates `reports/step6_prepare_summary.json` and the two flattened input TTL files. It **does not run HermiT** and is **not a reasoner pass**. If this times out, the final timestamped console line will tell us which specific import or processing stage is slow.

## Stage B: actual HermiT reasoning (10-minute absolute limit)

Only after Stage A passes:

```powershell
python -u scripts/validate_rail_reasoner.py --timeout 180 --hard-timeout 600
```

`--timeout 180` limits **each** ROBOT process to three minutes; `--hard-timeout 600` caps the full command at ten minutes, including graph preparation and any hang in Python/Java cleanup. You can change these for more generous resources once the bottleneck is identified.

If baseline reasoning times out, the command fails without running the candidate. Success requires BOTH actual ROBOT runs to complete with valid output.

Inspect:

```powershell
Get-Content .\reports\step6_reasoner_summary.json
Get-Content .\reports\step6_baseline_robot.log -Tail 60
Get-Content .\reports\step6_candidate_robot.log -Tail 60
Get-Content .\reports\step6_watchdog_status.json  # only created on hard timeout
```

**Important:** Files from an older or interrupted Step 6 run may still exist. Trust only reports timestamped from the latest run and matching current input hashes; never regard a stale `step6_reasoner_summary.json` as a fresh pass. No ontology approval is implied by a reasoning pass.

**Environment testing:** Offline regression tests pass in a Linux container. An actual Windows/Java/ROBOT invocation against the user's six local import files has not been executed here.
