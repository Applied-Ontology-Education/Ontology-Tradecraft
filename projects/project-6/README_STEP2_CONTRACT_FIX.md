# Project 6 — Step 2 Grok design-contract fix (v2)

## Why this fixes the current failure

The original `propose_rail_patterns_llm.py` allowed **2–4 object properties and 0–2 data properties** while `refine_rail_axioms.py` requires **exactly eight CREATE classes, four object properties and two data properties**. A live Grok response with only one data property produced 57 candidate triples and caused Step 3A to fail.

This fix updates Step 2 **only**. It does not change `run_all.py`, `generate_rail_axioms.py`, `refine_rail_axioms.py`, `src/module.ttl`, `imports/` or any ontology outputs.

## Updated behavior

1. Step 2 sends an explicit eight-class / six-property design contract, including the previously approved direct superclass choices required by Step 3A.
2. Step 2 validates each returned design *before* writing `generated/rail_pattern_proposal.json` and its PASS report.
3. A valid design proceeds directly to Step 3 and Step 3A. An incomplete design is rejected; Step 2 can make **one additional fresh Grok request** by default, with specific corrective feedback. Calls are bounded, no unchecked generated fields are silently invented.
4. If the second candidate also fails, Step 2 fails safely. Inspect `reports/step2_rejected_attempts.json`. No incomplete proposal is accepted.
5. Rejected-attempt explanations and the accepted proposal's attempt count are recorded as provenance. No API key is logged.
6. The original source and imports are never edited.

## Install from Project 6 root in PowerShell

```powershell
cd "C:\Users\cappa\Ontology-Tradecraft\projects\project-6"
Copy-Item .\scripts\propose_rail_patterns_llm.py .\scripts\propose_rail_patterns_llm_backup.py
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\Project6_Step2_Grok_Contract_Fix.zip" -DestinationPath . -Force
```

The ZIP contains only:

```
scripts/propose_rail_patterns_llm.py
scripts/test_propose_rail_patterns_llm.py
README_STEP2_CONTRACT_FIX.md
```

## Test without Grok API requests

```powershell
python -m unittest discover -s scripts -p "test_propose_rail_patterns_llm.py" -v
```

Expected: **11 tests, OK**. These cover the exact missing-data-property case, bounded retry, exhaustive rejection without a valid proposal, source preservation, and a synthetic complete Step 2 → Step 3 → Step 3A chain (62 → 63 triples). Your existing `test_run_all.py` should continue to pass.

## Re-run entire pipeline from original source

```powershell
python -u scripts/run_all.py --manifest templates/passenger_manifest_template.xlsx --flight-document FL-20261117-AA101=FPDOC-AA101
```

The pipeline archives prior intermediate files before rerunning. The Grok API is contacted again, so cost and output variability apply. Each of the two design attempts is a complete generation request; this is not a post-hoc patch. You can adjust the number of attempts when invoking Step 2 directly:

```powershell
python scripts/propose_rail_patterns_llm.py --max-design-attempts 3
```

When using `run_all.py`, it invokes Step 2 with its default of 2 attempts.

## Important caveat

These constraints intentionally reflect your previously reviewed rail design. They make the output compatible with the downstream **fixed** refinement script, but do not let Grok choose arbitrary class identities or direct parents. A controlled new design would require updating the downstream refinement policy, not bypassing validation. Passing the contract and SHACL/HermiT does not establish that every asserted travel fact is true.
