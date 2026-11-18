# Project 6 — Step 2 / Manifest vocabulary contract (v3)

## Root cause

The November 18 run passed rail TBox Steps 1–7 (421 source + 63 added = 484),
but `08_manifest_ingestion` failed because its v3 importer requires the
`departs_from_station` object property. The Step 2 v2 contract checked **only
four object plus two datatype properties**, permitting Grok to use different
IRIs (`has_departure_station`, `has_arrival_station`, `has_conveying_train`,
`has_intermediate_stop`, `has_train_identifier`, `has_rail_service_code`).
Those six names are structurally valid as a standalone rail TBox but are not
compatible with the existing, validated v3 manifest ingestion semantics.

## Fix

The Step 2 script now validates exactly **six named predicates and their RDF
domains/ranges**, in addition to eight class definitions and the existing
Step 3A subclass requirements:

| Required local name | OWL property | Domain | Range |
|---|---|---|---|
| `departs_from_station` | Object | `rail_journey` | `railway_station` |
| `arrives_at_station` | Object | `rail_journey` | `railway_station` |
| `occurs_aboard_train` | Object | `rail_journey` | `train` |
| `specifies_rail_journey` | Object | `rail_journey_plan` | `rail_journey` |
| `has_planned_departure_time` | Datatype | `rail_journey_plan` | `xsd:dateTime` |
| `has_train_service_code` | Datatype | `passenger_train` | `xsd:string` |

Grok still supplies candidate descriptions, English labels, definitions,
and rationales, but may not rename these preapproved concepts or alter their
domains and ranges. It receives these exact fields in its request; if a design
fails the contract, the script tries a second **fresh Grok request** by default
and records the rejected attempt. It never silently remaps, merges, or
reclassifies incompatible predicates. If both requests are incompatible,
Step 2 fails closed before OWL generation.

## Install (PowerShell; Project 6 root)

```powershell
cd "C:\Users\cappa\Ontology-Tradecraft\projects\project-6"
Copy-Item .\scripts\propose_rail_patterns_llm.py .\scripts\propose_rail_patterns_llm_before_manifest_fix.py
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\Project6_Step2_Manifest_Vocabulary_Contract_Fix.zip" -DestinationPath . -Force
```

**Only** `scripts/propose_rail_patterns_llm.py` and
`scripts/test_propose_rail_patterns_llm.py` are replaced. The source
`src/module.ttl`, six imports, workbook, runner, candidate/refinement scripts,
manifest scripts, and all generated outputs are **not** overwritten by
installation.

## Test the fix without calling Grok

```powershell
python -m unittest discover -s scripts -p "test_propose_rail_patterns_llm.py" -v
python -m unittest discover -s scripts -p "test_run_all.py" -v
```

Expected: **15 Step 2 tests** and **16 run_all tests**, both `OK`.

## Rebuild everything from source

```powershell
python -u scripts/run_all.py --preflight-only --manifest templates/passenger_manifest_template.xlsx
python -u scripts/run_all.py --manifest templates/passenger_manifest_template.xlsx --flight-document FL-20261117-AA101=FPDOC-AA101
```

The existing runner archives prior intermediates, invokes a **new Grok
proposal**, regenerates the entire rail TBox and ABox, and validates the
populated output with SHACL and source-scoped HermiT. Previous generated
`module_augmented.ttl` is intentionally *not* reused.

Expected structural counts for the sample workbook, if the full pipeline
succeeds: 421 original triples + 63 rail TBox additions = 484;
142 manifest ABox triples; 54 manifest-specific TBox triples; enriched ABox
171 triples; **709 total** with the optional physical flight-plan document.
The live result may use different English definitions, and exact RDF hashes
need not match earlier runs.

## Tests performed when preparing this patch

- 15 Step 2/3/3A unit tests, including malformed-property retry and a mock
  62 -> 63 triple candidate/refinement chain.
- 16 runner unit tests with synthetic v3 ingestion and enrichment fixtures.
- An offline integration check adding a mock-compliant 63-triple rail TBox to
  the uploaded 421-triple source, then actually importing the v3 Excel workbook,
  enriching and checking the resulting 709-triple ontology.

**Not performed:** authenticated Grok requests, full-import/global OWL
consistency, live ROBOT/HermiT, or pySHACL (not installed here). The real
Windows run is needed to confirm these.

This is an intentionally restricted, *reviewed* rail design. It does not
permit the LLM to explore new property identities freely in this pipeline.
An exploratory variant would need separate downstream mapping and validation.
