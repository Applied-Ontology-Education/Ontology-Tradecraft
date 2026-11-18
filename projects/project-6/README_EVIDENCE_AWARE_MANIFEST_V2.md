# Project 6 — Evidence-aware passenger manifest v2

## Purpose

This version **replaces** three Python scripts and the *synthetic Excel template* so that booking confirmation no longer causes the importer to instantiate an actual BFO Rail Journey process. The unchanged Project 6 source and `generated/module_augmented.ttl` are read-only inputs.

**Contents (relative to `project-6/`):**

- `scripts/ingest_passenger_manifest.py`
- `scripts/test_ingest_passenger_manifest.py`
- `scripts/enrich_manifest_relationships.py`
- `scripts/test_enrich_manifest_relationships.py`
- `scripts/validate_manifest_abox.py`
- `scripts/test_validate_manifest_abox.py`
- `templates/passenger_manifest_template.xlsx` (synthetic sample)

**Back up any hand-edited workbook before installing**: extracting the archive with `-Force` overwrites the file of the same name. Do not put real passenger information into Git.

## Excel contract changes

`Air_Manifest` is unchanged. `Rail_Manifest` replaces the old `journey_confirmed` column with:

| Column | Allowed / expected value |
|---|---|
| `booking_status` | `confirmed`, `pending`, `cancelled` |
| `journey_status` | `planned`, `in_progress`, `completed`, `cancelled` |
| `occurrence_evidence` | Empty for planned/cancelled; `operator_departure_record` or `vehicle_tracking_record` for in_progress; `operator_arrival_record` or `operator_completion_record` for completed |
| `evidence_reference` | External operator record identifier, required for in_progress/completed; otherwise empty |
| `physical_ticket_evidence_reference` | External evidence reference that a **material** rail ticket exists; without this no physical Rail Ticket instance is created |

Two manifest rows for the same rail trip must agree on journey status, evidence kind, and evidence reference; booking status and ticket evidence are passenger-specific. IDs, evidence references, and ISO-8601 timestamps must be Excel TEXT, without formulas. Times require UTC offsets.

**IMPORTANT LIMITATIONS:** An evidence reference is *attested by the workbook*; the script does not authenticate the operator system or verify real-world departure. The evidence reference is linked to the `rail_journey_plan` ICE through two newly declared data properties, not mistaken for an event or person. Physical ticket evidence reference is linked to the `rail_ticket` material bearer through a new data property. These three properties are declared in a **separate enrichment TBox**, never in the protected augmented ontology.

`--occurred-rail-trip` is now an **optional cross-check only** and cannot override workbook evidence. No CLI override is required to create supported process assertions.

## Installation on Windows (from project-6 root)

```powershell
cd "C:\Users\cappa\Ontology-Tradecraft\projects\project-6"
Copy-Item .\templates\passenger_manifest_template.xlsx .\templates\passenger_manifest_template_v1_backup.xlsx -ErrorAction SilentlyContinue
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\Project6_Manifest_Evidence_Aware_v2.zip" -DestinationPath . -Force
```

Verify installation before running:

```powershell
Test-Path .\scripts\ingest_passenger_manifest.py
Test-Path .\scripts\enrich_manifest_relationships.py
Test-Path .\scripts\validate_manifest_abox.py
Test-Path .\templates\passenger_manifest_template.xlsx
```

### 1 — Unit tests (all offline)

```powershell
python -m unittest discover -s scripts -p "test_ingest_passenger_manifest.py" -v
python -m unittest discover -s scripts -p "test_enrich_manifest_relationships.py" -v
python -m unittest discover -s scripts -p "test_validate_manifest_abox.py" -v
```

### 2 — Ingestion (must rerun; old generated ABox no longer matches new contract)

```powershell
python scripts/ingest_passenger_manifest.py --input templates/passenger_manifest_template.xlsx --validate-only
python scripts/ingest_passenger_manifest.py --input templates/passenger_manifest_template.xlsx
```

Synthetic sample: **5 rows, 4 people, 23 individuals, 113 ABox triples, 597 combined triples, zero actual Rail Journey processes**. Two rail booking rows are confirmed but travel status is `planned`. A third rail row has no physical ticket evidence, so no material Rail Ticket is invented for it.

### 3 — Relationship enrichment

```powershell
python scripts/enrich_manifest_relationships.py --input templates/passenger_manifest_template.xlsx --flight-document FL-20261119-AA101=FPDOC-AA101 --validate-only
python scripts/enrich_manifest_relationships.py --input templates/passenger_manifest_template.xlsx --flight-document FL-20261119-AA101=FPDOC-AA101
```

`--flight-document` is synthetic explicit evidence for a physical document; **omit it** for real data unless independently established. Synthetic output: 50 schema triples (10 properties: seven data, three object), 139 enriched ABox triples, **673 combined triples**, 0 rail train process participation assertions.

### 4 — SHACL and source-scoped HermiT

```powershell
python scripts/validate_manifest_abox.py --workbook templates/passenger_manifest_template.xlsx --shacl-only --strict-occurrence
python -u scripts/validate_manifest_abox.py --workbook templates/passenger_manifest_template.xlsx --strict-occurrence
Get-Content .\reports\manifest_semantic_validation.json
```

The SHACL command needs `pyshacl`. The reasoning command additionally needs your local ROBOT and Java. The local build environment used for packaging did **not** include ROBOT or pySHACL, so the external reasoning run must be executed in `.venv-mowl` on your Windows system. The 13 validator unit tests use synthetic/mocked checks and are not a substitute for real SHACL/HermiT execution.

## Safety and provenance

- Only authorized, access-controlled manifest inputs may be used. Keep live manifests, ABox output, logs, and reports out of version control.
- Both input ontology files remain unchanged. The generated files are overwritten intentionally on fresh **manifest** runs, not the original ontology.
- This is not a threat-risk assessment and creates no Person of Interest/threat classifications.
- A booked ticket, scheduled service, or confirmed reservation is not sufficient evidence of an occurring BFO process.
- A ticket ID without physical evidence is not a physical bearer.
- `source-scoped HermiT` validates an extracted local import module, not full import closure or truth of reported evidence.
- Existing `scripts/run_all.py` still handles the ontology-TBox pipeline. This package does **not yet** edit or integrate with that top-level orchestrator.
