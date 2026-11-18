# Project 6 — Passenger Manifest to Ontology Instances (first ABox stage)

## What this stage does

Reads a validated `.xlsx` workbook with `Air_Manifest` and `Rail_Manifest` worksheets, maps the rows to **existing** classes and predicates in `generated/module_augmented.ttl`, and emits new named individuals in a separate file. No classes, object properties, data properties, or automatic risk/Person-of-Interest labels are created. Both `src/module.ttl` and `generated/module_augmented.ttl` remain untouched.

The workbook includes **fictional demonstration passengers only**. It is not an actual flight/train manifest and does not establish that any individual traveled.

## Install

Unpack the ZIP at the root of `C:\Users\cappa\Ontology-Tradecraft\projects\project-6`:

```
project-6/
  scripts/
    ingest_passenger_manifest.py
    test_ingest_passenger_manifest.py
  templates/
    passenger_manifest_template.xlsx
```

The only Python dependency is `rdflib` (already in `.venv-mowl`). The script reads `.xlsx` using built-in `zipfile` and XML modules; no additional spreadsheet package is needed.

Run these commands in the **project-6 root** with `.venv-mowl` active:

```powershell
python -m unittest discover -s scripts -p "test_ingest_passenger_manifest.py" -v
python scripts/ingest_passenger_manifest.py --input templates/passenger_manifest_template.xlsx --validate-only
python scripts/ingest_passenger_manifest.py --input templates/passenger_manifest_template.xlsx
```

For an authorized manifest, make a copy of the workbook and replace synthetic records with real permitted data, then execute:

```powershell
python scripts/ingest_passenger_manifest.py --input data/passenger_manifest.xlsx
```

You can override the ontology and output paths using `--ontology`, `--output`, `--combined-output`, and `--report`.

## Workbook requirements

* Both sheets must exist with exact headers in row 1 and the prescribed column order; either sheet may have no passenger rows.
* One row represents a *manifest entry* for one passenger on one trip.
* `person_id` must be a stable upstream identifier; the script never merges people merely because names match. A `person_id` appearing in both air and rail sheets produces one person individual.
* `trip_id` identifies **one occurrence** of a flight or railway service, not merely a recurring flight/train number. Reuse that ID consistently for all passengers on the same departure.
* `entry_id` is unique per sheet/mode. Ticket IDs and boarding-pass IDs, when present, are unique in their respective sheets.
* IDs must be stored as **Excel Text**, preserving leading zeros. Avoid formulas and spreadsheet error cells; the script rejects these.
* Times must be textual ISO 8601 with a UTC offset, such as `2026-11-19T09:00:00-05:00`. In Excel you can prefix a single apostrophe to force text. Automatic Excel serial dates are rejected because they lose time-zone information.
* `journey_confirmed` is a rail-only `TRUE`/`FALSE` field. Use `TRUE` only with external confirmation that an actual rail journey took place. An upcoming reservation is not evidence of an actual BFO Process.
* `reservation_id` is optional. `boarding_pass_id` / `ticket_id` are optional and otherwise based on a stable entry ID.
* Blank rows are ignored; unsupported extra columns, conflicting person names, conflicting trip route/time, or duplicate boarding pass / ticket identifiers fail fast.

## Outputs

```
generated/manifest_instances.ttl              # ABox only; NEW named individuals
generated/module_augmented_with_manifest.ttl  # exact augmented TTL bytes + ABox
reports/manifest_ingestion_report.json        # counts, hashes, warnings, no passenger names
```

The first output is suitable for separate Protégé inspection and future ontology import. The second is a convenient fully combined file for SPARQL and reasoning checks; it does **not** overwrite `module_augmented.ttl`.

All generated resources use a separate namespace `https://example.org/ontologies/intel-ontology/manifest#` and deterministic, SHA-256-derived opaque keys. **Hashes are not anonymization.** Person names and reservation numbers are still stored as literals in the resulting TTL. Protect the workbook and generated TTL accordingly.

## Mapping to the actual augmented ontology

| Excel field / record | Ontology entity | Mapping type |
|---|---|---|
| Person record | `AgentOntology/person` | `rdf:type` named individual |
| Passenger to boarding pass (air) | `AgentOntology/possesses` | Object property |
| Boarding pass (air) | `AgentOntology/boarding_pass` | Physical bearer individual |
| Boarding pass to airline reservation info | `obo:BFO_0000101` (`is carrier of`) | Object property |
| Airline reservation info | `AgentOntology/airline_reservation` | Information content individual |
| `first_name`, `last_name`, `reservation_id` | `AgentOntology/has_first_name`, `has_last_name`, `has_reservation_number` | Data properties on reservation information |
| `boarding_gate`, `boarding_group`, `boarding_time` | `has_boarding_gate`, `has_boarding_group`, `has_boarding_time` | Data properties on airline reservation information |
| `departure_name`, `arrival_name` (air) | `has_departure_airport`, `has_arrival_airport` | Data properties on reservation info and flight plan |
| Air trip | `AgentOntology/Flight_plan` | Plan ICE individual; flight number in label; no flight occurrence inferred |
| `aircraft_tail_number` (air, if provided) | `AgentOntology/Commcercial_airplane`, `has_tail_number` | Physical airplane instance and data property; follows **existing misspelled class IRI** |
| Person to rail ticket | `AgentOntology/possesses` | Object property |
| Rail ticket | `intel:rail_ticket` | **Physical** material information bearer individual |
| Rail ticket to reservation information | `obo:BFO_0000101` (`is carrier of`) | Object property |
| Rail reservation info | `intel:rail_reservation_information` | ICE individual |
| Rail reservation to plan/actual journey | `obo:IAO_0000136` (`is about`) | Object property |
| Rail journey plan | `intel:rail_journey_plan` | ICE individual |
| `departure_time` (rail) | `intel:has_planned_departure_time` | `xsd:dateTime` data property on plan |
| `service_code` (rail) | `intel:has_train_service_code` | String data property on passenger train |
| Confirmed rail journey | `intel:rail_journey` | BFO Process individual only if verified |
| Departure, arrival, train links (confirmed rail) | `intel:departs_from_station`, `intel:arrives_at_station`, `intel:occurs_aboard_train` | Object properties from journey to railway station / train |
| Plan to confirmed rail journey | `intel:specifies_rail_journey` | Object property |

For air, `service_code`, departure time, airport IDs, and airplane assignment cannot yet all be expressed as connected object/data properties because there is no **flight occurrence** class/property linking passenger, travel, aircraft, and flight plan. The script does not invent such predicates. The same holds for linking person to a Passenger or Rail Passenger Role: without a verified bearer-of relation available in the loaded ontology, that role is not instantiated. These are **candidate TBox enhancements** for a later, separately reasoned extension.

## Follow-up development stages

1. Validate with a real authorized manifest layout and build a deterministic *field-mapping configuration* for multiple airlines/rail providers.
2. Add SHACL constraints for instance cardinality/typing, including correct use of actual versus planned journey data.
3. Propose missing flight occurrence and passenger-bears-role object properties as a **new TBox revision**, subject to review and HermiT.
4. Add a source-scoped OWL reasoner pass for the combined ABox + TBox output.
5. Integrate with `scripts/run_all.py` only after those steps pass. No Grok/mOWL retraining is required for routine manifest ingestion.

**Privacy:** This is a data transformation utility, not a system for inferring threat, criminality, or Person-of-Interest status from manifest entries. Only load data you are authorized to process.
