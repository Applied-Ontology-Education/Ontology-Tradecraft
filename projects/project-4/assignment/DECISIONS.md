# Project 4: Decisions and Debugging Log

A record for the reviewer. Everything here can be reproduced with the commands at the end.

## 1. Results at a glance

| Stage | Output | Result |
|---|---|---|
| ETL | `src/data/readings_normalized.csv` | 20 raw rows (A 6, B 8, C 6) → 15 clean rows; 5 dropped (non-numeric, blank, null) |
| RDF | `src/measure_cco.ttl` | 15 MICE, 10 SDCs, 8 artifacts, 5 units; the exact-IRI pattern ASK is true |
| SPARQL QC | 8 queries in `src/sparql/` | 0 rows each on the graph; each one catches a planted defect |
| SHACL | `src/cco_shapes.ttl` (8 NodeShapes) | Conforms; each shape catches a planted defect |
| CI | `.github/workflows/ontology_workflow.yml` | Adding `sensor_C.csv` → ETL → RDF → QC → SHACL, all green on GitHub Actions (after the detection fix in §2) |
| Grader | `.github/workflows/project4-tests.yml` | All steps pass on GitHub Actions (after the grader fix in §2) |

## 2. Debugging log: problems in the provided material

| Where | Symptom | Root cause | Fix |
|---|---|---|---|
| `normalize_readings.py` | `FileNotFoundError: sensor_C.csv` | Hard-coded `IN_C`, but sensor_C only arrives in `src/data/` at the end | Read every `*.csv` / `*.json` in `src/data/` |
| `normalize_readings.py` | Timestamps off by 4 hours | "Time (Local)" parsed as UTC | Localise to America/New_York (evidence below) |
| raw data | One asset, two ids | `Chiller 3` (A, C) vs `Chiller-3` (B); `Pump-A1 ` has a trailing space | Trim; whitespace → `-` |
| unit check | `psi` and `kPa` rejected | The grader lowercases unit labels and forbids `psi`/`kpa` | Convert pressure to Pa (exact factors); CCO has a Pascal unit but no kilopascal |
| `src/shacl/cco_shapes.ttl` | Grader: shapes file missing | Grader and README expect `src/cco_shapes.ttl` | Moved; `run_shacl_validate.py` updated |
| example shape | Would match nothing | Retired IRIs: `bfo.owl#bearer_of`, `ontologyrepository.com` CCO namespace | Current OBO / `commoncoreontologies.org` IRIs |
| `project4-tests.yml` | `ParseException: Expected SelectQuery, found 'UNION'` | A bare `FILTER NOT EXISTS {…} UNION {…}` is not valid SPARQL, so the step fails for every submission | Each FILTER in its own UNION branch, same meaning (separate commit; its own comment invites editing) |
| `.rq` stubs | Grader: queries do not parse | Comment-only files | 8 queries written |
| `ontology_workflow.yml` | Pushing `sensor_C.csv` ran green in 7 s, but ETL, RDF, QC and SHACL were all skipped | Run steps use `working-directory: projects/project-4/assignment`, and git reads pathspecs relative to the current directory, so the detect step searched `…/assignment/projects/project-4/assignment/src/data` and always found 0 added files. My first local simulation ran from the repo root and missed it | Anchor the pathspec at the repo root with `:(top)` (separate commit) |
| CI trigger | Would fail with "more than one new dataset" | `readings_normalized.csv` is also a new file under `src/data/` | Push the work first, then push the sensor_C commit separately (see §6) |

**Why America/New_York.** Sensors A and B overlap on four readings, and each pair lines up at exactly
UTC−4: Boiler-07 is 212 °F at 19:03 local in A and 100 °C at 23:03Z in B (212 °F = 100 °C), and
Chiller-3 is 68 °F at 08:01 local in A and 19.5 °C at 12:01Z in B. The site is "Buffaló-Center", and
17–19 March 2024 falls in Eastern Daylight Time.

## 3. Modelling decisions

- **Pattern (as graded).** Artifact `bearer of` SDC; MICE `is a measurement of` SDC; MICE `uses
  measurement unit` unit; MICE `has decimal value` / `has datetime value`. The grader queries without
  inference, so Material Artifact, SDC, MICE and Measurement Unit are asserted directly, next to the
  specific classes (Temperature, Pump, Degree Celsius Measurement Unit…).
- **Pressure is not `cco:Pressure`.** CCO's Pressure is a Force, which CCO puts under Process Profile,
  an occurrent. Typing a boiler's pressure as both that and the SDC the pattern requires would put one
  individual in two disjoint BFO categories. Fluid Pressure, Electric Potential Difference and
  Electrical Resistance are local BFO qualities with definitions. Temperature uses `cco:Temperature`,
  which is a quality.
- **Units.** C, F, Pa and V reuse CCO individuals. CCO has no ohm, so a local Ohm and a local
  "Measurement Unit of Electrical Resistance" class are added. Volt sits in CCO under "Measurement Unit
  of Electromagnetic Force", so the unit-matching shape pairs voltage with that class.
- **One MICE per reading.** The value is part of the MICE IRI. Without it, Boiler-07's two 23:04Z
  pressure readings (14.7 psi from A, 101.325 kPa from B) collapse into one MICE with two values once
  both are in Pa. The SHACL completeness shape caught this.
- **TBox slice.** The graph carries label, English definition and named superclasses up to BFO for
  every class and property it uses, copied from `cco_merged.ttl`, so the QC queries check real content.

## 4. Finding: the course pattern is inconsistent with CCO

Merging `measure_cco.ttl` with `cco_merged.ttl` is **inconsistent under both HermiT and ELK**. `robot
explain` gives the chain:

`uses measurement unit` ≡ inverse of `is measurement unit of` ⊑ BFO `generically depends on`, whose
range is *independent continuant and not spatial region*. So the unit individual (e.g. Pascal
Measurement Unit, an ICE and so a GDC) is inferred to be an independent continuant, and BFO makes
IC, SDC and GDC pairwise disjoint.

CCO's own pattern puts the unit and the value on the carrier: MICE `is concretized by` an Information
Quality Entity, which `inheres in` an Information Bearing Entity, and the IBE `uses measurement unit`
and `has decimal value` (whose CCO domain is Information Bearing Entity). `measure_rdflib.py
--cco-strict` writes that variant to `src/measure_cco_strict.ttl`, which is **consistent** with CCO
under HermiT and ELK. The graded file keeps the course pattern because the grader requires it.

## 5. QC queries and shapes (what "zero rows" means here)

- `all_classes_have_BFO_parent`: every non-BFO class reaches a BFO class via `rdfs:subClassOf+`.
- `annotation_whitespace`: label, definition, comment and altLabel values have no leading or trailing space.
- `min_1_eng_def`: classes and object properties have a non-empty `@en` skos:definition.
- `min_1_ontology_title`: each `owl:Ontology` has an rdfs:label.
- `no_duplicate_def` / `no_duplicate_labels`: lexical comparison, each clash reported once.
- `no_self-referential_def`: the label appears in the **definiens**. The opening definiendum
  ("A quality is …", "b is an independent continuant =Def …") is stripped first, otherwise every BFO
  elucidation is a false positive (the first draft flagged 7).
- `subclass_circularity`: a class is its own strict ancestor.
- Shapes 1–8 follow the numbered descriptions. SHACL forbids `VALUES` in SPARQL constraints, hence
  UNION/IN. The dangling-node shape ignores `rdfs:Resource`, which RDFS inference gives every node.
- `src/scripts/check_controls.py` plants one defect per query and per shape: **16/16 caught**.

## 6. Pushing (the order matters)

`ontology_workflow.yml` runs only when a push adds exactly **one** new file under `src/data/`.
`readings_normalized.csv` is new too, so:

1. push everything up to the grader-fix commit (the workflow sees no new data file and skips);
2. push the remaining commits (`sensor_C.csv` added, outputs regenerated). The workflow sees exactly one
   added file and runs.

Check that the run really executed: a run where "Run ETL" and the later steps show as *skipped* is
green but has validated nothing.

## Reproduce

```bash
cd projects/project-4/assignment
python src/scripts/normalize_readings.py
python src/scripts/measure_rdflib.py            # add --cco-strict for the CCO-consistent variant
python src/scripts/run_sparql_qc.py
python src/scripts/run_shacl_validate.py
python src/scripts/check_controls.py
```

Data-quality note, not changed: Chiller-4 reports 6.5 °F and 19.5 °F five minutes apart, which is
implausible for a chiller. It may be Celsius mislabelled at the source. Kept as reported, with the flag
recorded here.
