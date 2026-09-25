# Project 3: Mapping Decisions and Debugging Log

This is a companion to the data files in this folder. Every decision below can be regenerated
from the notebooks. The run order is under **Reproduce** at the end.

## 1. Results at a glance

| Pair | Round 1 candidates | Round 2 candidates (after enrichment) | Accepted from candidates | Mappings found only by definition review | Merged reasoning (HermiT and ELK) |
|---|---|---|---|---|---|
| BFO – IES | 0 | 16 | 0 | 3 (Particular Period ≡ temporal interval, Period Of Time ⊑ temporal region, Event ⊑ process) | consistent; 3 inferred cross-ontology axioms |
| CCOM – QUDT | 0 | 828 | 23 (16 ≡, 7 ⊑) | 5 (QUDT broad classes ⊑ CCO Measurement Unit) | consistent; 32 inferred |
| CCOT – OWL-Time | 96 | 152 | 2 asserted + 17 entailed | 4 BFO-level mappings (temporal instant ≡ time:Instant, …) | consistent; 13 inferred (6 need the mappings) |

Round 1 found **no** candidates for BFO–IES or CCOM–QUDT, because IES and CCOM contain no OWL
restrictions: the structural matcher has nothing to compare. So the OWL enrichment (step 7) is not
optional. Axiomatising the textual definitions is what makes these ontologies comparable at all.

The families/presence-only recipe is very loose: 828 candidates collapse to 23 true matches. Here
structural matching generates candidates; the definitions decide.

## 2. Debugging log: problems found and fixed

### In the provided code

| Where | Symptom | Fix |
|---|---|---|
| `src/data/` | The first notebook fails: the folder does not exist | Created `src/data/` |
| README step 3 | `compare_structures.py --bfo … --ies …` fails: *the following arguments are required: --left, --right* | Use `--left` / `--right` (see `compare_structures.py --help`) |
| `augment_w_definitions.ipynb` | Every QUDT definition is empty | `DEF_PROPS` used the QUDT **v2** namespace (`qudt/`); `qudt.ttl` is **v1** (`qudt#description`) and also uses Dublin Core 1.1 `dc:description`. Both added |
| `reasoner_run.ipynb` | `robot` not on PATH (Windows) | Uses `robot` on PATH or `java -jar $ROBOT_JAR` |
| `reasoner_run.ipynb` | ROBOT cannot write its output | `NamedTemporaryFile(delete=True)` is locked on Windows; switched to a temporary directory |
| `reasoner_run.ipynb` | OWL-Time mapping skipped | It looked for `time-mapping.ttl`; the grader expects `to-mapping.ttl` |
| Deprecation filter | QUDT deprecated classes leaked through | QUDT marks deprecation with `rdf:type owl:DeprecatedClass`, not `owl:deprecated true`; the SPARQL filter covers both |

### In the source ontologies (all marked `P3 repair` in the TTL)

| File | Reasoner symptom | Root cause | Repair |
|---|---|---|---|
| `qudt.ttl` | 6 triples "could not be parsed"; OWLAPI invents classes `error#Error1-3` | `floatPercentage`, `integerPercentage`, `string1024` are `rdfs:Datatype`s but also `rdfs:subClassOf xsd:…`, so OWLAPI reads them as classes | Removed the three `rdfs:subClassOf xsd:…` triples (the `owl:equivalentClass` datatype restriction already defines them) |
| `qudt.ttl` | Once parsed correctly: **inconsistent** (HermiT) | `qudt:description` has range `string1024` (max length 1024). The descriptions of `AreaThermalExpansionUnit` (1,107 chars) and `VolumeThermalExpansionUnit` (1,132 chars) are too long. Because these values sit on classes (punning), they count as data assertions and violate the range | Range relaxed to `xsd:string` |
| `qudt.ttl` | **161 unsatisfiable classes** (HermiT); ELK: fine | `Unit ⊑ typePrefix exactly 1` and `Unit ⊑ typePrefix value "U"`, while every subclass adds another value (`PhysicalUnit ⊑ typePrefix value "UPHS"`). Two values under an exactly-one constraint make the class empty. Type prefixes are class-level codes written as instance-level restrictions | `exactly 1` relaxed to `min 1` |
| `qudt.ttl` imports | Inconsistent when ROBOT fetches `dtype`/`vaem` from the web | The imported schemas use `xsd:date`/`xsd:anySimpleType`, which are outside the OWL 2 datatype map | Reasoning is done on the local files with `owl:imports` stripped (also makes it reproducible) |
| `time.ttl` | HermiT refuses the file: *Non-simple property `time:disjoint` … in disjoint properties axiom*; ELK accepts it | `time:disjoint` has transitive sub-properties (`before`, `after`), so it is non-simple. OWL 2 DL forbids non-simple properties in disjoint-properties axioms | Commented out `disjoint ⟂ notDisjoint` (both directions) |

**Why HermiT and ELK disagree.** ELK implements OWL 2 EL. It ignores cardinalities, `only`, datatype
facets, inverse properties and property disjointness, so it can never see the QUDT or OWL-Time
problems above. HermiT is complete for OWL 2 DL, so it finds them. After the repairs, both reasoners
agree on every merged pair (0 differences in `reasoner-report.xlsx`). The `upstream original` rows
in that report re-run the unrepaired files to show the difference.

## 3. OWL enrichment (step 7)

Each enrichment axiom is an OWL rendering of a textual definition. The definition is quoted above
the axiom in the TTL, between the `BEGIN/END Project 3 OWL enrichment` markers at the end of each file.

* **BFO**: `process ⊑ has participant some material entity`, `history ⊑ history of some material entity`,
  plus the dependence patterns of SDC, realizable entity and GDC. Each comes from BFO's own elucidation.
* **IES** (published as RDFS only): Event has participants, EventParticipant `isParticipantIn some Event`
  and `isParticipationOf some Entity`, State `isStateOf some Element`, ParticularPeriod has its
  required ISO-8601 representation.
* **CCOM ↔ QUDT**: every CCOM definition reads "used as a standard for measurement of X". Both sides
  now carry `quantityKind value qk:X`, using QUDT v2 quantity-kind IRIs because the v1 schema ships no
  quantity-kind vocabulary. That filler is what makes the candidate review decidable.
* **CCOT ↔ OWL-Time**: CCOT unit intervals get OWL-Time durations (Hour: `hasDuration some (Duration and unitType value unitHour and numericDuration value 1)`).
  Gregorian Day/Year get a `DateTimeDescription`. OWL-Time `ProperInterval` gets a beginning and an end
  instant, and `DateTimeInterval` gets a `DateTimeDescription`.

## 4. Key modelling decisions

1. **3D vs 4D (BFO – IES).** IES is a 4D (BORO) ontology in which every Entity is also its own
   whole-life State (`ies:Person ⊑ ies:Entity` *and* `ies:Person ⊑ ies:PersonState ⊑ ies:State`).
   Mapping `ies:State → occurrent` and `ies:Person → object` together makes `ies:Person`
   unsatisfiable. So I mapped only the **occurrent side** (Event, PeriodOfTime, ParticularPeriod),
   where a 4D spatio-temporal extent and a BFO occurrent really do line up. No continuant-side IES
   classes are mapped.
2. **Descriptions are not time (CCOT – OWL-Time).** Most round-2 candidates pair a CCOT temporal
   region (time itself) with an OWL-Time *description* (`TemporalPosition`, `DateTimeDescription`,
   `MonthOfYear`, `Duration`). Those are category errors and are rejected, even for pairs whose names
   look alike, such as CCOT *Year* and OWL-Time *Year*.
3. **Targeted vs global.** Instead of asserting 20+ CCOT leaf mappings, only
   `temporal interval ⊑ time:ProperInterval` and `temporal instant ⊑ time:Instant` are asserted. The
   reasoner places every CCOT interval and instant.
4. **Paired subclass files.** `ParticularPeriod ⊑ temporal interval` sits in `ies-mapping.ttl` and the
   converse in `bfo-mapping.ttl`. Neither file asserts an equivalence; the reasoner infers
   `ParticularPeriod ≡ temporal interval` from the merged pair. The same holds for
   `temporal instant ≡ time:Instant`.
5. **A hidden CCOT commitment.** The reasoner derives `Multi-Hour Temporal Interval ⊑ temporal interval`
   from CCOT alone, not from any mapping. CCOT asserts these classes only as one-dimensional temporal
   regions, which may have gaps, but `interval contains` has domain *temporal interval*, which is
   gap-free. See the `requires_mapping = False` rows in `reasoner-report.xlsx`.

## 5. Known limitations

* `compare_structures.py` does not exclude deprecated classes: OWL-Time `January` and `Year` appear as
  candidates. The review rejects them.
* The QUDT file is a v1 schema subset. The quantity-kind individuals use QUDT v2 IRIs.
* Not axiomatised or mapped (the definition is too vague or there is no counterpart): CCOM Area Moment
  of Inertia (its definition describes a mass distribution, but the quantity is the second moment of
  area), Electromagnetic Force, Flow, Sound Level.

## Reproduce

From `projects/project-3/assignment/`, with ROBOT (`robot` on PATH or `ROBOT_JAR`) and Java installed:

1. `notebooks/week5_endpoint_client.ipynb`: `*-class.xlsx`
2. `notebooks/class-axiom-generator.ipynb`: `*-axioms.xlsx`
3. `python src/compare_structures.py --left src/<a>.ttl --right src/<b>.ttl --outdir src/data/ --shape coarse --presence-only --normalize families` for (bfo-core, ies), (ccom, qudt) and (ccot, time)
4. `notebooks/augment_w_definitions.ipynb`: `*-structural-matches-with-defs.xlsx`
5. `notebooks/mapping_review.ipynb`: `mapping-review.xlsx`
6. `notebooks/reasoner_run.ipynb`: `reasoner-report.xlsx`
7. `pytest -q`
