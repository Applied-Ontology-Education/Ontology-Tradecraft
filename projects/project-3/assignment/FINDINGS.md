# Notes on the structural matching results

For each pair, I used `compare_structures.py` to find the classes that matched best, then checked each candidate by hand against its definitions before deciding whether to assert anything.

**BFO and IES.** IES had no `owl:Restriction` nodes at all (confirmed two ways: rdflib and ROBOT's parser), so there was nothing to match on at first. I went back and formalized four IES definitions into real restriction axioms (Account Holder, Money Transfer, Responsible Actor State, Radio Mast), matching BFO's own `allValuesFrom` style. That produced 76 candidates. Reading them, they're all my four classes paired against BFO's most abstract categories (spatial region, process, continuant, and similar), a structural coincidence rather than a real conceptual match, so I didn't assert equivalence.

**CCOT and TO.** `Year`/`Year` and `Month`/`Month of year` looked like strong candidates by label. But CCOT's `Year` and `Month` are temporal intervals (spans of time), while TO's are a duration and a calendar label, respectively. Different kinds of things sharing a name, so no equivalence asserted.

**CCOM and QUDT.** My CCOM-side matches had no definitions at first because they're actually imported from Common Core Ontologies, not defined in `ccom.ttl` itself. I fixed the definitions lookup to follow that import (also fixed a QUDT namespace bug). With real definitions visible, the pattern held: structural matches on abstract bookkeeping categories, not real semantic overlap.

Across all three, the lesson was the same: structural similarity isn't semantic equivalence, so I checked each candidate before asserting anything rather than taking a match at face value.

**One real mapping, found outside the structural matcher.** Everything above came from `compare_structures.py`'s candidates, which kept turning out to be structural coincidences once I read the actual classes. Separately, just reading IES's and BFO's top-level definitions directly, two did hold up: IES's `Event` ("an activity or incident... that occurred/started at a specific point in time") matches BFO's `process`, and IES's `Entity` ("a tangible thing like a Person, a Communications Device, or a Location") matches BFO's `independent continuant` (not `material entity`, since IES's own example of a Location isn't made of matter). Both are asserted as `rdfs:subClassOf` rather than `owl:equivalentClass`, since in both cases the BFO class is broader than the IES one, not identical to it. These are in `src/data/ies-mapping.ttl`.
