# Merged reasoning report (ontologies + mappings)

| Pair | Reasoner | Result |
|---|---|---|
| BFO + IES | HermiT | CONSISTENT |
| BFO + IES | ELK | CONSISTENT |
| CCOM + QUDT | HermiT | INCONSISTENT |
| CCOM + QUDT | ELK | CONSISTENT |
| CCOT + OWL-Time | HermiT | NOT RUN: ontology outside OWL 2 DL (non-simple property in disjointness axiom) |
| CCOT + OWL-Time | ELK | CONSISTENT |
| QUDT alone + 1 MassUnit individual (control) | HermiT | INCONSISTENT |

## Notes
- **CCOM + QUDT / HermiT:** the control row shows QUDT 1.1 alone becomes inconsistent as soon as any unit individual exists: `Unit` requires exactly one `typePrefix` while each subclass level adds a different required value (U, USE, UPHS, UMH). Any correct CCOM->QUDT mapping classifies CCO's unit individuals into QUDT classes and so triggers it. ELK ignores cardinality/hasValue axioms, hence CONSISTENT.
- **CCOT + OWL-Time / HermiT:** OWL-Time declares a disjointness axiom on a non-simple property, which OWL 2 DL forbids; HermiT refuses to load it. ELK (OWL 2 EL) can still check the pair.
- **BFO + IES:** consistent under both reasoners.
