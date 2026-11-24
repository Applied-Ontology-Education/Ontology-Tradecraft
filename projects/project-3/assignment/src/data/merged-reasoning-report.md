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

## Inferred cross-ontology subclass axioms (ELK)

Not asserted in either ontology or the mapping files. *Reasoned* = needs an equivalence or restriction; *transitive* = follows by chaining asserted links across a mapping.

### BFO + IES: reasoned (0)
- none

### BFO + IES: transitive (12)
- Arbitrary Period SubClassOf entity
- Arbitrary Period SubClassOf occurrent
- Arbitrary Period SubClassOf spatiotemporal region
- Particular Period SubClassOf entity
- Particular Period SubClassOf occurrent
- Particular Period SubClassOf spatiotemporal region
- Period Of Time SubClassOf entity
- Period Of Time SubClassOf occurrent
- Recurring Period SubClassOf entity
- Recurring Period SubClassOf occurrent
- Recurring Period SubClassOf spatiotemporal region
- spatiotemporal region SubClassOf Thing

### CCOM + QUDT: reasoned (122)
- Acceleration Unit SubClassOf Measurement Unit
- Acceleration Unit SubClassOf Measurement Unit of Acceleration
- Amount Of Substance Unit SubClassOf Measurement Unit
- Amount Of Substance Unit SubClassOf Measurement Unit of Amount of Substance
- Angular Acceleration Unit SubClassOf Measurement Unit
- Angular Acceleration Unit SubClassOf Measurement Unit of Acceleration
- Angular Mass Unit SubClassOf Measurement Unit
- Angular Mass Unit SubClassOf Measurement Unit of Rotational Inertia
- Angular Momentum Unit SubClassOf Measurement Unit
- Angular Momentum Unit SubClassOf Measurement Unit of Momentum
- Area Unit SubClassOf Measurement Unit
- Area Unit SubClassOf Measurement Unit of Area
- Force Unit SubClassOf Measurement Unit
- Force Unit SubClassOf Measurement Unit of Force
- Frequency Unit SubClassOf Measurement Unit
- Frequency Unit SubClassOf Measurement Unit of Frequency
- Length Unit SubClassOf Measurement Unit
- Length Unit SubClassOf Measurement Unit of Length
- Linear Acceleration Unit SubClassOf Measurement Unit
- Linear Acceleration Unit SubClassOf Measurement Unit of Acceleration
- Linear Momentum Unit SubClassOf Measurement Unit
- Linear Momentum Unit SubClassOf Measurement Unit of Momentum
- Mass Per Time Unit SubClassOf Measurement Unit
- Mass Per Time Unit SubClassOf Measurement Unit of Flow
- Mass Per Time Unit SubClassOf Measurement Unit of Mass Flow Rate
- Mass Per Volume Unit SubClassOf Measurement Unit
- Mass Per Volume Unit SubClassOf Measurement Unit of Density
- Mass Unit SubClassOf Measurement Unit
- Mass Unit SubClassOf Measurement Unit of Mass
- Measurement Unit of Acceleration SubClassOf Acceleration Unit
- Measurement Unit of Acceleration SubClassOf Physical Unit
- Measurement Unit of Acceleration SubClassOf Science And Engineering Unit
- Measurement Unit of Acceleration SubClassOf Space And Time Unit
- Measurement Unit of Acceleration SubClassOf Unit
- Measurement Unit of Amount of Substance SubClassOf Amount Of Substance Unit
- Measurement Unit of Amount of Substance SubClassOf Chemistry Unit
- Measurement Unit of Amount of Substance SubClassOf Physical Unit
- Measurement Unit of Amount of Substance SubClassOf Science And Engineering Unit
- Measurement Unit of Amount of Substance SubClassOf Unit
- Measurement Unit of Area SubClassOf Area Unit
- Measurement Unit of Area SubClassOf Physical Unit
- Measurement Unit of Area SubClassOf Science And Engineering Unit
- Measurement Unit of Area SubClassOf Space And Time Unit
- Measurement Unit of Area SubClassOf Unit
- Measurement Unit of Density SubClassOf Mass Per Volume Unit
- Measurement Unit of Density SubClassOf Mechanics Unit
- Measurement Unit of Density SubClassOf Physical Unit
- Measurement Unit of Density SubClassOf Science And Engineering Unit
- Measurement Unit of Density SubClassOf Unit
- Measurement Unit of Force SubClassOf Force Unit
- Measurement Unit of Force SubClassOf Mechanics Unit
- Measurement Unit of Force SubClassOf Physical Unit
- Measurement Unit of Force SubClassOf Science And Engineering Unit
- Measurement Unit of Force SubClassOf Unit
- Measurement Unit of Frequency SubClassOf Frequency Unit
- Measurement Unit of Frequency SubClassOf Physical Unit
- Measurement Unit of Frequency SubClassOf Science And Engineering Unit
- Measurement Unit of Frequency SubClassOf Space And Time Unit
- Measurement Unit of Frequency SubClassOf Unit
- Measurement Unit of Length SubClassOf Length Unit
- Measurement Unit of Length SubClassOf Physical Unit
- Measurement Unit of Length SubClassOf Science And Engineering Unit
- Measurement Unit of Length SubClassOf Space And Time Unit
- Measurement Unit of Length SubClassOf Unit
- Measurement Unit of Mass Flow Rate SubClassOf Mass Per Time Unit
- Measurement Unit of Mass Flow Rate SubClassOf Mechanics Unit
- Measurement Unit of Mass Flow Rate SubClassOf Physical Unit
- Measurement Unit of Mass Flow Rate SubClassOf Science And Engineering Unit
- Measurement Unit of Mass Flow Rate SubClassOf Unit
- Measurement Unit of Mass SubClassOf Mass Unit
- Measurement Unit of Mass SubClassOf Mechanics Unit
- Measurement Unit of Mass SubClassOf Physical Unit
- Measurement Unit of Mass SubClassOf Science And Engineering Unit
- Measurement Unit of Mass SubClassOf Unit
- Measurement Unit of Momentum SubClassOf Mechanics Unit
- Measurement Unit of Momentum SubClassOf Momentum Unit
- Measurement Unit of Momentum SubClassOf Physical Unit
- Measurement Unit of Momentum SubClassOf Science And Engineering Unit
- Measurement Unit of Momentum SubClassOf Unit
- Measurement Unit of Power SubClassOf Mechanics Unit
- Measurement Unit of Power SubClassOf Physical Unit
- Measurement Unit of Power SubClassOf Power Unit
- Measurement Unit of Power SubClassOf Science And Engineering Unit
- Measurement Unit of Power SubClassOf Unit
- Measurement Unit of Rotational Inertia SubClassOf Angular Mass Unit
- Measurement Unit of Rotational Inertia SubClassOf Mechanics Unit
- Measurement Unit of Rotational Inertia SubClassOf Physical Unit
- Measurement Unit of Rotational Inertia SubClassOf Science And Engineering Unit
- Measurement Unit of Rotational Inertia SubClassOf Unit
- Measurement Unit of Temperature SubClassOf Physical Unit
- Measurement Unit of Temperature SubClassOf Science And Engineering Unit
- Measurement Unit of Temperature SubClassOf Temperature Unit
- Measurement Unit of Temperature SubClassOf Thermodynamics Unit
- Measurement Unit of Temperature SubClassOf Unit
- Measurement Unit of Time SubClassOf Physical Unit
- Measurement Unit of Time SubClassOf Science And Engineering Unit
- Measurement Unit of Time SubClassOf Space And Time Unit
- Measurement Unit of Time SubClassOf Time Unit
- Measurement Unit of Time SubClassOf Unit
- Measurement Unit of Volume SubClassOf Physical Unit
- Measurement Unit of Volume SubClassOf Science And Engineering Unit
- Measurement Unit of Volume SubClassOf Space And Time Unit
- Measurement Unit of Volume SubClassOf Unit
- Measurement Unit of Volume SubClassOf Volume Unit
- Measurement Unit of Volumetric Flow Rate SubClassOf Physical Unit
- Measurement Unit of Volumetric Flow Rate SubClassOf Science And Engineering Unit
- Measurement Unit of Volumetric Flow Rate SubClassOf Space And Time Unit
- Measurement Unit of Volumetric Flow Rate SubClassOf Unit
- Measurement Unit of Volumetric Flow Rate SubClassOf Volume Per Time Unit
- Momentum Unit SubClassOf Measurement Unit
- Momentum Unit SubClassOf Measurement Unit of Momentum
- Power Unit SubClassOf Measurement Unit
- Power Unit SubClassOf Measurement Unit of Power
- Temperature Unit SubClassOf Measurement Unit
- Temperature Unit SubClassOf Measurement Unit of Temperature
- Time Unit SubClassOf Measurement Unit
- Time Unit SubClassOf Measurement Unit of Time
- Volume Per Time Unit SubClassOf Measurement Unit
- Volume Per Time Unit SubClassOf Measurement Unit of Flow
- Volume Per Time Unit SubClassOf Measurement Unit of Volumetric Flow Rate
- Volume Unit SubClassOf Measurement Unit
- Volume Unit SubClassOf Measurement Unit of Volume

### CCOM + QUDT: transitive (34)
- Linear Velocity Unit SubClassOf Measurement Unit
- Measurement Unit of Electromagnetic Force SubClassOf Mechanics Unit
- Measurement Unit of Electromagnetic Force SubClassOf Physical Unit
- Measurement Unit of Electromagnetic Force SubClassOf Science And Engineering Unit
- Measurement Unit of Electromagnetic Force SubClassOf Unit
- Measurement Unit of Energy SubClassOf Mechanics Unit
- Measurement Unit of Energy SubClassOf Physical Unit
- Measurement Unit of Energy SubClassOf Science And Engineering Unit
- Measurement Unit of Energy SubClassOf Unit
- Measurement Unit of Impulse SubClassOf Mechanics Unit
- Measurement Unit of Impulse SubClassOf Momentum Unit
- Measurement Unit of Impulse SubClassOf Physical Unit
- Measurement Unit of Impulse SubClassOf Science And Engineering Unit
- Measurement Unit of Impulse SubClassOf Unit
- Measurement Unit of Pressure SubClassOf Mechanics Unit
- Measurement Unit of Pressure SubClassOf Physical Unit
- Measurement Unit of Pressure SubClassOf Science And Engineering Unit
- Measurement Unit of Pressure SubClassOf Unit
- Measurement Unit of Sound Level SubClassOf Physical Unit
- Measurement Unit of Sound Level SubClassOf Science And Engineering Unit
- Measurement Unit of Sound Level SubClassOf Unit
- Measurement Unit of Speed SubClassOf Physical Unit
- Measurement Unit of Speed SubClassOf Science And Engineering Unit
- Measurement Unit of Speed SubClassOf Space And Time Unit
- Measurement Unit of Speed SubClassOf Unit
- Measurement Unit of Torque SubClassOf Mechanics Unit
- Measurement Unit of Torque SubClassOf Physical Unit
- Measurement Unit of Torque SubClassOf Science And Engineering Unit
- Measurement Unit of Torque SubClassOf Unit
- Measurement Unit of Work SubClassOf Mechanics Unit
- Measurement Unit of Work SubClassOf Physical Unit
- Measurement Unit of Work SubClassOf Science And Engineering Unit
- Measurement Unit of Work SubClassOf Unit
- Plane Angle Unit SubClassOf Measurement Unit

### CCOT + OWL-Time: reasoned (0)
- none

### CCOT + OWL-Time: transitive (46)
- Calendar Day SubClassOf Proper interval
- Calendar Day SubClassOf Temporal entity
- Calendar Day SubClassOf Time interval
- Calendar Month SubClassOf Proper interval
- Calendar Month SubClassOf Temporal entity
- Calendar Month SubClassOf Time interval
- Calendar Week SubClassOf Proper interval
- Calendar Week SubClassOf Temporal entity
- Calendar Week SubClassOf Time interval
- Calendar Year SubClassOf Proper interval
- Calendar Year SubClassOf Temporal entity
- Calendar Year SubClassOf Time interval
- Day SubClassOf Temporal entity
- Day SubClassOf Time interval
- Decade SubClassOf Temporal entity
- Decade SubClassOf Time interval
- Gregorian Day SubClassOf Proper interval
- Gregorian Day SubClassOf Temporal entity
- Gregorian Day SubClassOf Time interval
- Gregorian Year SubClassOf Proper interval
- Gregorian Year SubClassOf Temporal entity
- Gregorian Year SubClassOf Time interval
- Hour SubClassOf Temporal entity
- Hour SubClassOf Time interval
- Julian Date SubClassOf Temporal entity
- Julian Day SubClassOf Proper interval
- Julian Day SubClassOf Temporal entity
- Julian Day SubClassOf Time interval
- Julian Year SubClassOf Proper interval
- Julian Year SubClassOf Temporal entity
- Julian Year SubClassOf Time interval
- Minute SubClassOf Temporal entity
- Minute SubClassOf Time interval
- Modified Julian Date SubClassOf Temporal entity
- Modified Julian Date SubClassOf Time instant
- Month SubClassOf Temporal entity
- Month SubClassOf Time interval
- Reference Time SubClassOf Temporal entity
- Second SubClassOf Temporal entity
- Second SubClassOf Time interval
- Time of Day SubClassOf Temporal entity
- Unix Temporal Instant SubClassOf Temporal entity
- Week SubClassOf Temporal entity
- Week SubClassOf Time interval
- Year SubClassOf Temporal entity
- Year SubClassOf Time interval

