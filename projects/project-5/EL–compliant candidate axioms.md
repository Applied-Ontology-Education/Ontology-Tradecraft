## What are OWL 2 EL–compliant candidate axioms?

In this project, “candidate axioms” are logical axioms automatically proposed by an LLM based on enriched textual
definitions. “OWL 2 EL–compliant” means those axioms use only the constructors permitted by the OWL 2 EL profile so they
can be efficiently handled by EL reasoners like ELK and by MOWL’s EL embeddings.

Allowed (typical) OWL 2 EL patterns used in this pipeline:

- Subclass axioms: C ⊑ D
- Conjunctions on the right-hand side: C ⊑ D ⊓ E
- Existential restrictions: C ⊑ ∃R.D (some values from)
- Property hierarchies: R ⊑ S
- Domain and range axioms: Domain(R, C), Range(R, D)
- Class equivalences restricted to EL-safe forms (e.g., C ≡ D ⊓ ∃R.E), though in practice we primarily emit SubClassOf
  axioms

Commonly disallowed in OWL 2 EL (therefore not emitted by the generator):

- Negation/complement (¬C) and disjoint union
- Universal restrictions (∀R.C)
- Cardinality constraints (≤ n R, ≥ n R)
- Disjunction (C ⊔ D)
- Complex property chains beyond EL-safe forms and property characteristics like inverse-functional

Concrete Turtle examples of EL-compliant candidate axioms that the generator may produce:

1) Simple subclass
   :HandTool rdfs:subClassOf :Tool .

2) Conjunctive superclass
   :PoweredDevice rdfs:subClassOf [
   a owl:Class ;
   owl:intersectionOf ( :Device :PoweredArtifact )
   ] .

3) Existential restriction (some values from)
   :BatteryPoweredDevice rdfs:subClassOf [
   a owl:Restriction ;
   owl:onProperty :hasPowerSource ;
   owl:someValuesFrom :Battery
   ] .

4) Property hierarchy
   :hasBiologicalParent rdfs:subPropertyOf :hasParent .

5) Domain and range
   :hasComponent rdfs:domain :Artifact ;
   rdfs:range  :Component .

Why EL? The EL profile guarantees polynomial-time reasoning and aligns with the capabilities of ELK and MOWL’s
ELEmbeddings. Keeping candidate axioms within EL ensures fast reasoning, predictable behavior, and compatibility with
the training/evaluation steps in this pipeline.
