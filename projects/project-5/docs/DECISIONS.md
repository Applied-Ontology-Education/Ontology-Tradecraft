# Decisions

## Split (step 4)

### Input module keeps its original name
The input module is kept as `src/EventOntology.ttl` rather than renamed to
`src/cco-module.ttl` as the README's file list suggests. All scripts read it from
that path via `config.yaml`, and it is treated as read-only.

### Training data includes BFO-core
`src/train.ttl` = the module (minus held-out pairs and `owl:imports`) + a local
copy of BFO-core 2020 (`src/bfo-core.ttl`). BFO gives the embeddings upper-level
structure (hierarchy, disjointness, property domains). It is merged from a local
file rather than resolved as an import, so runs work offline and are
reproducible. The InformationEntityOntology import is excluded so the embedding
space stays about this module. Held-out pairs come only from the module's own
axioms, never from BFO.

About half of BFO-core's logical content is outside OWL 2 EL (`only`
restrictions, unions, complements, inverse and functional properties). ELK and
mOWL's normalizer ignore those axioms, so only the EL portion reaches training.

### Holdout size: 20% of all named subclass pairs
We hold out 66 pairs: 20% of all 328 named `A ⊑ B` pairs, rather than 20% of the
~115 pairs that are eligible on their own (~23). This matches the README's "≈20%" and
makes `mean_cos` more stable: its standard error is roughly 1.7× smaller with 66
pairs than with 23.

### Limitation: mostly internal classes are tested
Because a held-out class must still appear in training, leaf classes with no
other axioms (about 200 pairs) can never be held out. The held-out set is
therefore mostly pairs whose subclass has its own subclasses or an existential
restriction. `mean_cos` measures how well the model recovers links for these
better-connected classes, not for leaves.

### Declaring four CCO object properties
Four CCO object properties used in the module's restrictions (`ont00001777`,
`ont00001803`, `ont00001819`, `ont00001936`) are declared in imported modules
that we don't merge. The split adds `owl:ObjectProperty` declarations for them
to `train.ttl` so the OWL API parses those axioms reliably. No new IRIs are
created; these IRIs already appear in the module. A mOWL smoke test confirmed
all four reach the normalized GCIs.

## Evaluation (step 5)

### Mean cosine, not ranking metrics
The README specifies mean cosine similarity over held-out subclass–superclass
pairs, with a 0.70 threshold. The ontology-embedding literature usually reports
ranking metrics for subsumption prediction (Hits@k, mean rank, MRR). We report
mean cosine because the assignment requires it, with these known limitations:
- Cosine compares two class vectors, but ELEmbeddings models classes as balls
  (a center plus a radius). Cosine looks only at the centers' directions and
  ignores the radius and the containment geometry the model is trained for.
- For candidates of the form `A ⊑ ∃R.B` (step 6), cos(A, B) ignores R entirely.
We also report `baseline_mean_cos` on random non-subsumption pairs, so a high
score caused by collapsed embeddings can be detected.
