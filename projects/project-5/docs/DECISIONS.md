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
- For candidates of the form `A ⊑ ∃R.B` (step 6), cos(A, B) would ignore R
  entirely. Step 6 therefore scores those differently (see "Scoring existential
  candidates" below).
We also report `baseline_mean_cos` on random non-subsumption pairs, so a high
score caused by collapsed embeddings can be detected.

### Reporting mean_cos two ways
The README asks for the smallest τ ∈ {0.60–0.80} "achieving mean_cos ≥ 0.70".
Mean cosine over all held-out pairs does not depend on τ, so we read the rule
as the mean over pairs with cos ≥ τ (`mean_cos_at_tau`). That figure is ≥ τ by
construction, so we also report the raw mean over all held-out pairs
(`mean_cos_all`) and the share of pairs that clear τ (`coverage_at_tau`).
Final run: mean_cos_all = 0.151 (random baseline 0.054); at τ = 0.60,
mean_cos = 0.838 over 8 of 66 pairs (12.1%). Training pairs score 0.905, so the
model fits what it saw but generalizes weakly to held-out pairs, which by
construction cannot be deduced from training.

### Hyperparameters tuned on the validation split
Epochs (5–2000) and embed_dim (8, 16, 32, 50) were chosen by comparing
`mean_cos_all` against `baseline_mean_cos` on the validation split, which is
also the split mean_cos is reported on. With no separate test split, the
reported mean_cos may be slightly optimistic. Chosen: 2000 epochs, embed_dim 32,
learning rate 0.01, seed 42.

## Candidate generation (step 3)

### Vocabulary limited to the module and BFO-core
Candidates may only use the 364 classes of the Event module and BFO-core (the
classes mOWL was trained on, so every candidate can be scored in step 6) and
the 12 object properties the module already uses in its restrictions. Claims
about classes from other CCO modules (e.g. Agent) are left out. Labels for the
four CCO properties declared in ExtendedRelationOntology are stored in
`config.yaml`, taken from the CCO develop branch; the module file is not edited.

### Model, no temperature, and a response cache
Candidates are generated with Claude Opus 5.5 at high effort, because the task
requires ontological judgment (every-instance claims, property direction, the
"designed" hedge). The model does not accept a temperature setting, so
reproducibility comes from caching every response, keyed by a hash of the model
and the full prompt; reruns reuse the cache and give identical results. Full
run: 328 classes, 133 raw candidates, 125 kept, cost ≈ $1.65.

## Hybrid filtering (step 6)

### Scoring existential candidates
For `A ⊑ ∃R.B`, step 6 uses cos(A + R, B), where R is the learned property
vector, rather than cos(A, B). ELEmbeddings trains existentials as "A
translated by R falls inside B", so this is the similarity the model actually
learned; plain cos(A, B) ignores R and compares a process with its participant
as if one were a subclass of the other. Subclass candidates (`A ⊑ B`) use
cos(A, B) as in step 5. Decided before scoring any candidates. For comparison,
40 of 118 candidates reach cos ≥ 0.43 with this score versus 36 with plain
cos(A, B), so the choice did not materially change the outcome.

### LLM plausibility floor
A candidate is accepted only if combined = 0.7·cos + 0.3·llm ≥ τ **and** the
LLM plausibility score is ≥ 0.5. The README specifies only the weighted
average; without a floor, a high cosine could outvote an LLM judgment that the
axiom is wrong. The floor was set before scoring. In the final run no candidate
scored below 0.5 (lowest 0.55), so it rejected nothing.

The plausibility judge is a separate prompt from the generator and is given
CCO's original definition rather than the LLM-enriched one, so the two LLM
steps rest on different evidence.

### Hard checks before scoring
Candidates are rejected before any LLM call if they are already entailed by
the module + BFO (inferred with ELK), would create a subclass cycle, violate
the property's domain or range, or lack an embedding. Known approximations,
all of which err toward letting a candidate through (step 7's ELK run is the
backstop):
- Entailment of `A ⊑ ∃R.B` is checked against asserted existentials on A or
  its inferred ancestors, with B or a subclass of B as filler; property
  hierarchies and existentials inside equivalence definitions are ignored.
- The domain of `is object of` (ont00001936) is simplified from
  SDC ∪ GDC ∪ (independent continuant ⊓ ¬spatial region) to
  SDC ∪ GDC ∪ independent continuant.
- Union domains and ranges are expanded to their named members.

Four candidates of the form `X ⊑ has participant some continuant` were rejected
as range violations: `continuant` is broader than the property's range (it
includes spatial regions), and the axioms add almost nothing beyond "has a
participant".

Final run: 125 candidates; 3 already entailed, 4 range violations, 85 below τ,
33 accepted (LLM contribution rate 15%: 5 of 33 accepted axioms have cos < τ).

## Merge and reason (step 7)

### event-augmented.ttl holds asserted axioms only
The README calls the final ontology `src/module_augmented.ttl`; here it is
named `src/event-augmented.ttl` to match the Event module it extends (the path
is set in `config.yaml`). It is the original Event module (with its original
`owl:imports`) plus the 33 accepted axioms, and nothing else. ELK is run over
module + BFO-core + accepted axioms to check consistency and unsatisfiable
classes, but its inferred subclass axioms are not written into the file and BFO
is not copied in. This keeps the output comparable to the original module: the
only difference is the accepted axioms. The script also verifies that every
accepted axiom is present in the saved file. Final run: consistent, no
unsatisfiable classes.
