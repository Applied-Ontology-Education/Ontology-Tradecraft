# Decisions

## General

### Input module keeps its original name
The input module is kept as `src/EventOntology.ttl` rather than renamed to
`src/cco-module.ttl` as the README's file list suggests. All scripts read it
from that path via `config.yaml`, and nothing writes to it.

### Output file name
The README calls the final ontology `src/module_augmented.ttl`. Here it's
`src/event-augmented.ttl`, after the Event module it extends. The path is set
in `config.yaml`.

## Definition rewriting (step 2)

### Reusing earlier rewrites
Step 2 keeps the existing rewrite in `definitions_enriched.csv` for any class
whose original definition hasn't changed, and only asks the LLM about new or
changed definitions. Without this, every run of `run_all.py` would word the
definitions slightly differently. That would change every step-3 prompt and,
through them, the candidates, scores and accepted axioms.

## Candidate generation (step 3)

### Vocabulary limited to the module and BFO-core
Candidate axioms may only use the 364 classes of the Event module and
BFO-core, and the 12 object properties the module already uses. These are
exactly the classes mOWL was trained on, so step 6 can score every candidate.
Claims about classes from other CCO modules (e.g. Agent) are left out. The
four CCO properties' labels come from CCO's develop branch and are stored in
`config.yaml`, so the module file is never edited.

### Candidate shapes, cap, and the "designed" hedge
Candidates take only the two OWL 2 EL forms the assignment asks for:
`A ⊑ B` and `A ⊑ ∃R.B`, at most 3 per class (set in `config.yaml`). The prompt
tells the model to:
- propose only what the definition actually says, and nothing if it adds
  nothing new
- propose only claims true of every instance of the class
- never turn "designed to X" into a `realizes`, `bearer of` or participation
  axiom, because CCO says "designed" so that broken or unused artifacts still
  count

### Model, no temperature, and a response cache
Candidates are generated with Claude Opus 5.5 at high effort, because the task
needs ontological judgment: every-instance claims, property direction, and the
"designed" hedge. This model doesn't accept a temperature setting, so
reproducibility comes from saving every answer. A rerun with the same model
and the same prompt reuses the saved answer instead of asking again, so
results are identical and cost nothing. Full run: 328 classes, 133 raw
candidates, 125 kept after removing ones already in the module, cost about
$1.65.

## Split (step 4)

### Training data includes BFO-core
`src/train.ttl` = the module (minus held-out pairs and `owl:imports`) + a
local copy of BFO-core 2020 (`src/bfo-core.ttl`). BFO gives the embeddings
upper-level structure (hierarchy, disjointness, property domains). It is
merged from a local file rather than loaded as an import, so runs work offline
and are reproducible. The InformationEntityOntology import is left out so the
embeddings stay about this module. Held-out pairs come only from the module's
own axioms, never from BFO.

About half of BFO-core's axioms use features outside OWL 2 EL (such as `only`
restrictions, unions and inverse properties). mOWL ignores those, so only
BFO's EL axioms affect training.

### Holdout size: 20% of all named subclass pairs
I hold out 66 pairs, which is 20% of all 328 named `A ⊑ B` pairs. The
alternative was 20% of only the pairs that could safely be held out, which
would have been about 23. I chose 66 because it matches the README's "≈20%"
and gives a steadier `mean_cos`: averaging over 66 pairs is about 1.7× less
noisy than averaging over 23.

### Limitation: mostly internal classes are tested
A held-out class must still appear somewhere in training, so a leaf class with
no other axioms can never be held out (about 200 pairs are like this). The
test set is therefore mostly classes that have subclasses or existential
restrictions of their own, and `mean_cos` says little about leaf classes.

### Declaring four CCO object properties
Four CCO object properties used in the module (`ont00001777`, `ont00001803`,
`ont00001819`, `ont00001936`) are declared in other CCO modules that I don't
load. The split adds a declaration for each to `train.ttl` so the axioms that
use them load correctly. No new IRIs are created, and a test confirmed mOWL
trains on all four.

## Evaluation (step 5)

### Mean cosine, not ranking metrics
The README asks for mean cosine similarity over held-out subclass pairs, with
a 0.70 target. Research on ontology embeddings usually uses ranking measures
instead (e.g. "is the true superclass in the model's top 10?"). I report mean
cosine because the assignment requires it, with two known limitations:
- ELEmbeddings represents each class as a ball (a center and a radius). Cosine
  only compares the directions of the centers, so it ignores the radius and
  the "ball inside ball" shape the model is trained on.
- For `A ⊑ ∃R.B`, plain cos(A, B) ignores the property R. Step 6 handles this
  (see "Scoring existential candidates").

I also report `baseline_mean_cos`, the same score for random unrelated pairs.
If real pairs and random pairs scored equally high, the model would have
pushed all classes in the same direction and the score would mean nothing.

### Reporting mean_cos two ways
The README asks for "the smallest τ ∈ {0.60–0.80} achieving mean_cos ≥ 0.70."
Averaged over all held-out pairs, mean_cos doesn't change with τ, so choosing
a τ would be meaningless. I therefore read the rule as the average over pairs
scoring at or above τ (`mean_cos_at_tau`).

That average can never fall below τ, because every pair in it scored at least
τ. So I report two more numbers next to it:
- `mean_cos_all`: the plain average over all held-out pairs
- `coverage_at_tau`: the share of held-out pairs that reach τ

Final run:

| | Value |
|---|---|
| mean_cos_all | 0.151 (random pairs: 0.054) |
| mean_cos_at_tau, τ = 0.60 | 0.838 |
| coverage_at_tau | 8 of 66 pairs (12.1%) |

On the pairs it was trained on, the model scores 0.905. So it learns what it's
shown but has trouble with unseen pairs, which by design can't be worked out
from the training data.

### Hyperparameters tuned on the validation split
I chose the number of training passes (5 to 2000) and the embedding size (8,
16, 32, 50) by comparing `mean_cos_all` with the random baseline on the
validation pairs, the same pairs `mean_cos` is reported on. With no separate
test set, the reported `mean_cos` may be slightly optimistic. Final settings:
2000 epochs, embedding size 32, learning rate 0.01, seed 42.

## Hybrid filtering (step 6)

### Scoring existential candidates
For `A ⊑ B`, the cosine score is cos(A, B), as in step 5.

For `A ⊑ ∃R.B`, the score is cos(A + R, B), where R is the vector the model
learned for the property. ELEmbeddings learns existentials as "A, shifted by
R, lands inside B," so this measures what the model actually learned. Plain
cos(A, B) would ignore R and treat, for example, a process and its participant
as if one were a kind of the other.

I chose this before scoring any candidates. It turned out to make little
difference: 40 of 118 candidates reach the 0.43 needed to pass, compared with
36 under plain cos(A, B). (0.43 is the lowest cosine that can still pass:
0.7 × 0.43 + 0.3 × 1.0 ≈ 0.60 = τ.)

### LLM plausibility floor
A candidate is accepted only if both of these hold:
- combined score = 0.7 × cosine + 0.3 × LLM score ≥ τ (0.60)
- LLM score ≥ 0.5

The README only specifies the weighted score. I added the floor so that a high
cosine can't outvote an LLM judgment that the axiom is wrong. It was set
before scoring. In the final run, no candidate scored below 0.5 (the lowest
was 0.55), so the floor rejected nothing.

The LLM judge uses a separate prompt from the generator and sees CCO's
original definition, not the rewritten one, so the two LLM steps rely on
different evidence.

### Hard checks before scoring
Before any LLM call, a candidate is rejected if it:
- already follows from the module + BFO (checked with ELK)
- would make two classes equivalent (a subclass cycle)
- breaks the property's domain or range
- involves a class or property with no trained vector

These checks simplify in three places. Each simplification lets a candidate
*through* rather than wrongly rejecting it, and step 7's ELK run catches any
real conflict:
- For `A ⊑ ∃R.B`, "already follows" only checks whether A, or a class above A,
  already says `R some B` (or `R some` a subclass of B). It ignores
  sub-properties, and restrictions inside equivalence definitions.
- The domain of `is object of` (ont00001936) is simplified from "specifically
  dependent continuant, generically dependent continuant, or independent
  continuant that isn't a spatial region" to the same list without the
  spatial-region exception.
- When a domain or range is a union of classes, a class passes if it falls
  under any one of them.

Four candidates of the form `X ⊑ has participant some continuant` were
rejected because `continuant` is wider than the property allows (it includes
spatial regions). They said little more than "X has a participant" anyway.

Final run: 125 candidates. 3 already followed from the ontology, 4 broke a
range, 85 scored below τ, and **33 were accepted**. 5 of the 33 had a cosine
below τ and passed because of the LLM score, an LLM contribution rate of 15%.

## Merge and reason (step 7)

### The output contains only the original module plus the accepted axioms
`event-augmented.ttl` is the original Event module, with its original imports,
plus the 33 accepted axioms, and nothing else.

ELK checks module + BFO-core + accepted axioms for consistency and
unsatisfiable classes, but what ELK infers is **not** written into the file,
and BFO isn't copied in. That keeps the output directly comparable to the
original: the only difference is the 33 new axioms. The script also confirms
that all 33 are actually in the saved file.

Final run: consistent, with no unsatisfiable classes.
