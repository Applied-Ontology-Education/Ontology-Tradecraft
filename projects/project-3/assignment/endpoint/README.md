# Structure Matcher — Help Guide

Use this guide when running `compare_structures.py`  
It explains what each flag does, when to use it, and includes copy-pasteable recipes

## Required Paths

### `--left PATH` / `--right PATH`
Ontology files to compare (`.ttl`, `.owl`, RDF/XML, etc.)

### `--outdir PATH`
Where the output XLSX goes  
Default: current directory
Filename: `<left-stem>-<right-stem>-structural-matches.xlsx`

## Shape Granularity

### `--shape {exact|kind|coarse}`
Controls how fine-grained a class’s structure is encoded before matching

- **exact** (strictest)  
  Keeps **numbers** and **qualifiedness** (`q*` vs unqualified)
  - Examples:  
    - `R:qmin=1`, `R:min=2`, `R:card=1`, `R:only`, `R:some`, `R:has`  
  - Use when you want high-precision, “equivalence-grade” candidates

- **kind** (moderate)  
  Drops **numbers** but keeps qualifiedness  
  - Examples:  
    - `R:qmin`, `R:min`, `R:qmax`, `R:max`, `R:qcard`, `R:card`, `R:only`, `R:some`, `R:has`  
  - Use to find “same kind of constraint” without insisting on numeric equality

- **coarse** (loosest)  
  Drops **numbers** *and* collapses qualifiedness:  
  `qmin→min`, `qmax→max`, `qcard→card`.  
  - Examples:  
    - `R:min`, `R:max`, `R:card`, `R:only`, `R:some`, `R:has`  
  - Use for broad triage to see if any structural overlap exists at all

## Normalization

### `--normalize {off|entailment|families}`
Post-processes shapes so near-equivalent encodings can line up

- **off**  
  - No normalization; raw tokens must match per `--shape`.  
  - Use for final, precise matching

- **entailment** (tiny logical closure)  
  - Adds small, safe implications so different encodings match:  
    - `card n` ⇒ `min n` **and** `max n` (and `some` if `n≥1`)  
    - `qcard n` ⇒ `qmin n` **and** `qmax n` (and `some` if `n≥1`)  
    - `min≥1`, `qmin≥1`, `has` ⇒ add `some`  
  - Good middle ground: aligns `min 1` vs `some`, `card 1` vs `min 1 & max 1`

- **families** (very coarse)  
  - Collapses tokens into broad families:  
    - **E (Existential-ish):** `some`, `has`, `min≥1`, `card≥1`, …  
    - **U (Universal):** `only`  
    - **MIN**, **MAX**, **EXACT**  
  - Output shows `normalized_key` like `(('F:E', 1),)`
  - Use only for initial discovery — expect many semantically loose matches

## Token Multiplicity

### `--presence-only`
- Treat shapes as a **set** (ignore how many times a token appears)  
- Without it (default), shapes are **multisets** (token counts matter)  
- Use `--presence-only` for discovery; drop it for stricter, final matches

## Quick Recipes

### 1. Broad Match (find any overlap)
```
--shape coarse --normalize families --presence-only \
--follow-imports --imports-depth 2
```
### 2. Moderate Match triage (find some overlap)
```
--shape kind --normalize entailment --presence-only \
--follow-imports --imports-depth 2
```
### 3. Strict Match (find exact overlap)
```
--shape exact --normalize off \
--follow-imports --imports-depth 2
```

# Output File Guide

- R:some — an existential restriction (… some …)
- R:only — a universal restriction (… only …)
- R:has — a hasValue restriction (… value v)
- R:min=2 / R:max=1 / R:card=1 — (unqualified) cardinalities
- R:qmin=1 / R:qcard=1 — qualified cardinalities (owl:onClass)
- ×k suffix in output (e.g., R:card×2) = token appeared k times