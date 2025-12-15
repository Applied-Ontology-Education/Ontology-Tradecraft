# Week 5 Project: Zero-Touch Axiom Induction for CCO using MOWL + LLMs

## Learning Objectives

In this project, you will use machine learning to discover missing logical axioms in the Common Core Ontologies (CCO).
While CCO’s textual definitions contain rich semantics, most logical structure is still implicit.
Your task is to use MOWL (Machine Learning over OWL) and LLM-based semantic modeling to make that structure explicit
with no manual intervention.

Each student will select exactly one from the
eleven [CCO modules](https://github.com/CommonCoreOntology/CommonCoreOntologies/tree/develop/src/cco-modules), train a
MOWL model on its existing axioms, generate_candidates new candidate OWL 2 EL axioms from textual definitions, use MOWL
to filter those candidates by learned semantic similarity, and then auto-merge the enriched ontology.

Students should coordinate so that no two students are using the same CCO module for this project.

By completing this project, you will:

- Understand how both symbolic and neural models can support ontology engineering.
- Learn how to use MOWL to train embeddings over OWL 2 EL ontologies.
- Use LLMs to transform textual definitions into candidate logical axioms.
- Combine MOWL semantic similarity and LLM plausibility scoring for hybrid axiom filtering.
- Validate ontology quality and consistency using ROBOT, ELK, and OOPS!.
- Build a complete end-to-end ontology enrichment pipeline integrating symbolic and language-based AI.

## Project Overview

Each student will:

- Choose a unique CCO module (e.g., AgentOntology.ttl, ArtifactOntology.ttl, etc.).
- Extract textual definitions (skos:definition) into a CSV.
- Use an LLM to rewrite, normalize, and generate_candidates OWL
  2 [EL-compliant candidate axioms](https://www.w3.org/TR/owl2-profiles/#OWL_2_EL) from those (see “What are OWL 2
  EL–compliant candidate axioms?” below).
- Train a MOWL model on the existing axioms in your module to learn a semantic embedding space.
- Filter candidates using both MOWL cosine similarity and LLM-based plausibility scores, keeping axioms above the
  learned threshold.
- Automatically merge accepted axioms back into the module.
- Automatically reason with ELK via ROBOT to check for logical consistency.

## Project Details

1. Using RDFlib, write textual-definition extraction script, `scripts/extract_definitions.py`, that inputs
   `src/module.ttl` and outputs `data/definitions.csv` with columns: IRI, label, definition.
2. Write `scripts/preprocess_definitions_llm.py` to normalize and enrich definitions with an LLM and output
   `data/definitions_enriched.csv`; this should ensure definitions are standardized in a canonical form (e.g., “X is a Y
   that Zs”) and that ambiguity is removed, abbreviations expanded, and terminology aligned to CCO style.
3. Create `scripts/generate_candidates_llm.py` that takes enriched definitions and returns OWL 2 EL–compliant candidate
   axioms in `generated/candidate_el.ttl`.
4. Split `src/cco-module.ttl` into `src/train.ttl` (≈ 80%) and `src/valid.ttl` (≈ 20%), preserving all class and
   property declarations.

- Use:
  `python3 projects/project-5/scripts/split_train_valid.py --input src/InformationEntityOntology.ttl --train src/train.ttl --valid src/valid.ttl --ratio 0.8 --seed 42`.
- The splitter keeps all owl:Class/ObjectProperty/DataProperty declarations in BOTH splits and partitions only simple
  named-class rdfs:subClassOf axioms; this matches MOWL PathDataset expectations for training/evaluation.

5. Write `scripts/train_mowl.py` to train an ELEmbeddings model on `src/train.ttl`, evaluate on `src/valid.ttl`, and
   record results in `reports/mowl_metrics.json`; compute cosine similarity for held-out subclass–superclass pairs;
   choose the smallest threshold τ ∈ {0.60 – 0.80} achieving mean_cos ≥ 0.70.
6. Implement `scripts/filter.py` to score each (sub, super) pair by MOWL cosine similarity, query an
   LLM to rate semantic plausibility (0–1), then combine scores as a weighted average (e.g., 0.7 × cosine + 0.3 × LLM),
   returning a list of axioms meeting the threshold in `generated/accepted_el.ttl`.
7. Merge `generated/accepted_el.ttl` with `src/cco-module.ttl` using ROBOT, reason with ELK, and save the final ontology
   as `src/module_augmented.ttl`.
8. Write a single driver script `scripts/run_all.py` that executes the full pipeline:

- `extract_definitions.py` → `data/definitions.csv`
- `preprocess_definitions_llm.py` → `data/definitions_enriched.csv`
- `generate_candidates_llm.py` → `generated/candidate_el.ttl`
- Split axioms → `src/train.ttl`, `src/valid.ttl`
- `train_mowl.py` → `reports/mowl_metrics.json`
- `filter.py` → `generated/accepted_el.ttl`
- Merge + reason → `src/module_augmented.ttl`
  The script should print a concise report summarizing: mean_cos, chosen τ, number of accepted axioms, LLM
  contribution rate, and consistency status.

9. Submit the repository with all outputs so the main course repository.

## Ollama GPU selection (choose GPU 1)

Ollama does not take a direct "--gpu" flag. Instead, you select the GPU(s) by
setting environment variables before starting the Ollama server (ollama serve).
This repo provides a helper script that pins Ollama to a specific device.

- To start Ollama on GPU 1 (the second GPU):

  - Foreground:
    - `bash projects/project-5/scripts/start_ollama_gpu.sh --device 1`

  - Background (logs written to projects/project-5/logs/ollama_serve.log):
    - `bash projects/project-5/scripts/start_ollama_gpu.sh 1 --background`

The script sets both CUDA_VISIBLE_DEVICES (NVIDIA) and HIP_VISIBLE_DEVICES (AMD).
You can also preconfigure these in projects/project-5/.env:

- `OLLAMA_CUDA_VISIBLE_DEVICES=1`
- `OLLAMA_HIP_VISIBLE_DEVICES=1`

Then simply run:

- `bash projects/project-5/scripts/start_ollama_gpu.sh`

Your Python scripts (e.g., preprocess_definitions_llm.py) will use whatever
GPU the running Ollama server is bound to. Make sure you have pulled your model
and that `ollama serve` is running before executing the enrichment steps.

## Grading Criteria

1. mean_cos ≥ 0.70, ontology consistent, and at least several new axioms added automatically.
2. The entire workflow runs from one command without manual edits.

## Files in the Repository

- `data/definitions.csv` — Extracted SKOS definitions (columns: IRI, label, definition).
- `data/definitions_enriched.csv` — LLM-normalized/enriched definitions used for axiom generation.
- `generated/candidate_el.ttl` — OWL 2 EL candidate axioms produced via LLM + pattern prompts.
- `generated/accepted_el.ttl` — Candidates retained after hybrid MOWL cosine + LLM plausibility filtering.
- `reports/mowl_metrics.json` — Training/evaluation metrics (mean cosine, selected τ, and hybrid settings).
- `src/cco-module.ttl` — Original selected CCO module (input ontology).
- `src/train.ttl` — Training split of the ontology (≈ 80% of axioms).
- `src/valid.ttl` — Validation split of the ontology (≈ 20% of axioms).
- `src/module_augmented.ttl` — Final ontology after merging accepted axioms and reasoning with ELK.
- `scripts/extract_definitions.py` — Extracts SKOS definitions into `data/definitions.csv`.
- `scripts/preprocess_definitions_llm.py` — Normalizes/enriches definitions with an LLM.
- `scripts/generate_candidates_llm.py` — Generates EL-compliant candidate axioms from enriched definitions.
- `scripts/train_mowl.py` — Trains ELEmbeddings on `src/train.ttl` and evaluates on `src/valid.ttl`.
- `scripts/split_train_valid.py` — Deterministically splits an ontology into train/valid for MOWL PathDataset.
- `scripts/filter.py` — Combines MOWL cosine and LLM plausibility to keep axioms.
- `scripts/run_all.py` — One-command driver that executes the entire pipeline end-to-end.

## How the embeddings are created (vector reference retrieval)

When you enable vector-based reference retrieval (REFERENCE_MODE=vector), the project builds and queries a small local
vector database (Milvus Lite) of BFO/CCO reference terms. This helps select the most relevant glossary entries to inject
into the LLM prompt.

Overview

- Builder script: scripts/build_vector_db.py
- Query path: scripts/preprocessing/enriching.py (build_reference_context when reference_mode=vector)
- Libraries: sentence-transformers for embeddings, Milvus Lite (pymilvus) for vector search

Creation (indexing) steps

1) Source data: projects/project-5/data/bfo_cco_terms.csv with columns: label, definition, type (class|property|…)
2) Text preparation per entry: the input text is concatenated as "{label} {definition}".
3) Embeddings model: SentenceTransformer with model name from EMBEDDING_MODEL (default:
   sentence-transformers/all-MiniLM-L6-v2).
4) Normalization: model.encode(..., normalize_embeddings=True) so vectors are length-normalized.
5) Dimensionality: taken from the model (fallback 384 if not reported).
6) Storage: two Milvus Lite collections are created with an AUTOINDEX and COSINE metric:

- VECTOR_COLLECTION_CLASSES (default: ref_classes) for entries with type == "class"
- VECTOR_COLLECTION_PROPERTIES (default: ref_properties) for entries with type == "property"
  Each collection stores fields: label (VARCHAR), definition (VARCHAR), vector (FLOAT_VECTOR[dim]).

7) Insert and load: All vectors are inserted and the collections are loaded for querying.

Querying during enrichment

- Per row, we form a query string: "{label} {definition}" (of the current ontology element).
- We embed that query with the same model and normalize_embeddings=True.
- We choose the collection based on the element type: class → classes collection; property → properties collection;
  unknown → both.
- We run a vector search in Milvus with COSINE metric and limit = REF_TOP_K.
- Results are returned as "- {label}: {definition}" lines to become the reference_context injected into the prompt.
- If anything fails (missing deps/DB), the code logs a warning and falls back to the simple lexical retrieval path, so
  the pipeline keeps working.

Configuration

- VECTOR_DB_URI (default: data/milvus.db)
- VECTOR_COLLECTION_CLASSES (default: ref_classes)
- VECTOR_COLLECTION_PROPERTIES (default: ref_properties)
- EMBEDDING_MODEL (default: sentence-transformers/all-MiniLM-L6-v2)
- REF_TOP_K controls the number of entries retrieved per row.

Build and use

1) Build (run once, or after bfo_cco_terms.csv changes):

- python scripts/build_vector_db.py
- Optionally set env vars above to customize location, collections, or model.

2) Enable vector retrieval for enrichment:

- REFERENCE_MODE=vector (e.g., set in .env)

3) Run the enrichment as usual:

- python scripts/preprocess_definitions_llm.py

Notes

- The embeddings only affect how reference context is selected; they do not change the LLM itself.
- Collections are rebuilt by the builder script to keep the index consistent with the source CSV; adapt the script if
  you want incremental updates.

## Prompt configuration (PROMPT_CONFIG_FILE)

The enrichment script supports configurable prompts for classes and properties via a readable Markdown file (
recommended) or a legacy INI file (backward compatible).

- Default locations committed in this repo:
  - Markdown: `projects/project-5/prompts/default_prompts.md` (recommended)
  - INI: `projects/project-5/prompts/default_prompts.ini`

- Environment variable to point to a custom file (relative to project-5 or absolute):
  - `PROMPT_CONFIG_FILE=prompts/default_prompts.md`
  - This is already set in `projects/project-5/.env`.

- Auto-discovery (when `PROMPT_CONFIG_FILE` is not set), in order:
  1. `projects/project-5/prompts/prompt.md`
  2. `projects/project-5/prompts/prompt.ini`
  3. `projects/project-5/prompts/default_prompts.md`
  4. `projects/project-5/prompts/default_prompts.ini`
     If none exists, the script falls back to built-in defaults.

### Markdown structure (recommended)

Available variables inside templates: `{label}`, `{definition}`, `{reference_context}`.

Use the following headings (case-insensitive):

```
## Class

### System
You are a precise ontology editor and an expert in Basic Formal Ontology (BFO) and Common Core Ontologies (CCO). Improve CLASS definitions in a clear, concise, academic style that adheres to BFO principles. Use genus–differentia with the pattern: '"x" is a Y that Zs'.

### User
Below is the label for the ontology element X; improve its definition as explained:
Label: {label}
Definition: {definition}
Reference BFO/CCO glossary entries are:
 {reference_context} 
Rules: single sentence; academic tone; expand abbreviations; return only the improved definition without quotes or commentary.

## Property

### System
You are a precise ontology editor and an expert in Basic Formal Ontology (BFO) and Common Core Ontologies (CCO). Improve PROPERTY definitions in a clear, concise, academic style that adheres to BFO principles. Use the pattern: 'a X b' iff a is a Y that does Z.

### User
Below is the label for the ontology element X; improve its definition as explained:
Label: {label}
Definition: {definition}
Reference BFO/CCO glossary entries are:
 {reference_context} 
Rules: single sentence; academic tone; expand abbreviations; return only the improved definition without quotes or commentary.
```

Notes:

- Only the headings matter for parsing: `## Class` / `## Property` with `### System` and `### User` subsections under
  each. Content continues until the next `###` or `##` heading.
- Missing sections fall back to built-in defaults.

### INI structure (legacy, still supported)

```
[class]
system = ... class-specific instructions ...
user = Label: {label}\nDefinition: {definition}\nReference BFO/CCO glossary entries are:\n {reference_context}

[property]
system = ... property-specific instructions ...
user = Label: {label}\nDefinition: {definition}\nReference BFO/CCO glossary entries are:\n {reference_context}
```

To customize, copy `prompts/default_prompts.md` to `prompts/prompt.md` and edit. Alternatively, set `PROMPT_CONFIG_FILE`
to your own file path (Markdown or INI).
