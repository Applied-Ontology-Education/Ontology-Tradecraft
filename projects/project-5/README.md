# Week 5 Project: Zero-Touch Axiom Induction for CCO using MOWL + LLMs

## Learning Objectives
In this project, you will use machine learning to discover missing logical axioms in the Common Core Ontologies (CCO).
While CCO’s textual definitions contain rich semantics, most logical structure is still implicit.
Your task is to use MOWL (Machine Learning over OWL) and LLM-based semantic modeling to make that structure explicit with no manual intervention.

Each student will select exactly one from the eleven [CCO modules](https://github.com/CommonCoreOntology/CommonCoreOntologies/tree/develop/src/cco-modules), train a MOWL model on its existing axioms, generate new candidate OWL 2 EL axioms from textual definitions, use MOWL to filter those candidates by learned semantic similarity, and then auto-merge the enriched ontology.

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
  - Use an LLM to rewrite, normalize, and generate OWL 2 [EL-compliant candidate axioms](https://www.w3.org/TR/owl2-profiles/#OWL_2_EL) from those.
  - Train a MOWL model on the existing axioms in your module to learn a semantic embedding space.
  - Filter candidates using both MOWL cosine similarity and LLM-based plausibility scores, keeping axioms above the learned threshold.
  - Automatically merge accepted axioms back into the module.
  - Automatically reason with ELK via ROBOT to check for logical consistency.

## Project Details
  1. Using RDFlib, write textual-definition extraction script, `scripts/extract_definitions.py`, that inputs `src/module.ttl` and outputs `data/definitions.csv` with columns: IRI, label, definition.
  2. Write `scripts/preprocess_definitions_llm.py` to normalize and enrich definitions with an LLM and output `data/definitions_enriched.csv`; this should ensure definitions are standardized in a canonical form (e.g., “X is a Y that Zs”) and that ambiguity is removed, abbreviations expanded, and terminology aligned to CCO style.
  3. Create `scripts/generate_candidates_llm.py` that takes enriched definitions and returns OWL 2 EL–compliant candidate axioms in `generated/candidate_el.ttl`.
  4. Split `src/cco-module.ttl` into `src/train.ttl` (≈ 80%) and `src/valid.ttl` (≈ 20%), preserving all class and property declarations.
  5. Write `scripts/train_mowl.py` to train an ELEmbeddings model on `src/train.ttl`, evaluate on `src/valid.ttl`, and record results in `reports/mowl_metrics.json`; compute cosine similarity for held-out subclass–superclass pairs; choose the smallest threshold τ ∈ {0.60 – 0.80} achieving mean_cos ≥ 0.70.
  6. Implement `scripts/filter_candidates_hybrid.py` to score each (sub, super) pair by MOWL cosine similarity, query an LLM to rate semantic plausibility (0–1), then combine scores as a weighted average (e.g., 0.7 × cosine + 0.3 × LLM), returning a list of axioms meeting the threshold in `generated/accepted_el.ttl`.
  7. Merge `generated/accepted_el.ttl` with `src/cco-module.ttl` using ROBOT, reason with ELK, and save the final ontology as `src/module_augmented.ttl`.
  8. Write a single driver script `scripts/run_all.py` that executes the full pipeline:
     - `extract_definitions.py` → `data/definitions.csv`
     - `preprocess_definitions_llm.py` → `data/definitions_enriched.csv`
     - `generate_candidates_llm.py` → `generated/candidate_el.ttl`
     - Split axioms → `src/train.ttl`, `src/valid.ttl`
     - `train_mowl.py` → `reports/mowl_metrics.json`
     - `filter_candidates_hybrid.py` → `generated/accepted_el.ttl`
     - Merge + reason → `src/module_augmented.ttl`
    The script should print a concise report summarizing: mean_cos, chosen τ, number of accepted axioms, LLM contribution rate, and consistency status.
  9. Submit the repository with all outputs so the main course repository. 

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
- `scripts/filter_candidates_hybrid.py` — Combines MOWL cosine and LLM plausibility to keep axioms.
- `scripts/run_all.py` — One-command driver that executes the entire pipeline end-to-end.


