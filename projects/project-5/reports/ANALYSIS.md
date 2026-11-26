# Project 5 — FacilityOntology: results and analysis

Run: `python scripts/run_all.py` (one command; LLM replies cached in `data/llm_cache.json`, so re-runs are identical and need no API key).

| Metric | Value |
|---|---|
| mean_cos (held-out entailed pairs, M1, 3 seeds) | **0.755** (0.755 / 0.769 / 0.741) |
| valid.ttl direct held-out pairs | -0.016 |
| random-pair baseline | 0.081 |
| tau (smallest in 0.60–0.80 with retained mean >= 0.70) | 0.60 (instructor 60th-pct rule: 0.75) |
| candidates (LLM) | 72: 5 subclass (A), 67 bearer-of-function (B) |
| accepted (0.7 cos + 0.3 LLM >= tau, ELK-checked) | **2** |
| LLM contribution rate | 0% |
| ELK (ROBOT) | consistent, no unsatisfiable classes; candidates and accepted axioms are OWL 2 EL |

Accepted: `Fossil Fuel Power Plant ⊑ Thermal Power Plant`, `Petrochemical Refinery ⊑ Petroleum Manufacturing Facility`.

## Why only two axioms (finding, not a tuning target)
1. **Function axioms (B) get no support from MOWL.** Mean cosine 0.119 vs mean LLM plausibility 0.702. ELEmbeddings learns only from axioms in training; across all 11 CCO modules there are about 6 `bearer of some …` axioms, so the embedding has no signal linking facilities to Artifact Functions.
2. **Facility is a strict tree.** A second, subclass-only LLM pass (91 queries) found no further multi-parent links; the hybrid filter correctly rejected the weak ones (e.g. Fire Station ⊑ Storage Facility).
3. **Weight sensitivity (diagnostic, not applied):** candidates >= tau at cos/LLM weights 0.7/0.3 → 2, 0.6/0.4 → 2, 0.5/0.5 → 2, 0.4/0.6 → 2, 0.3/0.7 → 21, 0.0/1.0 → 64. The result is stable until the LLM dominates; the README weighting was kept.

## Method notes and judgment calls
- **Split:** instructor's connectivity-preserving rule (`documentation/embedding_example.py`), 73/18 SubClassOf edges; declarations kept in both files.
- **mean_cos uses entailed held-out pairs** (implied by train.ttl, never asserted), the test design in the ML-embeddings lecture. Direct valid.ttl pairs score ~0 because removing an edge from a tree cuts the subtree off from its parent, leaving no training signal; both are reported in `reports/mowl_metrics.json`.
- **Two embeddings:** M1 = train.ttl (README step 5; mean_cos, tau, pattern A). M2 = train.ttl + direct imports (ArtifactOntology, BFO), used only for pattern B because the function classes live there. After evaluation both are refit on the full module (M1F, M2F) to score candidates.
- **LLM:** Claude (`claude-sonnet-5-5`); only existing IRIs; each candidate cites its definition evidence; non-new, cyclic or unknown-label candidates are discarded (`reports/candidates_discarded.csv`). The installed SDK rejected `temperature`, so the cache, not temperature 0, guarantees reproducibility.
- **Imports** are pinned local copies (CCO develop @ 662f3a6) resolved via `src/catalog-v001.xml`; `src/module.ttl` is an identical copy of `src/cco-module.ttl` because README step 1 names that path.
- **requirements.txt:** upstream pins (`mowl==1.0.2`, `JPype1==1.6.0`) do not resolve; MOWL is on PyPI as `mowl-borg` (needs JPype1 1.5.x); `anthropic` added.
- **OOPS!** step is optional and non-fatal; the service returned HTTP 503 at run time. ROBOT report: `reports/robot_report.tsv`.
- Per-candidate scores and reasons: `reports/hybrid_review.csv`.
