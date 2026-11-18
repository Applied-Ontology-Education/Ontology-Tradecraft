# Project 6 — Scoped Cosine for New Excel-Generated ABox (v1)

## What is different

This update adds a **new, independent** Stage `09a_manifest_cosine` after manifest enrichment and before final HermiT/SHACL validation. It compares:

- **Baseline:** `generated/module_augmented.ttl` — original ontology plus rail TBox, before Excel ingestion.
- **Final:** `generated/module_augmented_with_manifest_enriched.ttl` — baseline plus Excel-derived individuals, assertions, and additional property declarations.

**Only these new ABox assertions get cosine scores:**

1. Newly introduced `rdf:type` assertions for individuals.
2. New object property assertions linking individuals.
3. New datatype property assertions involving individuals.

The report counts new schema property declarations but **does not** score original subclasses, imported ontology relations, the eight rail class proposals, TBox property declarations, OWL `NamedIndividual` markers, or annotation triples. A newly created class does not earn a point merely for being newly created. The unit of scoring is **an ABox assertion**, so one individual may appear in several comparisons.

## Scoring and limitations

A CPU inference-only pretrained SentenceTransformer model embeds **two different texts per assertion**:

- An evidence/context text generated from the newly ingested instance, its label, endpoint types, other non-target predicates, or literal value **kind** (not a raw ID/date); and
- A target description for the class or property obtained from its ontology label/definition (local imports are used as lookup context).

The raw cosine between those two embeddings is reported for each new assertion. The report gives a **micro mean across scored new assertions** (used for the reference 0.70 target) and per-type means, as well as a macro mean so that repeated data properties cannot be mistaken for independent model accuracy. No scores are manually overridden. Source text features such as auto-generated names can leak meaning; exact class names are removed from individual labels before embedding, but the method remains a **semantic plausibility heuristic, not independent proof**. `rdfs:domain`, `rdfs:range`, SHACL, HermiT, and verified workbook evidence are separate correctness tests. The 0.70 threshold is a requested operational threshold, *not* a scientifically calibrated ontology correctness criterion.

Raw manifest literal values and person names are not written to JSON, nor sent to a hosted embedding API. Embedding inference runs on the local machine. The model download is a one-time first-run requirement, not a training step. This step needs **no Java/mOWL training**, no XAI API key and no Grok calls.

## Installation

1. From `C:\Users\cappa\Ontology-Tradecraft\projects\project-6`, back up your current orchestration:

   ```powershell
   Copy-Item .\scripts\run_all.py .\scripts\run_all_before_scoped_cosine.py
   ```

2. Unzip `Project6_Scoped_New_ABox_Cosine.zip` into the `project-6` directory. Its `scripts/` files will land in your existing `scripts/` folder; it does not replace your manifest generation, enrichment, reasoner, or original `.ttl` data files.

   ```powershell
   Expand-Archive "$env:USERPROFILE\Downloads\Project6_Scoped_New_ABox_Cosine.zip" -DestinationPath . -Force
   ```

3. Activate `.venv-mowl` if needed. First run the no-ML audit **without reinstalling anything**:

   ```powershell
   python -u scripts/evaluate_manifest_delta_cosine.py --project-root . --audit-only
   ```

   Check the generated `reports/manifest_delta_cosine.json`:

   ```powershell
   $r = Get-Content .\reports\manifest_delta_cosine.json -Raw | ConvertFrom-Json
   $r.selected_assertions
   $r.counts
   $r.original_subclass_axioms_scored
   ```

   The final line must show `0` and `total_selected_assertions` must be positive.

4. To obtain actual cosine scores with a local pretrained text model:

   ```powershell
   python -m pip install sentence-transformers
   python -u scripts/evaluate_manifest_delta_cosine.py --project-root . --threshold 0.70
   ```

   On first use, the fixed `sentence-transformers/all-MiniLM-L6-v2` model snapshot may be downloaded from Hugging Face. Model weights are not trained in this stage. The model revision is pinned in the script. If working entirely offline after caching the model, add `--local-files-only`.

5. Summarize the report:

   ```powershell
   $r = Get-Content .\reports\manifest_delta_cosine.json -Raw | ConvertFrom-Json
   $r.results.mean_cosine
   $r.results.metric_threshold_met
   $r.results.by_category
   $r.unresolved_property_usages.Count
   ```

6. Run offline tests (no model downloads or Grok):

   ```powershell
   python -m unittest discover -s scripts -p "test_evaluate_manifest_delta_cosine.py" -v
   ```

## Integration with `run_all.py`

A new stage `09a_manifest_cosine` automatically executes **after Step 9 manifest enrichment and before Step 10 HermiT/SHACL validation**. Its report is verified for baseline/final/source hashes, strict exclusion of old classes/subclasses, and expected score counts. One-click runs require the `sentence-transformers` Python package; preflight reports missing dependencies before archiving prior outputs.

- If the cosine mean >= 0.70: terminal status `PASS_FULL_TBOX_ABOX_VALIDATED_COSINE_MET` *only after* final reasoner/SHACL success.
- If mean < 0.70: terminal status `PASS_FULL_TBOX_ABOX_VALIDATED_COSINE_BELOW_TARGET` after reasoner/SHACL success. **This explicitly means the 0.70 requirement is unmet; do not present it as passing.**

**Important:** For compatibility with your current Stage 7 controlled-merge gates, this initial update **does not remove the older Step 5A/5B/5C mOWL+Grok stages** from `run_all.py`. They no longer contribute to the new Excel-ABox 0.70 mean, but a complete `run_all.py` rerun may still take significant time and require Grok/Java. To test *only the new metric*, use the standalone command above on your already-generated TTL files; it reuses your produced input graphs without rerunning training or modifying any ontology.

If you change the manifest/ontology, regenerate your pipeline outputs before scoring. **Do not use an old `module_augmented_with_manifest_enriched.ttl` to claim results for a new input workbook.**

## Outputs

- `reports/manifest_delta_cosine.json`: scoped counts, model provenance, per-assertion hashed IDs and cosine scores, micro/macro averages and 0.70 status. No raw manifest literal values in JSON.
- `reports/step8_run_summary.json`: existing one-click report, augmented with `cosine_scope`, `cosine_mean`, `cosine_threshold_met`, and `cosine_scored_assertions`.

## Offline verification performed during development

- 23 unit/integration tests passed (5 scoped evaluator tests + 18 updated pipeline tests).
- Tested `--audit-only` against an earlier supplied pre-enrichment manifest graph: it selected 25 individual-type assertions, 18 object property assertions, and 33 data property assertions; the exact counts on your current 709-triple enriched ontology will differ.
- **No real SentenceTransformer cosine score was run in this offline environment. The 0.70 criterion is not yet demonstrated.**
