# Project 6: Project 5 joint ELEmbeddings + cosine transfer experiment

This ZIP is the **next isolated Project 6 Step 5 experiment**. It adapts a concrete successful approach from the user's prior **Project 5** implementation (`train_mowl_joint.py`): **native ELEmbeddings loss + direct cosine alignment of known training subclass pairs + a deterministic label/definition TF-IDF semantic adapter + inner development model selection + a once-only final outer evaluation**. It also retains the original **Project 6 80/20 train/validation split** and the previous raw mOWL baseline. No LLM API calls or downloads occur.

**Important:** The earlier Project 5 experimental report indicated an outer validation mean around **0.922158** and outer control mean around **0.213041**, but it was a *different ontology and data split*. Project 6's success **cannot be assumed**. We cannot execute real mOWL training in the assistant's environment. The script is intended for your **working Windows `.venv-mowl`** environment and will report honest results.

## What is carried forward from Project 5?

1. Real **joint** native mOWL ELEmbeddings training: subclass ball-inclusion EL loss stays in use. This is different from the prior Project 6 independent PyTorch contrastive embedding table, which failed to generalize.
2. Direct auxiliary cosine regularization of known training positives (plus *training-only taxonomy transitive closure*, capped at 600). No outer held-out relationship enters the loss.
3. Simple reproducible, hash-based TF-IDF text features derived from existing class labels, definitions and local class names. A trainable semantic adapter affects normalized class-center cosine.
4. Sweep of eight fixed candidate configurations on a deterministic **internal development subset of the 80% outer training data**, then retrain one selected model on the full outer 80%. The outer 20% never influences configuration selection or optimizer updates.
5. Both **raw cosine mean** and **separation from unasserted controls** are reported. We add more difficult controls sharing the rail held-out child's IRI.
6. Strict provenance, no direct or transitive held-out edge leakage, local imports with SHA checks, deterministic seeds, no use of old checkpoints.

**Difference from Project 5:** Project 5 used HermiT-entailed training-only logical positives. For this first Project 6 transfer, the helper uses only the *named taxonomy transitive closure* generated from **training-only** axioms because the current Step 5A emits only named subclass edges and class declarations. The source/import GCAs and property restrictions are still absent from the native EL training dataset. If the result remains below 0.70, retaining additional supported OWL 2 EL logical structures is a separate, principled next step. This experiment **does not** reinstate full imported axioms, which could leak withheld relationships.

## Files installed

- `scripts/experiment_joint_rail_mowl.py` — isolated joint mOWL training and evaluation experiment
- `scripts/test_experiment_joint_rail_mowl.py` — offline regression tests
- `README_PROJECT6_PROJECT5_JOINT_MOWL.md`

It **does not replace** `scripts/prepare_rail_mowl.py`, `scripts/evaluate_rail_mowl.py`, `scripts/refine_rail_hybrid.py`, `scripts/run_all.py`, the source ontology, or any baseline reports/vectors. No existing Step 5 or final validation success is altered.

## Preconditions

In the **Project 6 root** after successfully running the **new 80/20 Step 5A and Step 5B**, you should already have:

- `src/module.ttl` and six original local `imports/` files
- `generated/rail_axioms_refined.ttl`
- `generated/mowl_rail_train.ttl`, `generated/mowl_rail_valid.ttl`
- `reports/step5_split_report.json`
- `reports/step5_cosine_scores.json` from the *same* 80/20 run
- mOWL, PyTorch, NumPy and RDFLib available in `.venv-mowl`

**Do not manually create or edit any generated TTL, CSV, JSON or embedding checkpoint.** The experiment consumes the files already produced by the automated pipeline.

## Install in Windows PowerShell

From `C:\Users\cappa\Ontology-Tradecraft\projects\project-6` with `.venv-mowl` activated:

```powershell
Expand-Archive `
  -LiteralPath "$env:USERPROFILE\Downloads\Project6_Project5_Joint_mOWL_Transfer.zip" `
  -DestinationPath . `
  -Force
```

## Step 1: Check offline tests

```powershell
python -m unittest discover -s scripts -p "test_experiment_joint_rail_mowl.py" -v
```

**26** tests passed in the assistant's environment, including split/provenance validation, synthetic preflight, deterministic TF-IDF features, held-out controls, and original-file protection. They do **not** test actual mOWL Java/Python training here.

## Step 2: Preflight without retraining

```powershell
python -u scripts/experiment_joint_rail_mowl.py --project-root . --preflight-only
```

Expected on the current Project 6 split:

- Outer train: **665** named subclass pairs
- Outer validation: **167** named subclass pairs
- Rail outer holdouts: **2**
- Additional inner-development set: a supported subset of the 665 training pairs

If the report hashes or 80/20 partitions mismatch, preflight **fails instead of reusing stale data**.

## Step 3: Run the Project-5-style mOWL experiment

```powershell
python -u scripts/experiment_joint_rail_mowl.py `
  --project-root . `
  --seed 42 `
  --configs 8 `
  --tune-epochs 250 `
  --final-epochs 700 `
  --threshold 0.70
```

This runs eight internally tuned native mOWL models, selects using only the inner development split, and fits one new model on all 665 outer training relationships. The two outer rail relationships and 165 outer source/import relationships are evaluated **only after** selection and final training.

For a smoke test of runtime integration, run with `--configs 1 --tune-epochs 2 --final-epochs 2` **before** the full experiment. Such a smoke-test score has no evidential value toward the 0.70 threshold.

## Step 4: Inspect result

New outputs:

- `reports/step5_joint_project5_comparison.json`
- `generated/mowl_joint_project5_class_vectors.npz`
- `generated/mowl_joint_project5_dev_train.ttl`

PowerShell:

```powershell
$r = Get-Content .\reports\step5_joint_project5_comparison.json -Raw | ConvertFrom-Json
$r.status
$r.rail_holdout.mean
$r.source_import_holdout.mean
$r.all_holdout.mean
$r.rail_matched_child_control_auc
$r.rail_positive_minus_matched_child_control
$r.all_holdout_control_auc
$r.baseline_mowl
$r.threshold_met
```

Criteria (conservative): **rail held-out cosine >= 0.70** AND **all held-out cosine >= 0.70**, with **positive/control separation >= 0.15** and **AUC >= 0.70** for both overall and rail-specific matched-child controls. Failure is reported as `FAIL_HOLDOUT_COSINE_OR_DISCRIMINATION`; the output still contains scores and per-pair diagnostics.

After the experiment, share `reports/step5_joint_project5_comparison.json` to assess results. Integrate into `scripts/run_all.py` only if the new model actually generalizes.

## Validation caveats

- **Only two rail examples** are independently held out. Even a passing rail average alone is weak evidence. The larger source/import outer set and controls provide additional information.
- **Raw center cosine is symmetric** and does not establish asymmetric OWL subclass validity. The native EL loss and independent SHACL/HermiT remain necessary.
- Unasserted control pairs are **not OWL-proven negative statements**. They are discriminative diagnostics, not claims that their subclass statements are false.
- Definition text can implicitly disclose taxonomy relationships, so the study is a *transductive, axiom-held-out evaluation*, not a zero-knowledge unseen-class benchmark.
- No automatic Grok review is run, no axiom is automatically approved, no source or validated ontology is overwritten, and original `run_all.py` is unchanged.
- Project 6's previous completed 13-stage SHACL/scoped-HermiT run remains valid within its reported scope, while the 0.70 cosine requirement remains separately unresolved until experimentally demonstrated.
