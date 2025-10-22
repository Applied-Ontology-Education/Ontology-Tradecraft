# GitHub Actions Setup Guide

## What the Workflow Does
Whenever you **add one new dataset** (like `sensor_C.csv`) to the `src/data/` folder and push to GitHub, the provided workflow will automatically:

1. Run your **ETL script** to update `src/data/readings_normalized.csv`.  
2. Run your **RDFlib script** to rebuild `src/measure_cco.ttl`.  
3. Run **SPARQL QC** checks to verify annotation and logical consistency.  
4. Run **SHACL** validation to confirm data integrity.

If everything is correct, your workflow run will show a ✅ next to each step in GitHub Actions.

## Step 1 — Update Your ETL Script

Before running the workflow, update your ETL script (`src/scripts/normalize_readings.py`) so that it includes all files in the `src/data/` folder.  
Your script must read any new dataset (for example, `sensor_C.csv`) and combine it with the existing ones.

```from pathlib import Path
import pandas as pd

IN_A = Path("src/data/sensor_A.csv")
IN_B = Path("src/data/sensor_B.json")
IN_C = Path("src/data/sensor_C.csv")  # THIS A THE NEW LINE
OUT  = Path("src/data/readings_normalized.csv")

def main():
    print("[paths] A:", IN_A)
    print("[paths] B:", IN_B)
    print("[paths] C:", IN_C)         # THIS A THE NEW LINE
    df_a = load_sensor_a(IN_A)
    df_b = load_sensor_b(IN_B)
    df_c = load_sensor_a(IN_C)        # THIS A THE NEW LINE

    print(f"[normalize_readings] Input A rows: {len(df_a)}")
    print(f"[normalize_readings] Input B rows: {len(df_b)}")
    print(f"[normalize_readings] Input C rows: {len(df_c)}")    # THIS A THE NEW LINE

    combined = pd.concat([df_a, df_b, df_c], ignore_index=True) # UPDATED LINE
    cleaned = normalize_and_clean(combined)
```

## Step 2 — Check Respository Structure

Your repository should be structured as follow (note you will need to move `src/ontology_workflow.yml` to `.github/workflows` in your local setup): 
```
src/
  data/
    sensor_A.csv
    sensor_B.json
  scripts/
    normalize_readings.py
    build_rdf.py
    run_sparql_qc.py
    shacl_validate.py
  sparql/
    *.rq
src/
  cco_shapes.ttl
requirements.txt
.github/
  workflows/
    ontology-pipeline.yml  ← provided
```
## Step 3 - Trigger Workflow

When you move `sensor_C.csv` into the `src/data/` folder then commit and push your changes to GitHub, you should trigger a workflow. 

To check, following such a push, navigate to the Actions tab in your GitHub repository and look for the workflow named: "Ontology Workflow". Click it to view the progress and logs.

## Step 4 - Success
You’ve completed the project when:

✅ The workflow run shows green checks for every step.
✅ src/data/readings_normalized.csv includes rows from the new dataset.
✅ src/measure_cco.ttl has been updated.
✅ SPARQL QC prints “0 rows” for each query.
✅ SHACL validation shows “no Violations”.

## Troubleshooting 
- Workflow didn’t run: You must add a brand-new file under src/data/. Editing an existing file won’t trigger the workflow.
- ETL ran but missed your new data: Check that your ETL script (above) reads all files in src/data/.
- SPARQL or SHACL failed: Read the logs in the Actions view. Fix your RDF or SHACL constraints so that no errors or violations appear.
- You can check your workflow locally by navigating in your terminal to the `project-4/assignment` directory, then running: 

```
python src/scripts/normalize_readings.py
python src/scripts/measure_rdflib.py
python src/scripts/run_sparql_qc.py
python src/scripts/run_shacl_validate.py
```