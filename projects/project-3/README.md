# Week 5 Project: Targeted and Global Reasoning

This week’s project builds on skills in **SPARQL querying within Jupyter notebooks** and introduces **reasoning with OWL2 profiles** using standard reasoners (HermiT, ELK). Students will apply both targeted and global reasoning approaches to ontology mappings between:

- Basic Formal Ontology (BFO) and the [Information Exchange Standard](https://www.informationexchangestandard.org/) (IES)  
- Common Core Ontologies Units of Measure (CCOM) module and [Quantities, Units, Dimensions and Data Types](https://www.qudt.org/) (QUDT)  
- Common Core Ontologies Time Ontology (CCOT) module and the W3C [Time Ontology](https://www.w3.org/TR/owl-time/) (TO)  

## Learning Objectives

By the end of this project, students will be able to:

- Execute **SPARQL queries** directly in Jupyter notebooks to explore ontology content.  
- Execute **OWL2 reasoners** and debug inconsistencies.  
- Construct and assert translation definitions between ontologies   
- Construct and assert OWL axioms based on textual definitions to enrich semantic relationships
- Develop competence in **debugging code**, paying attention to **file names and directory structure**, reading **error messages, and so on

## Files in the Repository

- `readings/` - Lecture material and readings for this project. 
- `assignment/notebooks/week5_project.ipynb` — Notebook to extract IRIs and labels using SPARQL. 
- `assignment/notebooks/reasoner_run.ipynb` — Notebook to run OWL reasoning.
- `assignment/notebooks/augment_w_definitions.ipynb` — Notebook to extract IRIs, labels, and definitions with SPARQL. 
- `assignment/src/data/` — Where you will store your various submission files for this project.   
- `assignment/src/compare_structures.py` — Python script run in the terminal to compare axioms stucture across ontology files in `src/`. 
- `assignment/src/bfo-core.ttl` — Basic Formal Ontology file.  
- `assignment/src/ies.ttl` — Information Exchange Standard file.  
- `assignment/src/qudt.ttl` — QUDT file. 
- `assignment/src/ccot.ttl` - CCO Time Ontology file
- `assignment/src/ccom.ttl` - CCO Units of Measure file
- `assignment/src/time.ttl` - W3C Time Ontology file
- `assignment/tests/test_project3.py` — Automated grading tests for queries and reasoning.  
- `requirements.txt` - You tell me what this is. 
- `README.md` — Project description (this file).  

## Project

1. **Generate List of Labels and IRIs**  
   - Successfully run `week5_endpoint_client.ipynb` to return a list of IRIs and rdfs:labels for each BFO, IES, QUDT, CCOM, CCOT, and TO ontology
   - If you do this correctly, you will generate files with names such as `bfo-class.xlsx`, for BFO, IES, QUDT, CCOM, CCOT, and TO saved to `data/`

2. **Construct SPARQL & Return List of Labels, IRIs, and Axioms**  
    - Study the `week5_endpoint_client.ipynb` to understand how to leverage rdflib with SPARQL
    - Construct SPARQL queries for each ontology that are designed to return a complete list of classes and associated axioms. Note: Do not return deprecated classes; your query should a file with a column for IRIs, a column for rdfs:labels, and a column for axioms associated with each IRI 
    - Successfully run your SPARQL queries in a Jupyter notebook, named `class-axiom-generator.ipynb`; save this in the `notebooks` directory
    - Ensure your code saves the resulting lists following the convention: `ontology abbreviation-axioms.xlsx`. Note: Be sure your xlsx file has a column for "iri" and a column for "axiom"

3. **Compare Axiom Sets across Ontologies**
   - Successfully `compare_structures.py` which compares axiom sets from BFO/IES, CCOM/QUDT, and CCOT/TO; the script should return a list of structural matches
   - Check the results by running in a terminal the following: 
    ```
    pip install rdflib pandas openpyxl
    ```
    ```
    python compare_structures.py \
    --bfo src/bfo-core.ttl \
    --ies src/ies.ttl \
    --outdir src/data/ \
    --shape coarse --presence-only --normalize families
    ```
    ```
    python compare_structures.py \
    --ccom src/ccom.ttl \
    --qudt src/qudt.ttl \
    --outdir src/data/ \
    --shape coarse --presence-only --normalize families
    ```
    ```
    python compare_structures.py \
    --ccot src/ccot.ttl \
    --to src/time.ttl \
    --outdir src/data/ \
    --shape coarse --presence-only --normalize families
    ```
   - Read the help information below stored under **compare_structures.py Help** for guidance on flags and matching options
   - Your file should be saved in `src/data/` and named following the convention: `ontologyabbreviation1-ontologyabbreviation2-structural-matches.xlsx`

4. **Explore Semantic Relationships**
    - In the `*-structural-matches.xlsx` files note each row potentially exhibits some semantic relationship between ontologies
    - Inspect labels + axioms manually, keeping those with clear semantic relationship and dropping those without
    - For each pair deemed semantically related, assert owl:equivalence (if you believe they are semantically equivalent) or rdfs:subclassOf (if you believe one is a subclass of the other); the result should be triples like (replacing "class_1" with a BFO class and "class_x" with an IES class, and so on): 
        ```
        bfo:class_1 owl:equivalentClass ies:class_x .
        bfo:class_2 rdfs:subClassOf ies:class_y .
        ```
    - Create new files reflecting your choices, e.g. `bfo-mapping.ttl` / `ies-mapping.ttl`, `ccom-mapping.ttl` / `qudt-mapping.ttl`, `ccot-mapping.ttl` / `time-mapping.ttl`; be sure to store these in `src/data/`

5. **Install ROBOT and Run Reasoners**
    - Install the [ROBOT](https://robot.obolibrary.org/) tool if you have not already
    - Successfully execute the `reasoner_run.ipynb` notebook to run **HermiT** and **ELK** against your files, checking for consistency

6. **Return List of Labels, IRIs, and Textual Definitions**
    - Successfully run `augment_w_definitions.ipynb` which updates each `*-structural-matches.xlsx` with textual definitions for each IRI in the file
    - Ensure the results are stored following the naming convention `*-structural-matches-with-defs.xlsx` in the `data` directory

7. **OWL Enrichment**
   - Reviewing each row in each `*-structural-matches-with-definitions.xlsx` file and compare textual definitions to determine whether there is semantic overlap
   - In the event you identify semantic overlap between textual definitions, you should then locate the relevant IRIs in their respective ontology files and assert OWL axioms that reflect this semantic overlap
   - For example, if the textual definition for 'calendar month' in CCOT is paired with 'month of the year' in TO, then you should assert axioms in CCOT that reflect the textual definition of TO and then in TO assert axioms that reflect the textual definitions of CCOT
   - Do not change the files names for your augmented ontology files, e.g. `bfo-core.ttl` should remain the file name and remain in the `src` directory

8. **Repeat**
   - Return to step 2 and run your `class-axiom-generator.ipynb` scripts again replacing your existing `ontology abbreviation-axioms.xlsx` file with an updated version with the same name
   - Successfully run `compare_structures.py` again and then replace in the `data/` directory your file named following the convention: `ontologyabbreviation1-ontologyabbreviation2-structural-matches.xlsx`; keep the same name
   - Repeat step 4 and replace your `bfo-mapping.ttl` / `ies-mapping.ttl`, `ccom-mapping.ttl` / `qudt-mapping.ttl`, `ccot-mapping.ttl` / `to-mapping.ttl` stored in `data/` with files updated based on your richer axiom sets
   - Successfully execute the `reasoner_run.ipynb` notebook to run **HermiT** and **ELK** against your files, checking for: consistency, inferred equivalences/subclasses, and differences across reasoners

## Rubric (100 pts)

### SPARQL Exports: IRIs & Labels (15 pts)

- **(10 pts)** Each ontology (BFO, IES, CCOM, QUDT, CCOT, TO) has a `*-class.xlsx` containing at least one column of IRIs and one column of labels.  
- **(5 pts)** Files open without error; no obviously empty exports (≥10 rows total across all exports).  

### SPARQL Exports: IRIs, Labels & Axioms (20 pts)

- **(15 pts)** Each ontology has a `*-axioms.xlsx` with at least IRI and axiom(s) columns.  
- **(5 pts)** No deprecated classes included (evidence: no rows with labels containing “deprecated” or axioms containing `owl:deprecated "true"`).  

### Structural Matching Outputs (15 pts)

- **(15 pts)** For each pair (BFO–IES, CCOM–QUDT, CCOT–TO), a file named `data/{pair}-structural-matches.xlsx` or the reverse pair name exists. Each file has two IRI columns (any reasonable naming that includes “iri”) and ≥1 row.  

### Definitions Augmentation (20 pts)

- **(20 pts)** For each pair (BFO–IES, CCOM–QUDT, CCOT–TO), a `*-structural-matches-with-definitions.xlsx` exists with both left and right definition columns present and at least some non-empty values.  

### OWL Enrichment (20 pts)

- **(15 pts)** If any of `data/bfo-mapping.ttl`, `data/ies-mapping.ttl`, `data/ccom-mapping.ttl`, `data/qudt-mapping.ttl`, `data/ccot-mapping.ttl`, `data/to-mapping.ttl` exist, they parse and contain at least one `owl:equivalentClass` or `rdfs:subClassOf` triple.  
- **(5 pts)** Core ontology TTLs in `assignment/src/` still parse after enrichment.  

### Reproducibility & Organization (10 pts)

- **(5 pts)** Files are in the documented locations/names (`assignment/src/`, `assignment/src/data/`, notebook folder).  
- **(5 pts)** Required notebooks exist (`week5_project.ipynb`, `reasoner_run.ipynb`, `augment_w_definitions.ipynb`). (Execution is not autograded—presence only.)  

## Testing your Work
We will leverage GitHub Actions to automate the grading for projects. There will accordingly be `*-tests.yml` files under the `.github/workflows directory`. These files provide instructions for when tests within each project should run against your submissions. There is a trigger, for example, such that when you open a pull request to the class repository, tests will run againsts your pull request submission. The portion of the yml file that determines triggers for project 1 in `project3-tests.yml` looks like this: 
```
on:
  pull_request:
    paths:
      - "projects/project-3/**"
      - ".github/workflows/project3-tests.yml"
```
The block that starts with "pull_request" says on a pull request to my repository run the yml instructions that follow. 

I suspect you will want to test your work before you submit it to me though. If that is the case, then you will want to include another trigger that runs when you push updates to your own repository. To make that happen, you will need to update the yml file you have on your repository so it looks like: 
```
on:
  pull_request:
    paths:
      - "projects/project-3/**"
      - ".github/workflows/project3-tests.yml"
  push:
    paths:
      - "projects/project-3/**"
      - ".github/workflows/project3-tests.yml"
```
This additional block that starts with "push" says that on a push to your repository, run the yml instructions that follow. 

## compare_structures/py Help

Use this guide when running `compare_structures.py`  
It explains what each flag does, when to use it, and includes copy-pasteable recipes

### Required Paths

#### `--left PATH` / `--right PATH`
Ontology files to compare (`.ttl`, `.owl`, RDF/XML, etc.)

#### `--outdir PATH`
Where the output XLSX goes  
Default: current directory
Filename: `<left-stem>-<right-stem>-structural-matches.xlsx`

### Shape Granularity

#### `--shape {exact|kind|coarse}`
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

### Normalization

#### `--normalize {off|entailment|families}`
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

### Token Multiplicity

#### `--presence-only`
- Treat shapes as a **set** (ignore how many times a token appears)  
- Without it (default), shapes are **multisets** (token counts matter)  
- Use `--presence-only` for discovery; drop it for stricter, final matches

### Quick Recipes

#### 1. Broad Match (find any overlap)
```
--shape coarse --normalize families --presence-only \
--follow-imports --imports-depth 2
```
#### 2. Moderate Match triage (find some overlap)
```
--shape kind --normalize entailment --presence-only \
--follow-imports --imports-depth 2
```
#### 3. Strict Match (find exact overlap)
```
--shape exact --normalize off \
--follow-imports --imports-depth 2
```

### Output File Guide

- R:some — an existential restriction (… some …)
- R:only — a universal restriction (… only …)
- R:has — a hasValue restriction (… value v)
- R:min=2 / R:max=1 / R:card=1 — (unqualified) cardinalities
- R:qmin=1 / R:qcard=1 — qualified cardinalities (owl:onClass)
- ×k suffix in output (e.g., R:card×2) = token appeared k times