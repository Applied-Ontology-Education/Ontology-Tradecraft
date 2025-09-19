# Week 5 Project: Targeted and Global Reasoning

This week’s project builds on skills in **SPARQL querying within Jupyter notebooks** and introduces **reasoning with OWL2 profiles** using standard reasoners (HermiT, Pellet, ELK). Students will apply both targeted and global reasoning approaches to ontology mappings between:

- Basic Formal Ontology (BFO) and the [Information Exchange Standard](https://www.informationexchangestandard.org/) (IES)  
- Common Core Ontologies Units of Measure (CCOM) module and [Quantities, Units, Dimensions and Data Types](https://www.qudt.org/) (QUDT)  
- Common Core Ontologies Time Ontology (CCOT) module and the W3C [Time Ontology](https://www.w3.org/TR/owl-time/) (TO)  

## Learning Objectives

By the end of this project, students will be able to:

- Execute **SPARQL queries** directly in Jupyter notebooks to explore ontology content.  
- Execute **OWL2 reasoners** and debug inconsistencies.  
- Construct and assert translation definitions as owl:equivalentClass axioms   
- Construct and assert OWL axioms for potential mappings from textual definitions
- Test less than equivalent OWL classes using test instances  

## Project

1. **Task 1**  
   - Successfully run `app.py` in the endpoint directory 
   - Successfully run `week5_endpoint_client.ipynb` to retrieve a list of classes from the ontology and return a column for IRIs and a column for rdfs:labels
   - If you do this correctly, you will generate files with names such as `bfo-class.xlsx`, for BFO, IES, QUDT, CCOM, CCOT, and TO  

2. **Task 2**  
    - Study the `week5_endpoint_client.ipynb` to understand how to leverage rdflib with SPARQL
    - Construct SPARQL queries for each ontology that are designed to return a complete list of classes and associated axioms. Note: Do not return deprecated classes; your query should a file with a column for IRIs, a column for rdfs:labels, and a column for axioms associated with each IRI 
    - Successfully run your SPARQL queries in a Jupyter notebook, named `class-axiom-generator.ipynb`
    - Ensure your code saves the resulting lists following the convention: `ontology abbreviation-axioms.xlsx`

3. **Task 3**
   - Successfully `compare_structures.py` which compares axiom sets from BFO/IES, CCOM/QUDT, and CCOT/TO; the script should return a list of structural matches
   - Check the results by running in a terminal the following: 
    ```
    pip install rdflib pandas openpyxl
    ```
    ```
    python compare_structures.py \
    --bfo src/bfo-core.ttl \
    --ies src/ies.ttl \
    --outdir data/ \
    --shape coarse --presence-only --normalize families
    ```
    ```
    python compare_structures.py \
    --ccom src/ccom.ttl \
    --qudt src/qudt.ttl \
    --outdir data/ \
    --shape coarse --presence-only --normalize families
    ```
    ```
    python compare_structures.py \
    --ccot src/ccot.ttl \
    --to src/time.ttl \
    --outdir data/ \
    --shape coarse --presence-only --normalize families
    ```
   - Read the README.md help file stored in `assignment/endpoint/` for guidance on flags and matching options
   - Your file should be saved in `data/` named following the convention ontology `ontologyabbreviation1-ontologyabbreviation2-structural-matches.xlsx`

4. **Task 4**
    - In the `*-structural-matches.xlsx` files note each row is a potential candidate equivalent classes (IRI, label, axioms)
    - Inspect labels + axioms manually, keeping those with clear semantic alignment and dropping those without, e.g., *Week* ↔ *MonthOfYear*
    - For each accepted pair, assert equivalence or subclass:
        ```turtle
        bfo:continuant owl:equivalentClass ies:Entity .
        bfo:process rdfs:subClassOf ies:Event .
        ```
    - Create new files reflecting your choices, e.g. `bfo-mapping.ttl` / `ies-mapping.ttl`, `ccom-mapping.ttl` / `qudt-mapping.ttl`, `ccot-mapping.ttl` / `to-mapping.ttl`; be sure to store these in `data/`
    - Install the [ROBOT](https://robot.obolibrary.org/) tool if you have not already
    - Successfully execute the `reasoner_run.ipynb` notebook to run **HermiT** and **ELK** against your files, checking for: consistency, inferred equivalences/subclasses, and differences across reasoners

5. **Task 5**
    - Study the `week5_endpoint_client.ipynb` to understand how to leverage rdflib with SPARQL
    - Construct SPARQL queries for each ontology that are designed to return a complete list of classes and associated textual definitions. Note: Do not return deprecated classes; your query should a file with a column for IRIs, a column for rdfs:labels, and a column for textual definition associated with each IRI 
    - Successfully run your SPARQL queries in a Jupyter notebook, named `class-definitions-generator.ipynb`
    - Ensure your code saves the resulting lists following the convention: `ontology abbreviation-definitions.xlsx`

**Rubric:**

- **SPARQL Queries (50%)**  
  - Queries execute without error in Jupyter.  
  - Results correspond to expected ontology entities and relations.  

- **Reasoning (20%)**  
  - Reasoners (HermiT, ELK) are executed correctly.  
  - Correct classification and consistency-check results are generated.  

- **Code Quality & Reproducibility (30%)**  
  - Notebook runs start-to-finish without errors.  
  - Code and documentation are clearly presented.  

## Files in the Repository

- `readings/` - Lecture material and readings for this project. 
- `assignment/notebooks/week5_project.ipynb` — Main notebook for SPARQL queries and reasoning tasks. 
- `assignment/src/data/` — Where you will store your various submission files for this project.   
- `assignment/src/bfo-core.ttl` — Basic Formal Ontology file.  
- `assignment/src/ies.ttl` — Information Exchange Standard file.  
- `assignment/src/qudt.ttl` — QUDT file. 
- `assignment/src/ccot.ttl` - CCO Time Ontology file
- `assignment/src/ccom.ttl` - CCO Units of Measure file
- `assignment/src/time.ttl` - W3C Time Ontology file
- `assignment/tests/test_project3.py` — Automated grading tests for queries and reasoning.  
- `requirements.txt` - You tell me what this is. 
- `README.md` — Project description (this file).  
