# Week 2 Project: Enriching Raw Data

## Learning Objectives
By completing this project, you will:
- Leverage **ontology modeling strategies** to design a small RDF graph, **reusing** ontology elements where sensible.
- **Manually construct** RDF from **raw data** using basic design principles.
- Follow **Linked Data best practices** (stable HTTP IRIs, labels, datatypes).
- Pass an **autograder** that verifies your RDF **passes basic validation tests**.

## Project

Use the data sets below to complete the modeling tasks in RDF listed below. You are expected to reuse ontology content from the web, though you must make sure it is conformant to Basic Formal Ontology (BFO).

### aircraft_data.xlsx  
- **TASK-1** - Provide a BFO-conformant model of all columns of the **Airbus A320 NEO** row 
- **TASK-2** - The **Airbus A321-111** is designed to have a maximum knot approach speed of 142; after 5 approaches, an instance has obtained an average maximum knot approach speed of 139; provide a BFO-conformant model of this scenario. 

### soc_structure_definitions.xlsx  
- **TASK-3** - **Three SOC_TITLE entries** mention **“Aerospace Engineer”**; provide a BFO-conformant model of these entries and their respective SOC definitions.  

### employment_wage_May_2024.xlsx  
- **TASK-4** - **Three OCC_TITLE entries** mention **“Aerospace Engineer”**; provide a BFO-conformant model of all three entries and all columns associated with each. 

Your resulting RDF will be autograded using SPARQL, but you will not be required to create any SPARQL during this exercise.  

## Guidance 
- **Feel the Pain** - For this project, you must **hand-author** the RDF; do *not* write a Python ETL script.  
- **Required Files** - You must submit a `core.ttl` that defines classes and relations and `owl:imports` other ontologies which are stored in the `modules/` directory; you must submit an `instances.ttl` file for instances.
- **Labels and Definitions** - All classes, relations, and instances must have at least one `rdfs:label` for human readability and one `skos:definition` for human comprehension. 
- **Literals** - All literal values, and there will be many, must have an appropriate `xsd` datatypes. 
- **Key Classes/Relations** - Your base IRI is your choice, but you must use these names somewhere in your vocabulary:
  - **TASK-1** `Artifact Design`, `Jet Engine`, `Fixed Wing`, `has continuant part`, `has Model BADA`, `has Model FAA`, `has Length Ft`, `has Tail HeightFt`, `is about`
  - **TASK-2** `Test Process`, `is temporal part of`, `is measurement unit of`, `prescribes`, `Measurement Unit`
  - **TASK-3** `Act of Aircraft Processing`, `Adaptability Evaluation`, `is about`, `realizes`, `is output of`, `Artifact Design`, `inheres in`
  - **TASK-4** `quality`, `Act of Measuring`, `Total Employment`, `has area`, `o group`, `own code`
- **BFO Conformance** - You must leverage BFO classes and object properties correctly; for example, `Jet Engine` must be a subclass (directly or via intermediary classes) of `material entity`, `Test Process` must be a subclass (directly or via intermediary classes) of `Process`, `Act of Aircraft Design` must be a subclass (directly or via intermediary classes) of `generically dependent continuant`, and so on. 

## Automated Grading

The autograder checks:
- All `.ttl` files in `src/data/` parse.  
- **Task 1**: A320 NEO spec has 2 engines, Airbus as manufacturer, BADA/FAA codes typed, length/tail height values typed.  
- **Task 2**: A321-111 spec prescribes 142 knots; a test process with 5 approach part measurements averaging 139.  
- **Task 3**: 3 SOC Aerospace Engineer roles, with activity/output patterns modeled.  
- **Task 4**: 3 OCC Aerospace Engineer roles, with a population→measure→total employment pattern and at least one numeric total.  

Grading is all or nothing. For each, your submission must parse and pass the relevant tests for full credit; otherwise, you get nothing. Each task is worth **25% of the grade for this project, each**. 

## Testing your Work
We will leverage GitHub Actions to automate the grading for projects. There will accordingly be `*-tests.yml` files under the `.github/workflows directory`. These files provide instructions for when tests within each project should run against your submissions. There is a trigger, for example, such that when you open a pull request to the class repository, tests will run againsts your pull request submission. The portion of the yml file that determines triggers for project 1 in `project2-tests.yml` looks like this: 
```
on:
  pull_request:
    paths:
      - "projects/project-2/**"
      - ".github/workflows/project2-tests.yml"
```
The block that starts with "pull_request" says on a pull request to my repository run the yml instructions that follow. 

I suspect you will want to test your work before you submit it to me though. If that is the case, then you will want to include another trigger that runs when you push updates to your own repository. To make that happen, you will need to update the yml file you have on your repository so it looks like: 
```
on:
  pull_request:
    paths:
      - "projects/project-2/**"
      - ".github/workflows/project2-tests.yml"
  push:
    paths:
      - "projects/project-2/**"
      - ".github/workflows/project2-tests.yml"
```
This additional block that starts with "push" says that on a push to your repository, run the yml instructions that follow. 

## Files in the Repository
- `project-2/README.md` – Project description and guidance.  
- `project-2/notebooks` – Not needed for this project, but soon enough. 
- `project-2/readings/` – Readings to delight, amuse, inform, and perhaps confuse.  
- `project-2/assignment/src/` - Where you will locate the data files used for this project and `data` directory. 
- `project-2/assignment/src/data/` – Where you will store `core.ttl`, `*.ttl` modules, and `instances.ttl`  
- `project-2/assignment/tests/test_project2.py` – Autograder. Don't touch. 
- `project-2/assignment/requirements.txt` – Dependencies (`rdflib`, `pytest`).  

