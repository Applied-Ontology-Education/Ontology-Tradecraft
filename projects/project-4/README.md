# Week 4 Project: ETL + Mapping + Validation Midterm

## Learning Objectives

By the end of this project, you will be able to:

1. Extract, normalize, and transform provided measurement data into a structured format prior to building RDF.
2. Using RDFlib, construct RDF graphs that conform to a provided Common Core Ontologies (CCO) design pattern.
3. Construct and apply SPARQL quality-control (QC) queries to verify ontology integrity.
4. Construct and apply SHACL shapes to check semantic conformance and data integrity.
5. Automate semantic validation in a continuous integration (CI) pipeline using GitHub Actions and new data that is run through your ETL script.

This project serves as your midterm milestone, combining data engineering, ontology modeling, and semantic quality assurance within a reproducible workflow.

## Project Overview

You are provided with two raw data sources (`src/data/sensor_A.csv` and `src/data/sensor_B.json`) containing measurement data with inconsistent schemas, units, and encoding. Your task is to:

1. Combine both raw files into a clean intermediate dataset (`src/data/readings_normalized.csv`) using the Python library Pandas. Guidance for the construction of this script can be found in `src/ETL_Guide.md`. Store your script in `src/scripts/normalize_readings.py`. 
2. Using RDFlib, construct RDF triples that instantiate a provided CCO Measurement Design Pattern (`src/cco_measurement_design_pattern.pptx`), outputting the graph to `src/measure_cco.ttl`. Store your RDFLib script as `src/scripts/measure_rdflib.py`.
3. Using the descriptions in each `*.rq` file located in `src/sparql/`, construct 8 SPARQL QC queries, each of which should return **zero results** when the queried structure is valid. For example, one such query will evaluate whether each ontology element includes a unique `rdfs:label` value. If so, this query will return no results; otherwise, the offending ontology element will be returned. 
4. Run the `*.rq` queries automatically using `src/scripts/run_sparql_qc.py`, validating your ontology against the queries.
5. Using the descriptions found in the `src/cco_shapes.ttl` file, construct SHACL shapes to validate your RDF graph’s structure and datatype conformance. An example is provided that enforces every artifact bears some specifically dependent continuant. 
6. Run `cco_shapes.ttl` file automatically using `src/scripts/shacl_validate.py`, validating your ontology against the shapes.
7. I have provided a `yml` file that will run your ETL script whenever you add a new data set to the `src/data/` folder. To conclude this project, you will follow the guidance in `src/Workflow_Guide.md` to setup a GitHub Action workflow, after which you will move the data set `src/sensor_C.csv` into the `src/data/` folder. If done correctly, you will kick off a build which will run your ETL script to update `src/data/readings_normalized.csv` with this new data, followed by your RDFLib script which will update `src/measure_cco.ttl`, which will then be tested against your SPARQL and SHACL for validation. 

When you have successfully concluded this project, you will have created a CI workflow in which you clean data, transform it into a CCO-based RDF graph, which is then validated for quality and accuracy. This pipeline can be used to automate the integration of similarly structured data sources, resulting in high-quality CCO-based knowledge graphs. 

## Automated Grading

Your work will be automatically graded using the `.github/project4-tests.yml` workflow, which will check: 
1. That your `src/scripts/normalize_readings.py` script generates a `src/data/readings_normalized.csv` file according to the canonical scheme found in `src/ETL_Guide.md`.
2. That your `src/scripts/measure_rdflib.py` script generates a `src/measure_cco.ttl` file that aligns with the design pattern found in `cco_measurement_design_pattern.pptx`
3. Your `src/sparql/` includes 8 valid SPARQL queries according to the guidance provided.
4. Your `src/cco_shapes.ttl` file contains shapes according to the guidance provided. 
5. Your PR includes `data/sensor_C.csv` and runs successfully using the `.github/ontology_workflow.yml` file in the main course directory. 

**Files you Must Supply:**
You will need to supply:
1. `src/scripts/normalize_readings.py`
2. `src/data/readings_normalized.csv`
3. `src/scripts/measure_rdflib.py`
4. `src/measure_cco.ttl`
5. 8 queries under `src/sparql/`
6. A completed `src/cco_shapes.ttl` file
7. A successful build using `data/sensor_C.csv` and `.github/ontology_workflow.yml`

## Files in the Repository

In this repository you will find: 
- `src`- Folder which incldues the main content of the project
- `src/data`- Folder containing data sets used in this project, namely `sensor_A.csv` and `sensor_B.json`, as well as the `src/data/readings_normalized.csv` output of your `src/scripts/normalize_readings.py` script
- `src/scripts`- Folder containing python scripts to run SHACL and SPARQL against your produced ttl file, as well as where you will store your `src/scripts/normalize_readings` and `src/scripts/measure_rdflib.py` scripts
- `src/shacl/`- Where you will store your `cco_shapes.ttl` file 
- `src/sparql/`- Where you will store your 8 `.rq*` files 
- `cco_merged.ttl` - A merged ttl file of all Common Core Ontology modules; you should be using BFO and CCO IRIs from this file
- `ETL_Guide.md` - A walkthrough for how to construct an RDFlib script to ingest data and output a CCO design pattern aligned ontology
- `Workflow_Guide.md` - A walkthrough for how to construct a GitHub workflow to implement the semantic pipeline you build during this project
- `sensor_C.csv` - Data you will move to the `data/` folder to trigger your workflow
- `measure_cco.ttl` - The output of your `src/scripts/measure_rdflib.py` script that produces CCO design pattern aligned ttl
- `cco_measurement_design-pattern.pptx` - A basic CCO design pattern concerning measurements that I have provided for this project
- `src/ontology_workflow.yml` - The yml file you will move to your `.github/workflows` folder to set up the GitHub workflow
- `requirements.txt` - You tell me what this is...

