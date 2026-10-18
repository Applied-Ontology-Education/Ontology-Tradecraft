# Week 1 Project

## Learning Objectives
By completing this project, you will:

- Confirm that your Python environment is ready with required packages.
- Practice working with RDF data using `rdflib`.
- Experience the GitHub workflow (clone → commit → push → automated tests).
- Understand how **autograding** will work for future, more complex projects.

## Project
1. **Clone the starter repository** https://github.com/Applied-Ontology-Education/Ontology-Tradecraft.git.  
2. In your local repository, create a new Jupyter Notebook in `notebooks/` called `env-report.ipynb`, where you
   - Print your Python version.
   - Import `rdflib`, `SPARQLWrapper`, and `pyshacl`.
   - Parse the provided file `sample.ttl` and print the number of triples it contains.  
3. Commit and push your notebook and any code changes to GitHub.  
4. Verify that your submission passes the automated tests; if not, return to step 2. 

## Automated Grading
Your work will be graded automatically using **pytest** and GitHub Actions. Each push or pull request runs the tests, and results are shown in the **Checks** tab of your repo.

## Testing your Work
We will leverage GitHub Actions to automate the grading for projects. There will accordingly be `*-tests.yml` files under the `.github/workflows directory`. These files provide instructions for when tests within each project should run against your submissions. There is a trigger, for example, such that when you open a pull request to the class repository, tests will run againsts your pull request submission. The portion of the yml file that determines triggers for project 1 in `project1-tests.yml` looks like this: 
```
on:
  pull_request:
    paths:
      - "projects/project-1/**"
      - ".github/workflows/project1-tests.yml"
```
The block that starts with "pull_request" says on a pull request to my repository run the yml instructions that follow. 

I suspect you will want to test your work before you submit it to me though. If that is the case, then you will want to include another trigger that runs when you push updates to your own repository. To make that happen, you will need to update the yml file you have on your repository so it looks like: 
```
on:
  pull_request:
    paths:
      - "projects/project-1/**"
      - ".github/workflows/project1-tests.yml"
  push:
    paths:
      - "projects/project-1/**"
      - ".github/workflows/project1-tests.yml"
```
This additional block that starts with "push" says that on a push to your repository, run the yml instructions that follow. 

**Rubric:**
- **Python environment** (Python ≥ 3.0)  
- **Required libraries import successfully** 
- **RDF parsing works** (`sample.ttl` parsed with correct triple count = 2)   
- **Notebook executes without error**   

A passing grade requires all tests to succeed.

## Files in the Repository
- `notebooks/sample.ttl` – An RDF file with 2 triples.  
- `src/check_env.py` – A script that parses the sample RDF and prints results in JSON.  
- `tests/test_env.py` – Automated tests to verify your setup.  
- `.github/workflows/autograde.yml` – GitHub Actions workflow for autograding.  
- `requirements.txt` – Python dependencies.