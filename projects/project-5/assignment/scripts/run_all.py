#Assignmentsrun_all.py file
#single script to create the full pipeline 
 #   organizae my scriopts with inputs and outputs and then final run by robot with a print out of the output.

#1 Imports
import json #read mowl_metrcis.json fil
import subprocess #run the scripts in order and ROBOT  
import sys #need for the sys.executable to run the scripts in the same environment as this script
from pathlib import Path
import urllib.request #download ROBOT if it is not already present

#2 Paths
# 2) Paths and settings
BASE = Path(__file__).resolve().parent.parent      # the assignment folder
SCRIPTS = BASE / "scripts"
MODULE = BASE / "src" / "cco-module.ttl"
##CANDIDATES = BASE / "generated" / "candidate_el.ttl"
ACCEPTED = BASE / "generated" / "accepted_el.ttl"
AUGMENTED = BASE / "src" / "module_augmented.ttl"
#VECTORS = BASE / "data" / "mowl_embeddings.json"
METRICS = BASE / "reports" / "mowl_metrics.json"

ROBOT_JAR = BASE / "tools" / "robot.jar"
ROBOT_URL = "https://github.com/ontodev/robot/releases/download/v1.9.11/robot.jar"

# from readme:
#Write a single driver script `scripts/run_all.py` that executes the full pipeline:
 #    - `extract_definitions.py` → `data/definitions.csv`
  #   - `preprocess_definitions_llm.py` → `data/definitions_enriched.csv`
  #   - `generate_candidates_llm.py` → `generated/candidate_el.ttl`
  #   - Split axioms → `src/train.ttl`, `src/valid.ttl`
  #   - `train_mowl.py` → `reports/mowl_metrics.json`
  #   - `filter_candidates_hybrid.py` → `generated/accepted_el.ttl`
  #   - Merge + reason → `src/module_augmented.ttl`

#set of pairs of script and output file, in order, to run the full pipeline
STEPS = [
    ("extract_definitions.py","data/definitions.csv"),
    ("preprocess_definitions_llm.py","data/definitions_enriched.csv"),
    ("generate_candidates_llm.py","generated/candidate_el.ttl"),
    ("split_module.py","src/train.ttl, src/valid.ttl"),
    ("train_mowl.py","reports/mowl_metrics.json"),
    ("filter_candidates_hybrid.py","generated/accepted_el.ttl"),
]

# 3) Run Steps 1 to 6, one after the other
#    sys.executable = the Python of your .venv, so every script uses the same packages
#    check=True     = stop right away if a script fails
for script, output in STEPS: #the scripts nad outfile file
    print("\n=== Running", script, "->", output, "===", flush=True)   # status message before the next script runs
    subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=BASE, check=True) #runs a command (as if via the terminal) and run from the BASE (assignment) and have the installed librbaries available to the script. check=True means stop right away if a script fails
    
# 4) Get ROBOT the first time (about 80 MB)
if not ROBOT_JAR.exists(): #check to see if it is installed
    print("\nDownloading ROBOT to", ROBOT_JAR)
    ROBOT_JAR.parent.mkdir(exist_ok=True) #create the tools folder if it does not exist
    urllib.request.urlretrieve(ROBOT_URL, ROBOT_JAR) #download the robot.jar file from the ROBOT_URL and save it to ROBOT_JAR path.

# 5) merge  accepted axioms into the module and reason with ELK
#    --remove-redundant-subclass-axioms false keeps our new axioms in the output,
#    even when ELK can already infer them through the hierarchy.
print("\n=== Step 7: ROBOT merge + reason (ELK) -> src/module_augmented.ttl ===", flush=True)
robot = subprocess.run([
    "java", "-jar", str(ROBOT_JAR),
    "merge", "--input", str(MODULE), "--input", str(ACCEPTED), #combine the original module and the accepted axioms into one ontology
    "reason", "--reasoner", "ELK", #check the  ontology for consistency using  ELK reasoner
    "--remove-redundant-subclass-axioms", "false",
    "--output", str(AUGMENTED), #creation of the augmented module with the accepted axioms merged in and reasoned over
], cwd=BASE)

#    ROBOT stops with an error if the ontology is inconsistent or there are unsatisfiable classes. A clean exit means both checks passed.
if robot.returncode == 0:
    consistency = "CONSISTENT, 0 unsatisfiable classes"
else:
    consistency = "FAILED (see the ROBOT messages above)"

# 6) Final summary: read the numbers saved by Steps 5 and 6
metrics = json.loads(METRICS.read_text())
hybrid = metrics["hybrid"]

print("\n=== Project 5 summary ===")
print("Mean cosine (valid):", metrics["mean_cos"], " passed:", metrics["passed"])
print("Tau:", metrics["tau"])
print("Candidates:", hybrid["candidates"],
      " scored:", hybrid["scored"],
      " accepted:", hybrid["accepted"],
      " skipped (disjoint):", hybrid["skipped_disjoint"])
print("LLM model:", hybrid["llm_model"])
print("Reasoner check (ELK):", consistency)
print("Augmented module:", AUGMENTED)

# Stop with an error code if ROBOT failed, so a grader or CI sees the failure
if robot.returncode != 0:
    sys.exit(1)

    