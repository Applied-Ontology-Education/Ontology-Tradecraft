Project 5 notes

--Setup
    got the branch set up
    created all the folders and files (QUESTION __ Following same path as other project using an assignment top folder even though directions do not state.
    downloaded the cco module
    install pandas again.
    renamed my file from Information content entity to cco_modules

#Extract Definitions    
     reference prior scripts for can be leveraged as well as deciding on status of IRIs to pull
     include a statement to ignore deprecated IRIs
     ran script and got 151 rows and 1800 triples -- review the content.  no duplicate IRiS.  Each IRI has only one row (surprised by lack of duplicate entries)
      look up iif - if and only if

Step 2 -
    PICK THE LLM - Anthropic Claude Haiku 
    Perform all the set up activity
    Notes provided on the check llm connection
       - Script	Suggested max_output_tokens	to be added of 200.
            rewrite definitions	500	One definition plus thinking
            generate candidate axioms	1500	A JSON list of up to 3 axioms plus thinking
            rate plausibility	300	A short JSON score

Perform git status and clean up folders and update the requriements file to include perplexity AI and python-dotenv  (makes my settings available to scripts)    

Step 3:
   take the enriched definitions and return OWL 2 EL compliant candidate aciom in the candidate_el.ttl file from the definition_enriched.csv file
   for each class - send to the llm the definiion, current parent and all class lables
    LLM will choose up to 3 other classes where it is a kind of
    it willcheck each pick -- is it a real label, cannot be iteself, not in the module
    if it passes will be X subSubClassOf Y
   The general format is A 'class' is a 'supertype/parent class' that......

Step 4: need to generate the train and valid ttl files.
    - create a script to read the CCO module file and create the 2 files
        - scripts/split_module.py (not named in README - extra script, note in PR)
    - random number
        - random.seed(42) + sorted edges = same split every run
    - added networkx to count edges per class
        - only split edges where both classes have 2+ edges (keeps classes connected)
        - pip install networkx in .venv (requirements.txt alone does not install)
    - reviewed prior scripts, embeddings where there was prior code that could be reused.
        - reused split from embedding_example.py, changed 80/10/10 to 80/20
    - added declarations for 3 BFO parents not declared in the module
    - results: 110 edges, train = 90 (82%), valid = 20 (18%), all checks passed

Step 5:
    step 1 --get envronment ready based on the requirements files (update JPype1 from 1.6.0 to 1.5.1)
    java and mowl tested
    ran first set of the code to test and found error in scipy and fix structure and update requirements file
    start the script and add the mowl default values
    load the files into mowl and get the count of classes loaded

    step 2--
    set up the file to do the actual trainings.
        ran the script and received an error on a class
made updates to file referencing CCO modules

script is written - cosine of .196 was obtained

went in to update the settings:
increased emb dim and cosine went down

initial coding
mean_cos (valid):  0.196 (target 0.7)

change dim to from 50 to 30
mean_cos (valid):  0.148 (target 0.7)

change dim from 50  to 100
mean_cos (valid):  0.174 (target 0.7)

keep at 50 and try learning rate
.001 to .01
mean_cos (valid):  0.431 (target 0.7)

.01 to .1 - negatove so go back to .01
mean_cos (valid):  0.194 (target 0.7)

change epochs (passes from 1000 to 2000)
mean_cos (valid):  0.447 (target 0.7)

increase to 3000
mean_cos (valid):  0.459 (target 0.7)
increase to 5000 - not good.  go back to 3000
mean_cos (valid):  0.448 (target 0.7) 

change learning rate to 0.005
mean_cos (valid):  0.491 (target 0.7)

check margin 
change to .05
mean_cos (valid):  0.467 (target 0.7)
change to -.1
mean_cos (valid):  0.510 (target 0.7)
change to -.3
mean_cos (valid):  0.513 (target 0.7)

change epochs to 10000
mean_cos (valid):  0.496 (target 0.7)

mean_cos (valid):  0.548 (target 0.7)

continute to change the settings with out being able to get to .70

research and determine what is in the files
    looked at the train file and the cco module file to see what possible issues arise:
        Things I noticed were broken links and had thing as the parent.  Look at why the broken links occured.
            calendar system class is not in its correct hierarchy.
            abbreviated name is not in its correct hierarchy.
            designative name is not in its correct hierarchy
            etc.
      leaf plus first order parent.  if a deeper hierarchy does not show  
after updating the code -- new file was created and looked at and the structrue is quite different.
see if retraining gives a different result set. - no negative co-sines.
looked at the result set and compared the values and then looked at the json file where the cosine =.803
having the hierarchy, grandparents, in the training allowed 

Step 6: create hybrid filter
    - scripts/filter_candidates_hybrid.py -> generated/accepted_el.ttl
    - added to the train_mowl.py: save vectors to data/mowl_embeddings.json
        - Step 6 needs no Java/MOWL; rerun Step 6 after every Step 5 run
    - practice generated/candidate_el.ttl (6 axioms) until Step 3 is done (examples were chosen from the cc module)
    - per candidate:
        - skip duplicates + missing vectors (log why)
        - cosine: plain = cos(X, Y); "some" = cos(X + p, Z)
        - LLM plausibility 0-1 (JSON schema, cached)
        - hybrid = 0.7 cos + 0.3 llm; keep if >= τ (0.60)
    - math: cosine must be >= 0.43 even with LLM = 1.0
    - LLM contribution = accepted with cosine < τ
    - outputs: accepted_el.ttl, reports/hybrid_scores.csv, "hybrid" in mowl_metrics.json
    - practice result: 1 accepted (Legal Name -> Designative Name)

step 7 
generate candidate llm.
when running an erroo occured re: json so I needed to go back tot he llm helper to update to support 
when the candidates were created there were no labels
ran the fulll run and the result was 21
how to determine what was new versus a grandparent (MOWL training identified grnadparents in its result set and they were added to the train file.)
    skip-level: a grandparent or higher ancestor. It's true, but the module already implies it.
    new branch: a class from another part of the tree. what I am testing for and where the llm will add more value

7 - 
Robot for merging and reasoning java file (robot.jar). code for the merging and reasoning 
        java -jar tools/robot.jar merge --input src/cco-module.ttl --input generated/accepted_el.ttl \
  reason --reasoner ELK --remove-redundant-subclass-axioms false \
  --output src/module_augmented.ttl

check the file module augment file:  There are 10 of 10 accepted axioms

check the version of robot and create a folder for tools - Robot

create the run_all.py file
single script to create the full pipeline 
    organizae my scriopts with inputs and outputs and then final run by robot with a print out of the output.
   reference the readme file for the order of the files
   set up the code and run the check -- got an error.  did research and there was a LLM error due to token size.  Update the LLM helper to chnage the max size.  the long defintions require more tokens

need to reference robot from here so it can run on its own

Ran and received an errro on elk consistency.  this is because some of the accepted axioms have a class showing up under two classes that are disjont --took out 16 disjoint pairs.







 -- Update update my gitignore file to ingnore because the rull all file will download again
MOWL - machine learning over OWL.
1. DONE: pick a CCO ntology module
   DONE: extract the definitions (IRI,Label and definition) (extract definitions)
2. DONE: Use LLM to normalize and enrich the definitions ()
    For each class - look at the parent and review definitions 
    standardized, canonical format (NEED TO DEFINE)
    ambuiguity is removed, abbreviation spelled out- N/A, terminology is aligned to CCO Style (WHAT IS CCO STYLE?)
3. DONE - Only for classes: take enriched definition and create OWL 2 EL compliant candidacte axiom. https://www.w3.org/TR/owl2-profiles/#OWL_2_EL
4. DONE -- create a train and valid ttl file from CCO_MOdule  keeping ALL classes and properites (train.ttl, valid.ttl)
5. DONE write a script to train an embedding model, evaluate and record results.  use cosine similarity for held out sub/super class paris (generate candidate)
6. score each super/sub pair by MOWL cosine similarity.  query an LLM to rate semantic plausibiloty.  combine scores to create a weighted avaer and return list of axioms meeting threshold.
7. DONE take was waht created (Candidate_el) and accepted (accepted_el) and merge with CCO-module to an augment ontology and use robot and use 'elk' reasoner. (Validate ontology quality and consistency using ROBOT, ELK, and OOPS!.) WHAT IS OOPS? OWL EL

DONE create a script (run_all.py)to execute the full pipeline (above)     and the output will be a report to summarize (mean_cos, chosen T, accepted axiom count, LLM contribution rate and consistency status)
   
    Are there standards for the above metrics?
    Grading: mean_cos ≥ 0.70, ontology consistent, and at least several new axioms added automatically and must run under one command.
    Required components:
    # Core
mowl==1.0.2
JPype1==1.6.0
torch>=2.2,<3
rdflib==7.0.0
pandas==2.2.2

Delvierables:
- DONE - Pick module and download
- DONE - 'data/definitions.csv` — Extracted SKOS definitions (columns: IRI, label, definition).
- DONE - `data/definitions_enriched.csv` — LLM-normalized/enriched definitions used for axiom generation.
- - DONE -`generated/candidate_el.ttl` — OWL 2 EL candidate axioms produced via LLM + pattern prompts.
- done  `generated/accepted_el.ttl` — Candidates retained after hybrid MOWL cosine + LLM plausibility filtering.
- - DONE -`reports/mowl_metrics.json` — Training/evaluation metrics (mean cosine, selected τ, and hybrid settings).
- - DONE -`src/cco-module.ttl` — Original selected CCO module (input ontology).
- - DONE -`src/train.ttl` — Training split of the ontology (≈ 80% of axioms).
- - DONE -`src/valid.ttl` — Validation split of the ontology (≈ 20% of axioms).
- DONE`src/module_augmented.ttl` — Final ontology after merging accepted axioms and reasoning with ELK.
- - DONE -`scripts/extract_definitions.py` — Extracts SKOS definitions into `data/definitions.csv`.
- DONE `scripts/preprocess_definitions_llm.py` — Normalizes/enriches definitions with an LLM.
- DONE `scripts/generate_candidates_llm.py` — Generates EL-compliant candidate axioms from enriched definitions.
- - DONE -`scripts/train_mowl.py` — Trains ELEmbeddings on `src/train.ttl` and evaluates on `src/valid.ttl`.
-- DONE - `scripts/filter_candidates_hybrid.py` — Combines MOWL cosine and LLM plausibility to keep axioms.
- DONE`scripts/run_all.py` — One-command driver that executes the entire pipeline end-to-end.       