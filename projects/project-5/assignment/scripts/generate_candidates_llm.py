# Project 5 generate the candidate EL axioms with an LLM 
# Feach class, the LLM will read the enriched definition (definitions_enriched.csv) and pick
# other classes from the module that the class is also a kind of.
# Each become a simple EL axiom:   X rdfs:subClassOf Y
#
# Input:  data/definitions_enriched.csv   
#         src/cco-module.ttl              (class labels and IRIs)
# Output: generated/candidate_el.ttl      candidates )
#
# only plain subclass axioms :
#   - they are always OWL 2 EL
#   - scores (sub, super) pairs
#
# Run from  assignment folder:
#   python scripts/generate_candidates_llm.py      (all classes)
#   python scripts/generate_candidates_llm.py 5    (test: first 5 classes)

#  Imports
import sys
from pathlib import Path
import pandas as pd
import rdflib
from rdflib import RDF, RDFS, OWL, URIRef
from llm_helper import ask_llm, MODEL

#   Paths and settings
BASE = Path(__file__).resolve().parent.parent      #  assignment folder
CSV_IN = BASE / "data" / "definitions_enriched.csv" #input file with enriched definitions
MODULE = BASE / "src" / "cco-module.ttl" #input file with the CCO module in Turtle format
CANDIDATES = BASE / "generated" / "candidate_el.ttl" # output file with candidate EL axioms in Turtle format

MAX_PICKS = 3                                      # at most, 3, new superclasses per class to keep the candidate list small

# Optional test limit from the command line, e.g. "5"  
if len(sys.argv) > 1:  #read from the command line
    LIMIT = int(sys.argv[1])
else:
    LIMIT = None

#   Read the CCO module file: labels and list of named classes
module = rdflib.Graph()
module.parse(MODULE, format="turtle")

label_map = {}                                     # IRI -> label 
for s, o in module.subject_objects(RDFS.label):
    label_map[str(s)] = str(o)
label_map["http://purl.obolibrary.org/obo/BFO_0000030"] = "object"
label_map["http://purl.obolibrary.org/obo/BFO_0000019"] = "quality"
label_map["http://purl.obolibrary.org/obo/BFO_0000031"] = "generically dependent continuant"

iri_map = {}                                       # label -> IRI, the other way round (reverse)
for s in module.subjects(RDF.type, OWL.Class):     # named classes, as in split_module.py
    if isinstance(s, URIRef) and str(s) in label_map:
        iri_map[label_map[str(s)].lower()] = str(s) #lower case for consistency 

class_list = sorted(iri_map.keys())                # what the LLM may choose from (labels)
print("Classes the LLM may choose from:", len(class_list))

#  What the LLM is asked and will be the same for every class
INSTRUCTIONS = """You are an ontology editor for the Common Core Ontologies (CCO), which extends the BFO.
You get one class, its definition, its current parent, and a list of class labels.
Pick up to """ + str(MAX_PICKS) + """ classes from the list that EVERY instance of the class, subclass, is also an instance of,based on the definition.
Rules:
- Use labels exactly as written in the list.
- Do not pick the class itself or its current parent.
- If no class fits, return an empty list.
Return JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "superclasses": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["superclasses"],
    "additionalProperties": False,
}
# 5) Ask the LLM about each class
df = pd.read_csv(CSV_IN)

candidates = rdflib.Graph()
candidates.bind("cco", "https://www.commoncoreontologies.org/")

lines = []                                  #text lines for the file
asked = 0                                          # counters for the summary
rejected = 0

for _, row in df.iterrows():
    if row["entity_type"] != "class":              # skip object properties
        continue
    if LIMIT is not None and asked >= LIMIT:
        break

    sub = row["IRI"]
    prompt = "Class: " + str(row["label"]) + "\n"
    prompt += "Definition: " + str(row["enriched_definition"]) + "\n"
    prompt += "Current parent: " + str(row["parent_label"]) + "\n"
    prompt += "Class list:\n" + "\n".join(class_list) #one label per line

    answer = ask_llm(INSTRUCTIONS, prompt, SCHEMA)
    asked += 1

    # 5a) convert the  picked labels into axioms, after checking it
    for pick in answer["superclasses"][:MAX_PICKS]:
        key = pick.strip().lower()
        if key not in iri_map:                     # check is the LLM made up the label
            print("  rejected (not in module):", row["label"], "->", pick)
            rejected += 1 #counter for rejected items not in the module
            continue 
        sup = iri_map[key]  #superclass IRI  
        if sup == sub:                             # SUp and Sub IRI are the same (the class itself)
            rejected += 1
            continue
        if (URIRef(sub), RDFS.subClassOf, URIRef(sup)) in module: #do not allow duplicates from the module itself
            print("  rejected (already in module):", row["label"], "->", pick)
            rejected += 1
            continue
        if (URIRef(sub), RDFS.subClassOf, URIRef(sup)) in candidates:
            continue                               # the LLM picked it twice
        candidates.add((URIRef(sub), RDFS.subClassOf, URIRef(sup)))  #save the set of candidates
        print("  candidate:", row["label"], "SubClassOf", label_map[sup])

        # 5b) Keep a readable copy: a label comment, then the axiom
        lines.append("# " + str(row["label"]) + "  SubClassOf  " + label_map[sup])
        lines.append("<" + sub + "> rdfs:subClassOf <" + sup + "> .")
        lines.append("")

        ##candidates.add((URIRef(sub), RDFS.subClassOf, URIRef(sup))) remove to support new code for labels
        ##print("  candidate:", row["label"], "SubClassOf", label_map[sup])

# 6) Save the candidates as Turtle, with a label comment above each axiom
#    ##Note from research is that rdflib cannot write comments, so we write the lines ourselves)  
CANDIDATES.parent.mkdir(exist_ok=True)
header = [
    "# candidate_el.ttl -- written by scripts/generate_candidates_llm.py (Step 3)",
    "# Each comment line gives the labels for the axiom below it.",
    "",
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
    "",
]
CANDIDATES.write_text("\n".join(header + lines) + "\n", encoding="utf-8")

# 6a) Direct check: read the file back and count the axioms
check = rdflib.Graph()
check.parse(CANDIDATES, format="turtle")
print("Read back from file:", len(check), "axioms (should match Candidates below)")
