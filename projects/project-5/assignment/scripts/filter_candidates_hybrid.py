# --- Project 5 hybrid filter (MOWL cosine + LLM plausibility) 
# For each candidate (sub, super) pair:
#   cosine = how close MOWL puts the two classes      (0 to 1)
#   llm    = how plausible the LLM thinks the axiom is (0 to 1)
#   hybrid = 0.7 * cosine + 0.3 * llm
#   keep the axiom if hybrid >= τ  
#
# Input:  generated/candidate_el.ttl   (Step 3)
#         data/mowl_embeddings.json    (Step 5 vectors)
#         reports/mowl_metrics.json    (Step 5 τ)
#         src/cco-module.ttl           (labels and definitions)
# Output: generated/accepted_el.ttl    (the axioms we keep)
#         reports/mowl_metrics.json    (adds a "hybrid" section)
#
# Reuse: 
#   embedding_example.py            cosine on (child, parent) pairs and save report dictionary as JSON
#   augment_w_definitions.ipynb       read skos:definition for each class
#   preprocess_definitions_llm.py      ask_llm with a JSON schema
#
# Run from assignment folder: python scripts/filter_candidates_hybrid.py

# 1) Imports
import json
from pathlib import Path
import numpy as np #support for multidimensional arrays and mathematical operations, used here for calculating cosine similarity between class vectors that a simple list does not provide. Numpy is a powerful library for numerical computing in Python, and it provides efficient array operations and mathematical functions that are essential for tasks like vector calculations.
import rdflib
from rdflib import RDFS, URIRef, OWL
from rdflib.namespace import SKOS
from llm_helper import ask_llm, MODEL 

# 2) Paths and settings
BASE = Path(__file__).resolve().parent.parent      # the assignment folder
CANDIDATES = BASE / "generated" / "candidate_el.ttl"
ACCEPTED = BASE / "generated" / "accepted_el.ttl"
MODULE = BASE / "src" / "cco-module.ttl"
VECTORS = BASE / "data" / "mowl_embeddings.json"
METRICS = BASE / "reports" / "mowl_metrics.json"

COSINE_WEIGHT = 0.7 #the cosine similarity weight in the hybrid score calculation, which determines how much influence the MOWL cosine similarity has on the final decision to keep or drop a candidate axiom. A higher weight means that the cosine similarity will have a greater impact on the hybrid score, while a lower weight means that the LLM plausibility will have more influence.
LLM_WEIGHT = 0.3 #the LLM plausibility weight in the hybrid score calculation, which determines how much influence the LLM's assessment of plausibility has on the final decision to keep or drop a candidate axiom. A higher weight means that the LLM's plausibility score will have a greater impact on the hybrid score, while a lower weight means that the MOWL cosine similarity will have more influence.

# 3) Read τ (Step 5) and the class vectors 
metrics = json.loads(METRICS.read_text())
TAU = metrics["tau"]
print("τ from Step 5:", TAU)

vectors = json.loads(VECTORS.read_text())["classes"]   # {class IRI: [50 numbers]}

# 4) Read labels and definitions from the module (for the LLM prompt)
module = rdflib.Graph()
module.parse(MODULE, format="turtle")

label_map = {}
for s, o in module.subject_objects(RDFS.label):
    label_map[str(s)] = str(o)
# BFO parents the module uses but does not label (same as Step 2)
label_map["http://purl.obolibrary.org/obo/BFO_0000030"] = "object"
label_map["http://purl.obolibrary.org/obo/BFO_0000019"] = "quality"
label_map["http://purl.obolibrary.org/obo/BFO_0000031"] = "generically dependent continuant"

definitions = {}                                   # like get_definition() in augment_w_definitions
for s, o in module.subject_objects(SKOS.definition):
    definitions[str(s)] = str(o)


##ADD BECASUE OF ACCPETED AXIMONS BEIBG IN DISJOINT CLASSES:
# 4b) Disjoint classes: no instance can be in both classes of a pair
disjoint_pairs = set()
for a, b in module.subject_objects(OWL.disjointWith):
    disjoint_pairs.add((str(a), str(b)))
    disjoint_pairs.add((str(b), str(a)))           # disjointness works both ways

# BFO says these three are disjoint. BFO is imported, so it's not in the module file.
GDC = "http://purl.obolibrary.org/obo/BFO_0000031"
OBJECT = "http://purl.obolibrary.org/obo/BFO_0000030"
QUALITY = "http://purl.obolibrary.org/obo/BFO_0000019"
for a, b in [(GDC, OBJECT), (GDC, QUALITY), (OBJECT, QUALITY)]:
    disjoint_pairs.add((a, b))
    disjoint_pairs.add((b, a))

print("Disjoint pairs to check:", len(disjoint_pairs) // 2)

tree = rdflib.Graph()                              # the module's subclass tree ...
for s, o in module.subject_objects(RDFS.subClassOf):
    tree.add((s, RDFS.subClassOf, o))              # ... plus each axiom we accept (6d)    

# 5) What we ask the LLM (the same for every candidate)
INSTRUCTIONS = """You are an ontology reviewer for the Common Core Ontologies (CCO), which extend BFO.
Rate how plausible the proposed subclass axiom is, given the class definitions.
Return a number from 0 to 1:
  1.0 = clearly true for every instance of the subclass
  0.5 = unsure or only sometimes true
  0.0 = clearly false
Keep the reason to one short sentence.
Return JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "plausibility": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["plausibility", "reason"],
    "additionalProperties": False,
}

# 6) Read the candidates and score each (sub, super) pair
candidates = rdflib.Graph()
candidates.parse(CANDIDATES, format="turtle")

accepted = rdflib.Graph()                          # the axioms we keep
accepted.bind("cco", "https://www.commoncoreontologies.org/")

total = 0                                          # counters for the report
scored = 0
kept = 0
llm_helped = 0
clashed = 0

for sub, sup in candidates.subject_objects(RDFS.subClassOf):
    total += 1
    sub_name = label_map.get(str(sub), str(sub))
    sup_name = label_map.get(str(sup), str(sup))
    axiom = sub_name + " SubClassOf " + sup_name

    # 6a) Skip what we cannot score
    if not isinstance(sup, URIRef):                # [ owl:onProperty ... ] restriction skip blank nodes, which are not URIRefs and cannot be scored by MOWL or LLM
        print("  skipped (not a (sub, super) pair):", sub_name)
        continue
    if (sub, RDFS.subClassOf, sup) in module: #skip axioms already in the module, as they have already been accepted and do not need to be re-evaluated. This prevents unnecessary scoring and ensures that we only evaluate new candidate axioms that are not already part of the ontology.
        print("  skipped (already in module):", axiom)
        continue
        # every class above sub (and sub itself), every class above sup (and sup itself)
    sub_up = [str(c) for c in tree.transitive_objects(sub, RDFS.subClassOf)] #childs current lineage
    sup_up = [str(c) for c in tree.transitive_objects(sup, RDFS.subClassOf)] #proposed parents lineage
    clash = None
    for a in sub_up:
        for b in sup_up: #check every possible combination of the subclass's lineage and the proposed superclass's lineage to see if any pair of classes are disjoint. If a disjoint pair is found, it indicates that the proposed axiom would create a logical inconsistency in the ontology, as it would imply that an instance could belong to two disjoint classes. This check helps maintain the integrity of the ontology by preventing the acceptance of axioms that would violate disjointness constraints.
            if (a, b) in disjoint_pairs:
                clash = label_map.get(a, a) + " disjointWith " + label_map.get(b, b)
    if clash is not None:                          # sub would be under two disjoint classes. log the clash and print a message indicating that the axiom is being skipped due to the clash with the disjoint classes. This prevents the acceptance of axioms that would create logical inconsistencies in the ontology.
        print("  skipped (clashes with " + clash + "):", axiom)
        clashed += 1 #count how many axioms were skipped due to clashes with disjoint classes.
        continue #skip the candidate  - do not score or accept it 
    if str(sub) not in vectors or str(sup) not in vectors:
        print("  skipped (no MOWL vector):", axiom)
        continue

    # 6b) MOWL cosine, using NUMPY, as in embedding_example.py section 9
    #close to 1 means the two classes are close in the embedding space, which suggests that they are semantically similar and that the proposed subclass-superclass relationship is likely to be valid. A cosine similarity close to 0 indicates that the classes are orthogonal or unrelated in the embedding space, suggesting that the proposed relationship may not be appropriate. A negative cosine similarity would indicate that the classes are diametrically opposed in the embedding space, which would strongly suggest that the proposed relationship is invalid. In this context, we expect values between 0 and 1, as we are interested in how closely related the subclass and superclass are based on their embeddings.
    #close to -1 means the two classes are diametrically opposed in the embedding space, which would strongly suggest that the proposed relationship is invalid. In this context, we expect values between 0 and 1, as we are interested in how closely related the subclass and superclass are based on their embeddings.
    #clsoe to 0 indicates that the classes are orthogonal or unrelated in the embedding space, suggesting that the proposed relationship may not be appropriate. In this context, we expect values between 0 and 1, as we are interested in how closely related the subclass and superclass are based on their embeddings.

    a = np.array(vectors[str(sub)])
    b = np.array(vectors[str(sup)])
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) #cosine similarity calculation between the subclass and superclass vectors, which measures how similar the two classes are in the embedding space. The cosine similarity is computed as the dot product of the two vectors divided by the product of their magnitudes (norms). This value ranges from -1 to 1, where 1 indicates that the vectors are identical (high similarity), 0 indicates no similarity, and -1 indicates that they are diametrically opposed. In this context, we expect values between 0 and 1, as we are interested in how closely related the subclass and superclass are based on their embeddings.


    # 6c) LLM plausibility, same ask_llm as Step 2
    prompt = "Proposed axiom: " + axiom + "\n" #candidate axiom is included in the prompt to provide context for the LLM, allowing it to evaluate the plausibility of the proposed subclass-superclass relationship based on the definitions of the classes involved. This helps the LLM generate a more informed and accurate assessment of the axiom's validity.
    prompt += "Definition of " + sub_name + ": " + definitions.get(str(sub), "none") + "\n"
    prompt += "Definition of " + sup_name + ": " + definitions.get(str(sup), "none") + "\n"
    answer = ask_llm(INSTRUCTIONS, prompt, SCHEMA, max_tokens=500) #call to the ask_llm function to get the LLM's assessment of the plausibility of the proposed axiom   
    llm = float(answer["plausibility"])
    llm = min(1.0, max(0.0, llm))                  # keep it in 0..1 (clamp, as in embedding_example.py)

#LLM Plausibility supports the capture of nuanced judgments, based on the defiintions/comments about the proposed axiom's validity, which may not be fully captured by a sparse ontology. 
# By combining the LLM's assessment with MOWL cosine similarity in a weighted manner, we can create a more robust and comprehensive evaluation of the candidate axiom, taking into account both the semantic similarity of the classes and the LLM's reasoning based on their definitions.

    # 6d) Hybrid score and decision
    hybrid = COSINE_WEIGHT * cos + LLM_WEIGHT * llm
    scored += 1
    if hybrid >= TAU: #acceptance threshold for the hybrid score. 
        print(f"  KEEP  hybrid={hybrid:.3f}  cos={cos:.3f}  llm={llm:.2f}  {axiom}")
        accepted.add((sub, RDFS.subClassOf, sup))
        tree.add((sub, RDFS.subClassOf, sup))      # later pairs are checked against it too
        kept += 1
        if cos < TAU:                              # cosine alone would have dropped it
            llm_helped += 1 #determines the LLM's contribution, even though the MOWL cosine similarity alone would have rejected them. This metric helps quantify the added value of incorporating the LLM's assessment into the hybrid scoring process, demonstrating how the LLM can help identify plausible axioms that may not be captured by the MOWL cosine similarity alone.
    else:
        print(f"  drop  hybrid={hybrid:.3f}  cos={cos:.3f}  llm={llm:.2f}  {axiom}")
    print("        LLM reason:", answer["reason"])

# 7) Save the accepted axioms
accepted.serialize(ACCEPTED, format="turtle")

# 8) Add the hybrid results to mowl_metrics.json (report format from section 13 of embedding_example.py)
if kept > 0:
    contribution = llm_helped / kept
else:
    contribution = 0.0                             # nothing kept: avoid dividing by zero

metrics["hybrid"] = {
    "cosine_weight": COSINE_WEIGHT,
    "llm_weight": LLM_WEIGHT,
    "tau": TAU,
    "llm_model": MODEL,
    "candidates": total,
    "scored": scored,
    "accepted": kept,
    "skipped_disjoint": clashed,
    "llm_contribution_rate": round(contribution, 3),
    "llm_contribution_rule": "share of accepted axioms whose cosine alone is below tau",
}
METRICS.write_text(json.dumps(metrics, indent=2))

print(f"\nCandidates: {total}  scored: {scored}  accepted: {kept}  skipped (disjoint): {clashed}")
print(f"LLM contribution rate: {contribution:.1%}")
print("Saved:", ACCEPTED)
print("Saved:", METRICS)