# --- Project 5, create the train (~80%) and valid (~20%) ttl files 
# Input:  src/cco-module.ttl
# Output: src/train.ttl   the whole module MINUS the held-out 'subClassOf' axioms
#         src/valid.ttl   all declarations PLUS the held-out 'subClassOf' axioms
#
# Sections 2 & 3 reuse documentation/embedding_example.py
# ( 
#
#   python scripts/split_module.py

import random, rdflib, networkx as nx
from pathlib import Path
from rdflib import URIRef
from rdflib.namespace import RDF, RDFS, OWL

random.seed(42)                                     # same as embedding_example.py

BASE = Path(__file__).resolve().parent.parent       # the assignment folder
TTL_IN = BASE / "src" / "cco-module.ttl"
TRAIN_OUT = BASE / "src" / "train.ttl"
VALID_OUT = BASE / "src" / "valid.ttl"

# Load cco-module 
g = rdflib.Graph(); # create the empty graph
g.parse(TTL_IN, format="turtle") #load the empty graph with the coo module data

# Parse and extract classes + rdfs:subClassOf edges   same as embedding_example.py
def iri(u):  #if U is an IRI return the IRI; if blank return None. 
    return str(u) if isinstance(u, URIRef) else None 

classes = set() #Unique Named Classes
for s, p, o in g.triples((None, RDF.type, OWL.Class)):  #find every triple that matches the ?s rdf:type OWL.class pattern
    u = iri(s) #only keep the subject is it has an IRI (ignore the none values)
    if u: classes.add(u)

edges = []
for s, p, o in g.triples((None, RDFS.subClassOf, None)):  #matches the ?s subclassOf ?o pattern
    u, v = iri(s), iri(o) #convert both sides - u is the child and v is the parent
    if u and v:   #only if both are named)
        classes.add(u); classes.add(v) #add the ends
        edges.append((u, v))  #store the child,parent as a pair

edges = [(u, v) for (u, v) in edges if u in classes and v in classes] #only edges whose ends are BOTH in classes.
edges = sorted(set(edges))   #remove duplicate edges and put in alphabetical order sort so the seed gives the same split every run
print(f"Classes: {len(classes):,}  SubClassOf edges: {len(edges):,}")  #print the result set

# Connectivity-preserving split
#   1st pass  80/20 (train/valid); no test set in Project 5
UG_full = nx.Graph(); UG_full.add_nodes_from(classes); UG_full.add_edges_from(edges) #UG - undirected graph and using networkx
target_train = int(0.8 * len(edges))
target_valid = len(edges) - target_train

def can_remove_edge(edge_list, a, b):               # reuse code from embedding_example.py
    deg = {} #empty dictionary
    for x, y in edge_list:  #for each edge add '1' to both ends
        deg[x] = deg.get(x, 0) + 1
        deg[y] = deg.get(y, 0) + 1
    return deg.get(a,0) > 1 and deg.get(b,0) > 1 #returns true if greater than 1 (both ends still have at least 2 edges after removal)

train_set = set(edges)
deg_full = dict(UG_full.degree()) #counts the edges, per class, in the full graph and puts it into a dictionary
candidates = [(u,v) for (u,v) in edges if deg_full.get(u,0) >= 2 and deg_full.get(v,0) >= 2]  #filter for classes with 2 or more edges
random.shuffle(candidates)

valid_edges = [] #first pass
for (u,v) in candidates:
    if len(valid_edges) < target_valid and (u,v) in train_set: #no more links once the train set is met 
        tmp = list(train_set); tmp.remove((u,v)) #after removal what is left
        if can_remove_edge(tmp, u, v): train_set.remove((u,v)); valid_edges.append((u,v)) #any removal can impact the counts for future loops
    if len(valid_edges) >= target_valid:
        break #stop once the train set percentage is met (target)

train_edges = sorted(train_set)
valid_edges = sorted(valid_edges)
print(f"Split sizes: train={len(train_edges)}, valid={len(valid_edges)}")
print(f"  (target valid={target_valid}; the connectivity rule allowed {len(valid_edges)})")

# write the split as two Turtle files for MOWL (training and valid)

#  BFO parents are used in the module but not declared .
#     Make a declaration (Class rdf:type owl:Class) for each one.
missing_declarations = [] #empty list to hold the missing declarations
for c in sorted(classes): #loop through the classes and check , in alphabetical order
    triple = (URIRef(c), RDF.type, OWL.Class) #standard triple for a class declaration
    if triple not in g: #check to see if the triple exists. 
        missing_declarations.append(triple) #if not exists, add it to the missing_declarations list

#  train.ttl = the whole module, minus the held-out edges, plus the missing BFO declarations
train_g = rdflib.Graph() #create the train graph
train_g.parse(TTL_IN, format="turtle")          # a fresh copy of the module
for (u, v) in valid_edges: #remove the held-out edges from the train graph
    train_g.remove((URIRef(u), RDFS.subClassOf, URIRef(v)))
for triple in missing_declarations: #add the missing declarations to the train graph
    train_g.add(triple)
train_g.remove((None, OWL.imports, None)) #drop owl.imports to only use this module
 # allow the held out child linked to teh grandparent relationship
 # (child -> parent is heald out, but keep child -> grandparent)
grandparent_links = 0 #reconnet the children to the grandparent
for (u, v) in valid_edges:  #loop through the held out pairs of child/parent
    for gp in train_g.objects(URIRef(v), RDFS.subClassOf): #find the grandparent
        if isinstance(gp, URIRef):  #skip any blank nodes
            train_g.add((URIRef(u), RDFS.subClassOf, gp)) #add the child subclass of gp.
            grandparent_links +=1
print("Grandparent links added to train", grandparent_links)

#  valid.ttl = every declaration (rdf:type triple for a named entity), plus the missing BFO declarations, plus the held-out edges
valid_g = rdflib.Graph() #create the valid graph
for prefix, namespace in g.namespaces():        # keep the same prefixes (cco:, obo: ...)
    valid_g.bind(prefix, namespace) #clean prefixes
for s, p, o in g.triples((None, RDF.type, None)):
    if isinstance(s, URIRef):                   # skip blank nodes (restrictions)
        valid_g.add((s, p, o))
for triple in missing_declarations:
    valid_g.add(triple) #missing triples are added to the valid graph to ensure that all upper level parent classes are properly declared
for (u, v) in valid_edges:
    valid_g.add((URIRef(u), RDFS.subClassOf, URIRef(v)))

train_g.serialize(TRAIN_OUT, format="turtle")
valid_g.serialize(VALID_OUT, format="turtle")

#  No held-out edge may still be in train
overlap = 0
for (u, v) in valid_edges:
    if (URIRef(u), RDFS.subClassOf, URIRef(v)) in train_g:
        overlap += 1
print("Check - held-out edges still in train (should be 0):", overlap)

#  Train + valid should add up to all edges
print("Check - train + valid = all edges:", len(train_edges) + len(valid_edges) == len(edges))

#  Every class in a held-out edge must still have an edge in train
train_classes = set()
for (u, v) in train_edges:
    train_classes.add(u)
    train_classes.add(v)
isolated = 0
for (u, v) in valid_edges:
    if u not in train_classes or v not in train_classes:
        isolated += 1
print("Check - held-out edges with a class missing from train (should be 0):", isolated)

#  Both files should declare the same number of named classes
train_class_count = 0
for c in set(train_g.subjects(RDF.type, OWL.Class)):
    if isinstance(c, URIRef):                   # count named classes only
        train_class_count += 1
valid_class_count = 0
for c in set(valid_g.subjects(RDF.type, OWL.Class)):
    if isinstance(c, URIRef):
        valid_class_count += 1
print(f"Check - classes declared: train={train_class_count}, valid={valid_class_count}")

print(f"Saved: {TRAIN_OUT}")
print(f"Saved: {VALID_OUT}")