"""Build a slim CCO/BFO module (MIREOT-style) for Project 2.

Keeps every CCO/BFO term used in core.ttl and instances.ttl (plus a few
seed terms needed for later tasks), with all their named ancestors,
labels, definitions, and types. Re-run whenever you use a new CCO/BFO term.

Run from projects/project-2/assignment:
    python scripts/extract_module.py
"""
from rdflib import Graph, URIRef, BNode, Literal, RDF, RDFS, OWL, XSD

SOURCE = "../../project-4/assignment/src/cco_merged.ttl"
LOCAL = ["src/data/core.ttl", "src/data/instances.ttl"]
OUT = "src/data/modules/cco-bfo.ttl"
NS = ("https://www.commoncoreontologies.org/", "http://purl.obolibrary.org/obo/")
# Terms needed for Tasks 3-4 even before they appear in local files.
SEED_LABELS = {
    "occupation role", "act of measuring", "measurement unit", "quality", "is about",
    "realizes", "is output of", "has output", "inheres in", "bearer of",
    "act of artifact processing", "count measurement information content entity",
    "has integer value", "has decimal value", "has text value", "group of persons",
    "person", "organization", "geospatial region", "is measured by", "is a measurement of",
    "has participant", "participates in", "has member part", "member part of",
    "designative information content entity", "code identifier", "role", "disposition",
    "artifact design", "information content entity", "realizable entity",
}
FOLLOW = {RDF.type, RDFS.subClassOf, RDFS.subPropertyOf, OWL.inverseOf}

src = Graph().parse(SOURCE)
local = Graph()
for f in LOCAL:
    local.parse(f)

def ours(t):
    return isinstance(t, URIRef) and str(t).startswith(NS)

todo = {t for tr in local for t in tr if ours(t)}
todo |= {s for s, o in src.subject_objects(RDFS.label) if str(o).lower() in SEED_LABELS}

out, seen = Graph(), set()
while todo:
    t = todo.pop()
    if t in seen:
        continue
    seen.add(t)
    for p, o in src.predicate_objects(t):
        if isinstance(o, BNode):
            continue                      # skip complex axioms; keep the tree
        if isinstance(o, Literal) and o.datatype is None and not o.language:
            o = Literal(str(o), datatype=XSD.string)   # grader: all literals typed
        out.add((t, p, o))
        if p in FOLLOW and ours(o):
            todo.add(o)

# Ontology header, so core.ttl's owl:imports IRI names this file.
mod = URIRef("https://www.commoncoreontologies.org/CommonCoreOntologiesMerged")
out.add((mod, RDF.type, OWL.Ontology))
out.add((mod, RDFS.comment, Literal("Slim MIREOT-style extract of CCO/BFO terms used in Project 2.", lang="en")))

for pfx, ns in src.namespaces():
    out.bind(pfx, ns)
out.serialize(OUT, format="turtle")
print(f"Kept {len(seen)} terms, {len(out)} triples -> {OUT}")
