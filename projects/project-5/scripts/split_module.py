"""Step 4: split src/cco-module.ttl into src/train.ttl (~80%) and src/valid.ttl (~20%).

Uses the instructor's connectivity-preserving rule (documentation/embedding_example.py): a named
SubClassOf edge may be held out only if both endpoints keep degree >= 1 in the remaining training edges.
train.ttl = the whole module minus the held-out edges (so every declaration, label and definition stays).
valid.ttl = every class/property declaration plus the held-out edges.
owl:imports are dropped from both splits so they are self-contained training/evaluation inputs."""
import random
import sys
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

sys.path.insert(0, str(Path(__file__).parent))
from common import MODULE, SRC  # noqa: E402

SEED, VALID_FRAC = 42, 0.20


def main():
    g = Graph().parse(MODULE)
    g.remove((None, OWL.imports, None))
    edges = sorted((s, o) for s, o in g.subject_objects(RDFS.subClassOf) if isinstance(s, URIRef) and isinstance(o, URIRef))
    target = round(VALID_FRAC * len(edges))
    deg = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1; deg[v] = deg.get(v, 0) + 1
    cand = [e for e in edges if deg[e[0]] >= 2 and deg[e[1]] >= 2]
    random.Random(SEED).shuffle(cand)
    held = []
    for u, v in cand:
        if len(held) >= target:
            break
        if deg[u] > 1 and deg[v] > 1:
            held.append((u, v)); deg[u] -= 1; deg[v] -= 1

    train = Graph(); valid = Graph()
    for pfx, ns in g.namespaces():
        train.bind(pfx, ns); valid.bind(pfx, ns)
    for t in g:
        if not (t[1] == RDFS.subClassOf and (t[0], t[2]) in set(held)):
            train.add(t)
    decl_types = (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty, OWL.Ontology)
    for t in decl_types:
        for s in g.subjects(RDF.type, t):
            valid.add((s, RDF.type, t))
    for u, v in held:
        valid.add((u, RDFS.subClassOf, v)); valid.add((v, RDF.type, OWL.Class))
    train.serialize(SRC / "train.ttl", format="turtle")
    valid.serialize(SRC / "valid.ttl", format="turtle")
    n_train = len(edges) - len(held)
    print(f"split_module: {len(edges)} SubClassOf edges -> train {n_train} ({n_train / len(edges):.0%}), "
          f"valid {len(held)} ({len(held) / len(edges):.0%}); declarations kept in both")


if __name__ == "__main__":
    main()
