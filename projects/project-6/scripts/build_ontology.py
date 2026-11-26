"""Stage 0: build ontology/p6_extension_base.ttl from the approved spec and a slim CCO extract (ROBOT BOT module)."""
import sys
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

sys.path.insert(0, str(Path(__file__).parent))
from common import BASE_EXT, GEN, IMPORTS, P6, SLIM, expand, load_spec, robot  # noqa: E402

# every CCO/BFO term the patterns, ETL, shapes and queries use (seed for the slim module)
CCO_TERMS = """ont00000108 ont00000253 ont00000853 ont00000965 ont00000487 ont00000003 ont00000077 ont00000923 ont00000829
ont00000827 ont00001180 ont00001916 ont00001808 ont00001908 ont00001765 ont00001766 ont00001764 ont00001767 ont00001773
ont00001772 ont00000890 ont00001942 ont00000226 ont00000192 ont00000958 ont00000686 ont00000228""".split()
BFO_TERMS = "BFO_0000171 BFO_0000101 BFO_0000176 BFO_0000178 BFO_0000084 BFO_0000029 BFO_0000031 BFO_0000004".split()


def main():
    spec = load_spec()
    g = Graph()
    g.bind("p6", Namespace(P6)); g.bind("cco", Namespace("https://www.commoncoreontologies.org/")); g.bind("obo", Namespace("http://purl.obolibrary.org/obo/"))
    ont = URIRef(P6 + "extension")
    g.add((ont, RDF.type, OWL.Ontology))
    g.add((ont, RDFS.comment, Literal("Project 6 GTFS rail-schedule extension of CCO. Base = approved spec; see p6_extension.ttl for reasoner-accepted LLM axioms.")))
    g.add((ont, OWL.imports, URIRef(P6 + "cco_slim")))
    for c, d in spec["local_classes"].items():
        u = URIRef(expand(c))
        g.add((u, RDF.type, OWL.Class)); g.add((u, RDFS.label, Literal(d["label"], lang="en")))
        g.add((u, SKOS.definition, Literal(d["definition"], lang="en"))); g.add((u, RDFS.subClassOf, URIRef(expand(d["parent"]))))
    for p, d in spec["local_properties"].items():
        u = URIRef(expand(p))
        g.add((u, RDF.type, OWL.DatatypeProperty)); g.add((u, RDFS.subPropertyOf, URIRef(expand(d["parent"]))))
        g.add((u, RDFS.label, Literal(p.split(":")[1], lang="en"))); g.add((u, RDFS.range, URIRef(expand(d["range"]))))
    g.serialize(BASE_EXT, format="turtle")

    GEN.mkdir(exist_ok=True)
    terms = GEN / "_slim_terms.txt"
    terms.write_text("\n".join([f"https://www.commoncoreontologies.org/{t}" for t in CCO_TERMS] +
                               [f"http://purl.obolibrary.org/obo/{t}" for t in BFO_TERMS]) + "\n")
    args = []
    for f in sorted(IMPORTS.glob("*.ttl")):
        args += ["--input", str(f)]
    robot("merge", *args, "extract", "--method", "BOT", "--term-file", str(terms),
          "annotate", "--ontology-iri", P6 + "cco_slim", "--output", str(SLIM))
    terms.unlink()
    n = len(Graph().parse(SLIM))
    print(f"build_ontology: {len(spec['local_classes'])} local classes, {len(spec['local_properties'])} local properties -> "
          f"ontology/p6_extension_base.ttl; CCO slim module {n} triples -> ontology/cco_slim.ttl")


if __name__ == "__main__":
    main()
