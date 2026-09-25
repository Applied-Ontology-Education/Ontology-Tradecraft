#!/usr/bin/env python3
"""Negative controls: prove each SPARQL QC query and each SHACL shape catches the defect it targets.

For every check, a copy of measure_cco.ttl gets exactly one planted defect. The check must flag the
defective copy, and must pass the clean graph (run_sparql_qc.py / run_shacl_validate.py show that).
Run after measure_rdflib.py:  python src/scripts/check_controls.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

SRC = Path(__file__).resolve().parents[1]
DATA, SHAPES, QUERIES = SRC / "measure_cco.ttl", SRC / "cco_shapes.ttl", SRC / "sparql"
OBO = Namespace("http://purl.obolibrary.org/obo/")
CCO = Namespace("https://www.commoncoreontologies.org/")
EX = Namespace("http://example.org/tradecraft/p4/")
X = Namespace("http://example.org/tradecraft/p4/control/")
MICE, SDC, ART, MU = CCO.ont00001163, OBO.BFO_0000020, CCO.ont00000995, CCO.ont00000120


def first_mice(g: Graph, unit=None) -> URIRef:
    for m in sorted(g.subjects(RDF.type, MICE)):
        if unit is None or (m, CCO.ont00001863, unit) in g:
            return m
    raise LookupError("no MICE")


def qc_defects():
    def orphan(g):  g += [(X.Orphan, RDF.type, OWL.Class), (X.Orphan, RDFS.label, Literal("Orphan", lang="en")),
                          (X.Orphan, SKOS.definition, Literal("A class with no parent.", lang="en"))]
    def space(g):   g.add((X.Spaced, RDFS.label, Literal(" Spaced ", lang="en")))
    def nodef(g):   g += [(X.NoDef, RDF.type, OWL.Class), (X.NoDef, RDFS.subClassOf, OBO.BFO_0000019),
                          (X.NoDef, RDFS.label, Literal("No Def", lang="en"))]
    def notitle(g): g.add((X.onto, RDF.type, OWL.Ontology))
    def dupdef(g):  g += [(X.A, SKOS.definition, Literal("Same text.", lang="en")),
                          (X.B, SKOS.definition, Literal("Same text.", lang="en"))]
    def duplab(g):  g += [(X.C, RDFS.label, Literal("Twin", lang="en")), (X.D, RDFS.label, Literal("Twin"))]
    def selfref(g): g += [(X.Widget, RDF.type, OWL.Class), (X.Widget, RDFS.subClassOf, OBO.BFO_0000040),
                          (X.Widget, RDFS.label, Literal("widget", lang="en")),
                          (X.Widget, SKOS.definition, Literal("A widget is an object made of widget parts.", lang="en"))]
    def cycle(g):   g += [(X.P, RDF.type, OWL.Class), (X.Q, RDF.type, OWL.Class),
                          (X.P, RDFS.subClassOf, X.Q), (X.Q, RDFS.subClassOf, X.P)]
    return {"all_classes_have_BFO_parent.rq": orphan, "annotation_whitespace.rq": space,
            "min_1_eng_def.rq": nodef, "min_1_ontology_title.rq": notitle, "no_duplicate_def.rq": dupdef,
            "no_duplicate_labels.rq": duplab, "no_self-referential_def.rq": selfref,
            "subclass_circularity.rq": cycle}


def shacl_defects():
    def no_sdc(g):     g += [(X.bare, RDF.type, ART), (X.bare, RDFS.label, Literal("bare artifact"))]
    def unmeasured(g): g += [(X.sdc, RDF.type, SDC), (X.sdc, RDFS.label, Literal("unmeasured sdc"))]
    def two_sdc(g):    g.add((first_mice(g), CCO.ont00001966, EX["sdc/Pump-A1/pressure"]))
    def no_value(g):   g.remove((first_mice(g), CCO.ont00001769, None))
    def empty(g):      g.add((first_mice(g), RDFS.comment, Literal("  ")))
    def bad_unit(g):
        m = first_mice(g, CCO.ont00001606)                     # a Celsius temperature reading
        g.remove((m, CCO.ont00001863, None)); g.add((m, CCO.ont00001863, CCO.ont00001559))   # -> Pa
    def duplicate(g):
        m = first_mice(g)
        for p, o in list(g.predicate_objects(m)):
            g.add((X.copy, p, o if p != RDFS.label else Literal("copy")))
    def dangling(g):   g.add((first_mice(g), CCO.ont00001863, X.nowhere))
    def mistyped(g):   g.add((first_mice(g), RDF.type, ART))
    return {"ArtifactHasSDCShape": no_sdc, "SDCMeasuredShape": unmeasured, "MICEMeasuresOneSDCShape": two_sdc,
            "MICECompleteShape": no_value, "NoEmptyLiteralShape": empty, "UnitMatchesSDCShape": bad_unit,
            "NoDuplicateMICEShape": duplicate, "NoDanglingOrMistypedNodeShape (dangling)": dangling,
            "NoDanglingOrMistypedNodeShape (mistyped)": mistyped}


def main() -> int:
    base = Graph().parse(DATA)
    shapes = Graph().parse(SHAPES)
    failures = 0
    print("SPARQL QC negative controls")
    for name, plant in qc_defects().items():
        g = Graph(); g += base; plant(g)
        rows = list(g.query((QUERIES / name).read_text(encoding="utf-8")))
        ok = len(rows) > 0
        failures += not ok
        print(f"  {'caught' if ok else 'MISSED'}  {name}: {len(rows)} row(s)")
    print("SHACL negative controls")
    for name, plant in shacl_defects().items():
        g = Graph(); g += base; plant(g)
        conforms, report, _ = validate(g, shacl_graph=shapes, inference="rdfs", advanced=True)
        shape = URIRef("http://example.org/tradecraft/p4/shapes/" + name.split()[0])
        hits = [r for r in report.subjects(RDF.type, URIRef("http://www.w3.org/ns/shacl#ValidationResult"))
                if (r, URIRef("http://www.w3.org/ns/shacl#sourceShape"), None) in report
                and (shape in set(report.objects(r, URIRef("http://www.w3.org/ns/shacl#sourceShape")))
                     or any((shape, None, s) in shapes for s in report.objects(r, URIRef("http://www.w3.org/ns/shacl#sourceShape"))))]
        ok = not conforms and len(hits) > 0
        failures += not ok
        print(f"  {'caught' if ok else 'MISSED'}  {name}: {len(hits)} result(s) from this shape")
    print("all controls caught" if not failures else f"{failures} control(s) missed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
