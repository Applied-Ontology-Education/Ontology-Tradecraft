#!/usr/bin/env python3
"""Build src/measure_cco.ttl: the CCO measurement design pattern, one instance per normalised reading.

    artifact  --bfo:bearer of (BFO_0000196)-->  SDC
    MICE      --cco:is a measurement of (ont00001966)-->  SDC
    MICE      --cco:uses measurement unit (ont00001863)-->  unit
    MICE      --cco:has decimal value (ont00001769)-->  "value"^^xsd:decimal
    MICE      --cco:has datetime value (ont00001767)-->  "timestamp"^^xsd:dateTime

The grader queries without inference, so each individual is typed with the pattern class itself
(Material Artifact, SDC, MICE, Measurement Unit) as well as its more specific class. The graph also
carries a TBox slice (label, English definition, named superclasses up to BFO) for every class and
property it uses, taken from cco_merged.ttl, so the SPARQL QC queries have something real to check.

Course pattern vs CCO: with cco_merged.ttl, this graph is INCONSISTENT (HermiT and ELK agree).
In CCO, `uses measurement unit` is the inverse of `is measurement unit of`, a sub-property of BFO
`generically depends on`, whose range is independent continuant. So the unit (a GDC) would have to
be an independent continuant, and BFO makes the two disjoint. CCO's own pattern puts the unit and
the value on an Information Bearing Entity: MICE is concretized by an Information Quality Entity,
which inheres in the IBE. `--cco-strict` writes that variant to measure_cco_strict.ttl (consistent
with CCO). The default output keeps the course pattern because the grader requires it.

Local terms (namespace EX) exist only where CCO has no suitable term:
- CCO's Pressure is a Force, which CCO places under Process Profile (an occurrent), so it cannot be
  the SDC a boiler bears. Fluid Pressure, Electric Potential Difference and Electrical Resistance are
  defined here as BFO qualities.
- CCO has Volt and Pascal units but no ohm, so an Ohm unit and its unit class are defined here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

SRC = Path(__file__).resolve().parents[1]
IN_CSV = SRC / "data" / "readings_normalized.csv"
CCO_FILE = SRC / "cco_merged.ttl"
OUT = SRC / "measure_cco.ttl"
OUT_STRICT = SRC / "measure_cco_strict.ttl"

OBO = Namespace("http://purl.obolibrary.org/obo/")
CCO = Namespace("https://www.commoncoreontologies.org/")
EX = Namespace("http://example.org/tradecraft/p4/")

SDC, QUALITY = OBO.BFO_0000020, OBO.BFO_0000019
BEARER_OF = OBO.BFO_0000196
ARTIFACT, PUMP = CCO.ont00000995, CCO.ont00000116
MICE, MU = CCO.ont00001163, CCO.ont00000120
IS_MEASURE_OF, USES_MU = CCO.ont00001966, CCO.ont00001863
HAS_DECIMAL, HAS_DATETIME = CCO.ont00001769, CCO.ont00001767
IBE, IQE = CCO.ont00000253, CCO.ont00000314
IS_CONCRETIZED_BY, INHERES_IN = OBO.BFO_0000058, OBO.BFO_0000197

LOCAL_CLASSES = {
    EX.FluidPressure: ("Fluid Pressure", QUALITY,
        "A quality of a portion of fluid, or of the artifact containing it, that is the force per unit area the fluid exerts on its boundary."),
    EX.ElectricPotentialDifference: ("Electric Potential Difference", QUALITY,
        "A quality of an electrical circuit that is the work needed per unit charge to move a test charge between two of its terminals."),
    EX.ElectricalResistance: ("Electrical Resistance", QUALITY,
        "A quality of a conductor that is the opposition it offers to the passage of an electric current."),
    EX.MeasurementUnitOfElectricalResistance: ("Measurement Unit of Electrical Resistance", MU,
        "A Measurement Unit that is used as a standard for measurement of the opposition of a conductor to electric current."),
}
KIND_CLASS = {"temperature": CCO.ont00000441, "pressure": EX.FluidPressure,
              "voltage": EX.ElectricPotentialDifference, "resistance": EX.ElectricalResistance}
UNIT_IRI = {"C": CCO.ont00001606, "F": CCO.ont00001724, "Pa": CCO.ont00001559,
            "V": CCO.ont00001450, "ohm": EX.Ohm}


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-")


def english(values):
    vals = [v for v in values if isinstance(v, Literal)]
    return next((v for v in vals if (v.language or "").startswith("en")), vals[0] if vals else None)


def add_tbox(g: Graph, cco: Graph, terms: set[URIRef]) -> None:
    """Copy label, English definition, type and named superclasses for each term and its ancestors."""
    todo, seen = list(terms), set()
    while todo:
        t = todo.pop()
        if t in seen:
            continue
        seen.add(t)
        for ty in cco.objects(t, RDF.type):
            if ty in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.NamedIndividual) or isinstance(ty, URIRef):
                g.add((t, RDF.type, ty))
                if ty not in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.NamedIndividual):
                    todo.append(ty)
        lab = english(cco.objects(t, RDFS.label))
        if lab is not None:
            g.add((t, RDFS.label, Literal(str(lab).strip(), lang="en")))
        dfn = english(cco.objects(t, SKOS.definition))
        if dfn is not None:
            g.add((t, SKOS.definition, Literal(str(dfn).strip(), lang="en")))
        for sup in cco.objects(t, RDFS.subClassOf):
            if isinstance(sup, URIRef) and sup != OWL.Thing:
                g.add((t, RDFS.subClassOf, sup))
                todo.append(sup)


def main(strict: bool = False) -> None:
    df = pd.read_csv(IN_CSV, dtype={"value": str})
    cco = Graph().parse(CCO_FILE)
    g = Graph()
    for p, ns in (("obo", OBO), ("cco", CCO), ("ex", EX), ("skos", SKOS)):
        g.bind(p, ns)

    onto = URIRef(str(EX).rstrip("/"))
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, RDFS.label, Literal("Project 4 sensor measurement graph", lang="en")))

    for cls, (label, parent, definition) in LOCAL_CLASSES.items():
        g.add((cls, RDF.type, OWL.Class))
        g.add((cls, RDFS.label, Literal(label, lang="en")))
        g.add((cls, SKOS.definition, Literal(definition, lang="en")))
        g.add((cls, RDFS.subClassOf, parent))
    g.add((EX.Ohm, RDF.type, OWL.NamedIndividual))
    g.add((EX.Ohm, RDF.type, EX.MeasurementUnitOfElectricalResistance))
    g.add((EX.Ohm, RDFS.label, Literal("Ohm Measurement Unit", lang="en")))

    used = {SDC, ARTIFACT, PUMP, MICE, MU, BEARER_OF, IS_MEASURE_OF, USES_MU, HAS_DECIMAL, HAS_DATETIME,
            *([IBE, IQE, IS_CONCRETIZED_BY, INHERES_IN] if strict else []),
            QUALITY, *[c for c in KIND_CLASS.values() if c.startswith(str(CCO))],
            *[u for u in UNIT_IRI.values() if u.startswith(str(CCO))]}
    add_tbox(g, cco, used)

    for unit in set(df["unit_label"]):
        g.add((UNIT_IRI[unit], RDF.type, MU))                      # asserted, not inferred

    for row in df.itertuples(index=False):
        art = EX[f"artifact/{slug(row.artifact_id)}"]
        sdc = EX[f"sdc/{slug(row.artifact_id)}/{row.sdc_kind}"]
        # One IRI per reading. The value is part of the key: after unit normalisation, two sources can
        # report the same SDC at the same instant in the same unit with different values
        # (Boiler-07 pressure at 23:04Z: 14.7 psi and 101.325 kPa), and they must stay two MICE.
        mice = EX[f"mice/{slug(row.artifact_id)}/{row.sdc_kind}/{slug(row.timestamp)}/{slug(row.value)}-{row.unit_label}"]

        g.add((art, RDF.type, ARTIFACT))
        if row.artifact_id.lower().startswith("pump"):
            g.add((art, RDF.type, PUMP))
        g.add((art, RDFS.label, Literal(row.artifact_id, lang="en")))
        g.add((art, BEARER_OF, sdc))

        g.add((sdc, RDF.type, SDC))
        g.add((sdc, RDF.type, KIND_CLASS[row.sdc_kind]))
        g.add((sdc, RDFS.label, Literal(f"{row.sdc_kind} of {row.artifact_id}", lang="en")))

        g.add((mice, RDF.type, MICE))
        g.add((mice, RDFS.label, Literal(
            f"{row.artifact_id} {row.sdc_kind} reading of {row.value} {row.unit_label} at {row.timestamp}", lang="en")))
        g.add((mice, IS_MEASURE_OF, sdc))
        carrier = mice
        if strict:                      # CCO pattern: unit and value live on the information bearer
            iqe, carrier = URIRef(f"{mice}/iqe"), URIRef(f"{mice}/ibe")
            g.add((mice, IS_CONCRETIZED_BY, iqe))
            g.add((iqe, RDF.type, IQE))
            g.add((iqe, RDFS.label, Literal(f"information quality of {row.artifact_id} record {row.timestamp} {row.value} {row.unit_label}", lang="en")))
            g.add((iqe, INHERES_IN, carrier))
            g.add((carrier, RDF.type, IBE))
            g.add((carrier, RDFS.label, Literal(f"sensor record of {row.artifact_id} {row.sdc_kind} {row.value} {row.unit_label} at {row.timestamp}", lang="en")))
        g.add((carrier, USES_MU, UNIT_IRI[row.unit_label]))
        g.add((carrier, HAS_DECIMAL, Literal(row.value, datatype=XSD.decimal)))
        g.add((carrier, HAS_DATETIME, Literal(row.timestamp, datatype=XSD.dateTime)))

    out = OUT_STRICT if strict else OUT
    g.serialize(out, format="turtle")
    print(f"[measure_rdflib] {len(df)} readings -> {out} ({len(g)} triples)")


if __name__ == "__main__":
    import sys
    main(strict="--cco-strict" in sys.argv)
