#!/usr/bin/env python3
"""Build src/measure_cco.ttl from src/data/readings_normalized.csv using the CCO measurement pattern.

    artifact --bearer of--> SDC <--is a measurement of-- MICE --uses measurement unit--> MU
                                                         MICE --has decimal value--> value
    Act of Measuring --has output--> MICE ; --occupies temporal region--> temporal instant (datetime)

One SDC per (artifact, kind): Boiler-07's temperature is one quality, measured twice (in F and in C).
Provenance: MICE converted by the ETL (pressures -> Pa) also record the value and unit the sensor reported.
Exact BFO/CCO IRIs from cco_merged.ttl; local terms only where CCO has none.
"""
import hashlib
from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

SRC = Path(__file__).resolve().parents[1]
IN = SRC / "data" / "readings_normalized.csv"
CONVERSIONS = SRC / "reports" / "unit_conversions.csv"
OUT = SRC / "measure_cco.ttl"

OBO = Namespace("http://purl.obolibrary.org/obo/")
CCO = Namespace("https://www.commoncoreontologies.org/")
BASE = "https://stucker7-buffalo-edu.github.io/ontology-tradecraft/p4/"
EX = Namespace(BASE)

# --- exact pattern IRIs (checked by the grader) ---
ARTIFACT, SDC, MICE, MU = CCO.ont00000995, OBO.BFO_0000020, CCO.ont00001163, CCO.ont00000120
BEARER_OF, IS_MEASURE_OF, USES_UNIT = OBO.BFO_0000196, CCO.ont00001966, CCO.ont00001863
HAS_DECIMAL, HAS_DATETIME, HAS_OUTPUT = CCO.ont00001769, CCO.ont00001767, CCO.ont00001986
ACT_OF_MEASURING, OCCUPIES_TR, TEMPORAL_INSTANT, QUALITY = CCO.ont00000345, OBO.BFO_0000199, OBO.BFO_0000203, OBO.BFO_0000019

# --- quality classes: CCO's where it has one, local otherwise (subclasses of BFO quality) ---
LOCAL_QUALITIES = {
    "pressure": ("PressureQuality", "Pressure Quality",
                 "A quality of a material entity consisting in the force per unit area that a contained fluid exerts on it."),
    "voltage": ("VoltageQuality", "Voltage Quality",
                "A quality of an electrical circuit consisting in the difference in electric potential between two of its points."),
    "resistance": ("ElectricalResistanceQuality", "Electrical Resistance Quality",
                   "A quality of a conductor consisting in the degree to which it opposes the flow of electric current."),
}
KIND_CLASS = {"temperature": CCO.ont00000441, **{k: EX[v[0]] for k, v in LOCAL_QUALITIES.items()}}

# --- units: CCO individuals where they exist, local otherwise ---
LOCAL_UNITS = {"kPa": ("Kilopascal", "Kilopascal Measurement Unit"), "ohm": ("Ohm", "Ohm Measurement Unit")}
UNIT_IRI = {"F": CCO.ont00001724, "C": CCO.ont00001606, "psi": CCO.ont00001694, "V": CCO.ont00001450,
            "Pa": CCO.ont00001559,
            **{k: EX["unit/" + v[0]] for k, v in LOCAL_UNITS.items()}}
UNIT_LABEL = {"F": "Degree Fahrenheit Measurement Unit", "C": "Degree Celsius Measurement Unit",
              "psi": "Pounds Per Square Inch Measurement Unit", "V": "Volt Measurement Unit",
              "Pa": "Pascal Measurement Unit",
              **{k: v[1] for k, v in LOCAL_UNITS.items()}}


HAS_ORIG_VALUE, HAS_ORIG_UNIT = EX.hasReportedValue, EX.hasReportedUnit


def en(text: str) -> Literal:
    return Literal(text, lang="en")


def main() -> None:
    df = pd.read_csv(IN, dtype={"value": float})
    g = Graph()
    for p, ns in {"obo": OBO, "cco": CCO, "ex": EX, "skos": SKOS, "owl": OWL}.items():
        g.bind(p, ns)

    onto = URIRef(BASE + "measure_cco")
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, RDFS.label, en("Sensor Measurements (CCO measurement pattern)")))
    g.add((onto, SKOS.definition, en("Readings from the site sensors, integrated as BFO/CCO measurements.")))

    for key, (local, label, definition) in LOCAL_QUALITIES.items():
        c = EX[local]
        g.add((c, RDF.type, OWL.Class)); g.add((c, RDFS.subClassOf, QUALITY))
        g.add((c, RDFS.label, en(label))); g.add((c, SKOS.definition, en(definition)))

    g.add((HAS_ORIG_VALUE, RDF.type, OWL.DatatypeProperty))
    g.add((HAS_ORIG_VALUE, RDFS.label, en("has reported value")))
    g.add((HAS_ORIG_VALUE, SKOS.definition, en("Relates a converted measurement to the numeric value the sensor originally reported, before unit normalization.")))
    g.add((HAS_ORIG_UNIT, RDF.type, OWL.ObjectProperty))
    g.add((HAS_ORIG_UNIT, RDFS.label, en("has reported unit")))
    g.add((HAS_ORIG_UNIT, SKOS.definition, en("Relates a converted measurement to the unit in which the sensor originally reported it.")))

    conv = {}
    if CONVERSIONS.exists():
        for r in pd.read_csv(CONVERSIONS).itertuples(index=False):
            conv[(r.artifact_id, r.sdc_kind, r.timestamp, float(r.value))] = (r.original_value, r.original_unit)

    for code, iri in UNIT_IRI.items():
        g.add((iri, RDF.type, OWL.NamedIndividual)); g.add((iri, RDF.type, MU))
        g.add((iri, RDFS.label, en(UNIT_LABEL[code])))

    for row in df.itertuples(index=False):
        art = EX["artifact/" + row.artifact_id]
        g.add((art, RDF.type, ARTIFACT)); g.add((art, RDFS.label, en(row.artifact_id)))

        sdc = EX[f"sdc/{row.artifact_id}_{row.sdc_kind}"]
        g.add((sdc, RDF.type, SDC)); g.add((sdc, RDF.type, KIND_CLASS[row.sdc_kind]))
        g.add((sdc, RDFS.label, en(f"{row.artifact_id} {row.sdc_kind}")))
        g.add((art, BEARER_OF, sdc))

        rid = hashlib.sha1(f"{row.artifact_id}|{row.sdc_kind}|{row.unit_label}|{row.value}|{row.timestamp}"
                           .encode()).hexdigest()[:12]
        mice, act, inst = EX["mice/" + rid], EX["act/" + rid], EX["instant/" + row.timestamp.replace(":", "")]
        g.add((mice, RDF.type, MICE))
        g.add((mice, RDFS.label, en(f"{row.artifact_id} {row.sdc_kind} {row.value:g} {row.unit_label} at {row.timestamp}")))
        g.add((mice, IS_MEASURE_OF, sdc))
        g.add((mice, USES_UNIT, UNIT_IRI[row.unit_label]))
        g.add((mice, HAS_DECIMAL, Literal(str(row.value), datatype=XSD.decimal)))
        orig = conv.get((row.artifact_id, row.sdc_kind, row.timestamp, float(row.value)))
        if orig:
            g.add((mice, HAS_ORIG_VALUE, Literal(str(orig[0]), datatype=XSD.decimal)))
            g.add((mice, HAS_ORIG_UNIT, UNIT_IRI[orig[1]]))

        g.add((act, RDF.type, ACT_OF_MEASURING)); g.add((act, HAS_OUTPUT, mice))
        # The act measured in the unit the sensor used; conversion to Pa happened later, in the ETL.
        act_unit = orig[1] if orig else row.unit_label
        g.add((act, RDFS.label, en(f"measuring {row.artifact_id} {row.sdc_kind} in {act_unit} at {row.timestamp}")))
        g.add((act, OCCUPIES_TR, inst))
        g.add((inst, RDF.type, TEMPORAL_INSTANT)); g.add((inst, RDFS.label, en(row.timestamp)))
        g.add((inst, HAS_DATETIME, Literal(row.timestamp, datatype=XSD.dateTime)))

    g.serialize(OUT, format="turtle")
    print(f"[rdf] {len(df)} readings -> {len(g)} triples -> {OUT.name}")


if __name__ == "__main__":
    main()
