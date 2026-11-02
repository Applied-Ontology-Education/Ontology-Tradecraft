from pathlib import Path
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import XSD
import pandas as pd

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA = PROJECT_DIR / "src" / "data" / "readings_normalized.csv"
ONTOLOGY = PROJECT_DIR / "src" / "cco_merged.ttl"
OUTPUT = PROJECT_DIR / "src" / "measure_cco.ttl"


# Namespaces
OBO = Namespace("http://purl.obolibrary.org/obo/")
CCO = Namespace("https://www.commoncoreontologies.org/")
PROJ = Namespace("https://example.org/project4/")

# Classes in provided pattern
ARTIFACT = CCO["ont00000995"]  # material artifact
MICE = CCO["ont00001163"]  # measurement information content entity
SDC = OBO["BFO_0000020"]  # sdc
UNIT = CCO["ont00000120"]  # measurement unit

# Properties in provided pattern
BEARER_OF = OBO["BFO_0000196"]
IS_MEASUREMENT_OF = CCO["ont00001966"]
USES_MEASUREMENT_UNIT = CCO["ont00001863"]
HAS_DECIMAL_VALUE = CCO["ont00001769"]

# Map measurement units to classes
UNITS = {
    "C": CCO["ont00001606"],
    "F": CCO["ont00001724"],
    "kilopascal": CCO["ont00002000"],
    "ohm": CCO["ont00002001"],
    "pounds per square inch": CCO["ont00001694"],
    "V": CCO["ont00001450"],
}


# Create and manage IDs for individuals
def id(individual, map, individual_dict, individual_type):
    if individual in individual_dict.keys():
        id = individual_dict[individual]["ID"]
    else:
        if len(map.keys()) == 0:
            n = "0000000"
        else:
            n = "{:07d}".format(int(sorted(map.keys())[-1].split("_")[1]) + 1)
        id = f"P4_{n}"
        map[id] = individual
        individual_dict[individual] = {
            "ID": id,
            "Label": individual,
            "Type": individual_type
        }
    return id


def main():
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df)} rows")
    g = Graph()
    prefixes = {"obo": OBO, "cco": CCO, "rdf": RDF, "rdfs": RDFS, "xsd": XSD}
    individuals = {}
    mappings = {}
    for prefix, full in prefixes.items():
        g.bind(prefix, full)

    g.add((SDC, RDFS.label, Literal("specifically dependent continuant")))
    g.add((UNIT, RDFS.label, Literal("Measurement Unit")))
    g.add((ARTIFACT, RDFS.label, Literal("Material Artifact")))
    g.add((MICE, RDFS.label, Literal("Measurement Information Content Entity")))
    g.add((OBO["BFO_0000196"], RDFS.label, Literal("bearer of")))
    g.add((CCO["ont00001769"], RDFS.label, Literal("has decimal value")))
    g.add((CCO["ont00001863"], RDFS.label, Literal("uses measurement unit")))
    g.add((CCO["ont00001966"], RDFS.label, Literal("is a measurement of")))
    for index, row in df.iterrows():
        artifact = str(row["artifact_id"])
        sdc = str(row["sdc_kind"])
        unit = str(row["unit_label"])
        value = float(row["value"])
        timestamp = str(row["timestamp"])

        artifact_id = PROJ[id(artifact, mappings, individuals, ARTIFACT)]
        g.add((artifact_id, RDF.type, ARTIFACT))
        g.add((artifact_id, RDFS.label, Literal(artifact)))
        sdc_label = f"{sdc} of {artifact}"
        sdc_id = PROJ[id(sdc_label, mappings, individuals, SDC)]
        g.add((sdc_id, RDF.type, SDC))
        g.add((sdc_id, RDFS.label, Literal(sdc_label)))
        g.add((artifact_id, BEARER_OF, sdc_id))
        mice_label = f"measurement of {sdc_label} at {timestamp}"
        mice_id = PROJ[id(mice_label, mappings, individuals, MICE)]
        g.add((mice_id, RDF.type, MICE))
        g.add((mice_id, RDFS.label, Literal(mice_label)))
        g.add((mice_id, HAS_DECIMAL_VALUE, Literal(value, datatype=XSD.decimal)))
        g.add((mice_id, IS_MEASUREMENT_OF, sdc_id))
        g.add((UNITS[unit], RDF.type, UNIT))
        g.add((UNITS[unit], RDFS.label, Literal(unit)))
        g.add((mice_id, USES_MEASUREMENT_UNIT, UNITS[unit]))

    print(f"Constructed {len(g)} triples")
    g.serialize(destination=str(OUTPUT), format="turtle")
    print(f"Wrote to {OUTPUT}")


if __name__ == "__main__":
    main()
