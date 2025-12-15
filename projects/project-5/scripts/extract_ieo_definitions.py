import logging

import pandas as pd
import rdflib
from rdflib.namespace import RDF, RDFS, OWL

from util.logger_config import config

logger = logging.getLogger(__name__)
from pathlib import Path

config(logger)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
GENERATED_ROOT = PROJECT_ROOT / "generated"
REPORTS_ROOT = PROJECT_ROOT / "reports"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
ONTOLOGIES_ROOT = PROJECT_ROOT / "src"
INPUT_FILE = ONTOLOGIES_ROOT / "InformationEntityOntology.ttl"
OUTPUT_FILE = DATA_ROOT / "definitions.csv"


def read_ttl(path):
    g = rdflib.Graph()
    g.parse(path, format="turtle")
    return g


def extract_definitions(g):
    rows = []
    query = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?s ?label ?definition {
        ?s skos:definition ?definition .
        OPTIONAL { 
            ?s rdfs:label ?label .
        }
    }
    """
    rows = []
    for row in g.query(query):
        s = row[0]
        label = row[1]
        definition = row[2]

        # Determine if the subject is a class or a property based on rdf:type
        types = set(g.objects(s, RDF.type))
        entity_type = "class"
        if any(t in {OWL.Class, RDFS.Class} for t in types):
            entity_type = "class"
        elif any(
            t in {
                RDF.Property,
                OWL.ObjectProperty,
                OWL.DatatypeProperty,
                OWL.AnnotationProperty,
            }
            for t in types
        ):
            entity_type = "property"

        if entity_type == "class":
            label = label.capitalize()
        if entity_type == "property":
            label = label.lower()
        rows.append([str(s), str(label), str(definition), entity_type])
    return rows


def rows_to_df(rows):
    return pd.DataFrame(rows, columns=["iri", "label", "definition", "type"])


def write_df_to_csv(df, OUTPUT_FILE):
    df.to_csv(OUTPUT_FILE, index=False)


def main():
    logger.info("=======================================")
    logger.info("Extracting definitions from TTL file")
    logger.info("=======================================")
    logger.info("Reading TTL file...")
    g = read_ttl(INPUT_FILE)
    logger.info("Extracting definitions...")
    rows = extract_definitions(g)
    logger.info(f"Found {len(rows)} definitions.")
    logger.info("Writing to CSV...")
    df = rows_to_df(rows)
    write_df_to_csv(df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
