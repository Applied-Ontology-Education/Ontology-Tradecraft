from pathlib import Path
import sys

import pandas as pd
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDFS, SKOS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = PROJECT_ROOT / "src" / "cco-module.ttl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "definitions.csv"

REQUIRED_COLUMNS = ["IRI", "label", "definition"]


def extract_definitions(source_file):
    """Extract and validate SKOS definitions from the Quality Ontology."""

    ontology = Graph()
    ontology.parse(source_file, format="turtle")

    records = []
    skipped = []

    for concept, definition in ontology.subject_objects(SKOS.definition):

        if not isinstance(concept, URIRef):
            skipped.append((str(concept), "Subject is not an IRI"))
            continue

        labels = list(ontology.objects(concept, RDFS.label))

        if not labels:
            skipped.append((str(concept), "Missing rdfs:label"))
            continue

        label = str(labels[0]).strip()
        definition = str(definition).strip()

        if not label:
            skipped.append((str(concept), "Empty label"))
            continue

        if not definition:
            skipped.append((str(concept), "Empty definition"))
            continue

        records.append({
            "IRI": str(concept),
            "label": label,
            "definition": definition
        })

    results = pd.DataFrame(records, columns=REQUIRED_COLUMNS)

    results.drop_duplicates(subset="IRI", inplace=True)
    results.sort_values("label", inplace=True)
    results.reset_index(drop=True, inplace=True)

    return results, skipped, len(ontology)


def main():
    print("QUALITY ONTOLOGY - DEFINITION EXTRACTION")
    print("-" * 45)

    if not SOURCE_FILE.exists():
        print(f"ERROR: Source file not found: {SOURCE_FILE}")
        return 1

    try:
        definitions, skipped, triple_count = extract_definitions(SOURCE_FILE)

        if definitions.empty:
            print("ERROR: No valid definitions were found.")
            return 1

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        definitions.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

        print(f"Ontology triples loaded: {triple_count}")
        print(f"Valid definitions extracted: {len(definitions)}")
        print(f"Records skipped: {len(skipped)}")
        print(f"Output saved to: {OUTPUT_FILE}")

        if skipped:
            print("\nSkipped records:")
            for iri, reason in skipped:
                print(f"  {iri}: {reason}")

        print("\nSample extracted definitions:")
        print(definitions.head(5).to_string(index=False))

        print("\nEXTRACTION COMPLETED SUCCESSFULLY")
        return 0

    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
