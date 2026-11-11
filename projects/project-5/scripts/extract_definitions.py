from pathlib import Path
import sys

import pandas as pd
from rdflib import Graph


# ============================================================
# PROJECT 5 - STEP 1: EXTRACT ONTOLOGY DEFINITIONS
# ============================================================

# Resolve project directories relative to this script.
# Expected location:
# projects/project-5/scripts/extract_definitions.py

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

# Source ontology
MODULE_FILE = SRC_DIR / "module.ttl"

# Output CSV
OUTPUT_FILE = DATA_DIR / "definitions.csv"


# ============================================================
# SPARQL QUERY
# ============================================================

CLASS_LABEL_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT ?iri ?label ?definition
WHERE {

    ?iri skos:definition ?definition .

    OPTIONAL {
        ?iri rdfs:label ?label .
    }

    FILTER(isIRI(?iri))

    FILTER NOT EXISTS {
        ?iri owl:deprecated ?deprecated .

        FILTER(
            ?deprecated = true ||
            LCASE(STR(?deprecated)) = "true"
        )
    }
}
ORDER BY ?label ?iri
"""


# ============================================================
# EXTRACT DEFINITIONS
# ============================================================

def extract_definitions(ttl_path: Path) -> pd.DataFrame:
    """
    Load a Turtle ontology and extract resources with
    SKOS definitions using a SPARQL SELECT query.

    Returns:
        DataFrame containing:
        - iri
        - label
        - definition
    """

    print(f"Loading ontology: {ttl_path}")

    graph = Graph()
    graph.parse(str(ttl_path), format="turtle")

    print(f"RDF triples loaded: {len(graph)}")
    print("Executing SPARQL query...")

    results = graph.query(CLASS_LABEL_QUERY)

    rows = []

    for row in results:
        values = row.asdict()

        iri_term = values.get("iri")
        label_term = values.get("label")
        definition_term = values.get("definition")

        rows.append({
            "iri": str(iri_term)
                if iri_term is not None else None,

            "label": str(label_term)
                if label_term is not None else None,

            "definition": str(definition_term)
                if definition_term is not None else None,
        })

    df = pd.DataFrame(
        rows,
        columns=["iri", "label", "definition"]
    )

    # Remove exact duplicate records.
    df.drop_duplicates(inplace=True)

    # Sort output consistently.
    df.sort_values(
        by=["label", "iri"],
        na_position="last",
        inplace=True
    )

    df.reset_index(drop=True, inplace=True)

    return df


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("PROJECT 5 - STEP 1: EXTRACT ONTOLOGY DEFINITIONS")
    print("=" * 65)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Source dir   : {SRC_DIR}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Input file   : {MODULE_FILE}")
    print(f"Output file  : {OUTPUT_FILE}")
    print()

    # Check that the ontology exists.
    if not MODULE_FILE.is_file():
        print("ERROR: Source ontology not found.")
        print(f"Expected: {MODULE_FILE}")
        return 1

    try:
        # Parse ontology and extract definitions.
        df = extract_definitions(MODULE_FILE)

        print()
        print(f"Definition records extracted: {len(df)}")

        # Prevent silently accepting an empty result.
        if df.empty:
            print("ERROR: No definitions were extracted.")
            print(
                "Check that module.ttl contains "
                "SKOS definition annotations."
            )
            return 1

        # Create output directory if needed.
        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # Export CSV.
        df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8"
        )

        print()
        print("CSV successfully generated.")
        print(f"Saved to: {OUTPUT_FILE}")

        # Display first five rows.
        print()
        print("First five extracted records:")
        print(df.head(5).to_string(index=False))

        print()
        print("=" * 65)
        print("STEP 1 STATUS: PASS")
        print("=" * 65)

        return 0

    except Exception as exc:
        print()
        print(f"ERROR: {type(exc).__name__}: {exc}")
        print("STEP 1 STATUS: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
