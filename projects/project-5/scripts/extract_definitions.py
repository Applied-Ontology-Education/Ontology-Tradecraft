from pathlib import Path

import pandas as pd
from rdflib import Graph


# ============================================================
# PROJECT PATHS
# ============================================================

# This script should live at:
#
# projects/project-5/scripts/extract_definitions.py
#
# Therefore parents[1] is:
#
# projects/project-5/

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = SRC_DIR / "module.ttl"
OUTPUT_FILE = DATA_DIR / "definitions.csv"


# ============================================================
# SPARQL QUERY
# ============================================================

DEFINITION_QUERY = """
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

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
    Parse the CCO module and extract resources that have
    skos:definition annotations.

    Output columns:
        IRI
        label
        definition
    """

    graph = Graph()

    graph.parse(
        ttl_path,
        format="turtle",
    )

    results = graph.query(
        DEFINITION_QUERY
    )

    rows = []

    for row in results:
        values = row.asdict()

        iri = values.get("iri")
        label = values.get("label")
        definition = values.get("definition")

        rows.append(
            {
                "IRI": str(iri) if iri is not None else "",
                "label": str(label) if label is not None else "",
                "definition": (
                    str(definition)
                    if definition is not None
                    else ""
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "IRI",
            "label",
            "definition",
        ],
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input file   : {INPUT_FILE}")
    print(f"Output file  : {OUTPUT_FILE}")
    print()

    # --------------------------------------------------------
    # Verify module exists
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find CCO module:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Ensure output directory exists
    # --------------------------------------------------------

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Extract definitions
    # --------------------------------------------------------

    print(
        f"Reading definitions from {INPUT_FILE.name}..."
    )

    df = extract_definitions(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # Verify results
    # --------------------------------------------------------

    if df.empty:
        print(
            "WARNING: No skos:definition annotations were found."
        )

    # --------------------------------------------------------
    # Save canonical Project 5 output
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("Definition extraction complete.")
    print(f"Definitions extracted: {len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")

    if not df.empty:
        print()
        print("Sample:")
        print(
            df.head(5).to_string(
                index=False
            )
        )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()