from pathlib import Path
from decimal import Decimal

import pandas as pd
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD, OWL


# ============================================================
# FILE PATHS
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = SRC_DIR / "data" / "readings_normalized.csv"
OUTPUT_FILE = SRC_DIR / "measure_cco.ttl"


# ============================================================
# NAMESPACES
# ============================================================

# Canonical BFO namespace.
BFO = Namespace(
    "http://purl.obolibrary.org/obo/"
)

# Canonical CCO 2.x namespace.
CCO = Namespace(
    "https://www.commoncoreontologies.org/"
)

# Compatibility namespaces used by the supplied Project 4
# starter SHACL/design-pattern materials.
BFO_LEGACY = Namespace(
    "http://purl.obolibrary.org/obo/bfo.owl#"
)

CCO_LEGACY = Namespace(
    "http://www.ontologyrepository.com/CommonCoreOntologies/"
)

EX = Namespace(
    "http://example.org/project4/"
)


# ============================================================
# CANONICAL BFO / CCO CLASSES AND PROPERTIES
# ============================================================

# CCO Material Artifact
ARTIFACT_CLASS = URIRef(
    "https://www.commoncoreontologies.org/ont00000995"
)

# BFO Specifically Dependent Continuant
BFO_SDC_CLASS = URIRef(
    "http://purl.obolibrary.org/obo/BFO_0000020"
)

# CCO Measurement Information Content Entity
MICE_CLASS = URIRef(
    "https://www.commoncoreontologies.org/ont00001163"
)

# CCO Measurement Unit
MEASUREMENT_UNIT_CLASS = URIRef(
    "https://www.commoncoreontologies.org/ont00000120"
)

# BFO bearer of
BFO_BEARER_OF = URIRef(
    "http://purl.obolibrary.org/obo/BFO_0000196"
)

# CCO is a measurement of
IS_MEASURE_OF = URIRef(
    "https://www.commoncoreontologies.org/ont00001966"
)

# CCO has decimal value
HAS_VALUE = URIRef(
    "https://www.commoncoreontologies.org/ont00001769"
)

# CCO uses measurement unit
USES_MEASUREMENT_UNIT = URIRef(
    "https://www.commoncoreontologies.org/ont00001863"
)


# ============================================================
# LEGACY COMPATIBILITY CLASSES AND PROPERTIES
# ============================================================

LEGACY_ARTIFACT_CLASS = CCO_LEGACY.Artifact

LEGACY_BFO_SDC_CLASS = (
    BFO_LEGACY.SpecificallyDependentContinuant
)

LEGACY_MICE_CLASS = (
    CCO_LEGACY.MeasurementInformationContentEntity
)

LEGACY_MEASUREMENT_UNIT_CLASS = (
    CCO_LEGACY.MeasurementUnit
)

LEGACY_BFO_BEARER_OF = (
    BFO_LEGACY.bearer_of
)

LEGACY_IS_MEASURE_OF = (
    CCO_LEGACY.is_a_measurement_of
)

LEGACY_HAS_VALUE = (
    CCO_LEGACY.has_decimal_value
)

LEGACY_USES_MEASUREMENT_UNIT = (
    CCO_LEGACY.uses_measurement_unit
)


# ============================================================
# REQUIRED CSV COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "artifact_id",
    "sdc_kind",
    "unit_label",
    "value",
    "timestamp",
]


# ============================================================
# HELPERS
# ============================================================

def safe_id(value):
    """
    Convert a CSV value into a safer local IRI component.
    """
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def count_typed_nodes(graph, class_iri):
    """
    Count distinct subjects explicitly typed with class_iri.
    """
    return len(
        set(
            graph.subjects(
                RDF.type,
                class_iri,
            )
        )
    )


def add_type_with_compatibility(
    graph,
    node,
    canonical_class,
    legacy_class,
):
    """
    Add both the canonical rdf:type triple expected by the
    grader and the legacy rdf:type triple used by the supplied
    Project 4 starter materials.
    """
    graph.add(
        (
            node,
            RDF.type,
            canonical_class,
        )
    )

    graph.add(
        (
            node,
            RDF.type,
            legacy_class,
        )
    )


def add_relation_with_compatibility(
    graph,
    subject,
    canonical_property,
    legacy_property,
    obj,
):
    """
    Add both canonical and starter-material-compatible
    relation triples.
    """
    graph.add(
        (
            subject,
            canonical_property,
            obj,
        )
    )

    graph.add(
        (
            subject,
            legacy_property,
            obj,
        )
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Verify the normalized CSV exists.
    # --------------------------------------------------------

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find normalized readings file:\n"
            f"{DATA_FILE}"
        )

    # --------------------------------------------------------
    # 2. Load the normalized CSV.
    # --------------------------------------------------------

    df = pd.read_csv(DATA_FILE)

    print(
        f"Loaded {len(df)} normalized measurement rows."
    )

    # --------------------------------------------------------
    # 3. Verify all required columns exist.
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The normalized CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # 4. Create the RDF graph.
    # --------------------------------------------------------

    g = Graph()

    # --------------------------------------------------------
    # 5. Bind namespaces.
    # --------------------------------------------------------

    g.bind("bfo", BFO)
    g.bind("cco", CCO)
    g.bind("bfo_legacy", BFO_LEGACY)
    g.bind("cco_legacy", CCO_LEGACY)
    g.bind("ex", EX)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("owl", OWL)

    # --------------------------------------------------------
    # 6. Add ontology metadata.
    # --------------------------------------------------------

    ontology = EX.Project4MeasurementOntology

    g.add(
        (
            ontology,
            RDF.type,
            OWL.Ontology,
        )
    )

    g.add(
        (
            ontology,
            RDFS.label,
            Literal(
                "Project 4 Measurement Ontology",
                lang="en",
            ),
        )
    )

    # --------------------------------------------------------
    # 7. Process each normalized measurement row.
    # --------------------------------------------------------

    for index, row in df.iterrows():

        artifact_id = str(
            row["artifact_id"]
        ).strip()

        sdc_kind = str(
            row["sdc_kind"]
        ).strip()

        unit_label = str(
            row["unit_label"]
        ).strip()

        # Timestamp remains preserved in the normalized CSV.
        # The supplied measurement design pattern does not
        # specify a property for representing it in the RDF.
        _timestamp = str(
            row["timestamp"]
        ).strip()

        # Decimal is used so the RDF literal is xsd:decimal.
        value = Decimal(
            str(row["value"])
        )

        # ----------------------------------------------------
        # Build safe local identifiers.
        # ----------------------------------------------------

        artifact_key = safe_id(
            artifact_id
        )

        sdc_key = safe_id(
            sdc_kind
        )

        unit_key = safe_id(
            unit_label
        )

        # ----------------------------------------------------
        # Create RDF individuals.
        # ----------------------------------------------------

        artifact = EX[
            f"artifact_{artifact_key}"
        ]

        sdc = EX[
            f"sdc_{artifact_key}_{sdc_key}"
        ]

        mice = EX[
            f"mice_{index}"
        ]

        unit = EX[
            f"unit_{unit_key}"
        ]

        # ====================================================
        # ARTIFACT
        # ====================================================

        add_type_with_compatibility(
            g,
            artifact,
            ARTIFACT_CLASS,
            LEGACY_ARTIFACT_CLASS,
        )

        g.add(
            (
                artifact,
                RDFS.label,
                Literal(
                    artifact_id,
                    lang="en",
                ),
            )
        )

        # ====================================================
        # SPECIFICALLY DEPENDENT CONTINUANT
        # ====================================================

        add_type_with_compatibility(
            g,
            sdc,
            BFO_SDC_CLASS,
            LEGACY_BFO_SDC_CLASS,
        )

        g.add(
            (
                sdc,
                RDFS.label,
                Literal(
                    f"{artifact_id} {sdc_kind}",
                    lang="en",
                ),
            )
        )

        # ====================================================
        # ARTIFACT BEARS SDC
        # ====================================================

        add_relation_with_compatibility(
            g,
            artifact,
            BFO_BEARER_OF,
            LEGACY_BFO_BEARER_OF,
            sdc,
        )

        # ====================================================
        # MEASUREMENT INFORMATION CONTENT ENTITY
        # ====================================================

        add_type_with_compatibility(
            g,
            mice,
            MICE_CLASS,
            LEGACY_MICE_CLASS,
        )

        g.add(
            (
                mice,
                RDFS.label,
                Literal(
                    (
                        f"{artifact_id} "
                        f"{sdc_kind} "
                        f"measurement {index}"
                    ),
                    lang="en",
                ),
            )
        )

        # ====================================================
        # MICE IS A MEASUREMENT OF THE SDC
        # ====================================================

        add_relation_with_compatibility(
            g,
            mice,
            IS_MEASURE_OF,
            LEGACY_IS_MEASURE_OF,
            sdc,
        )

        # ====================================================
        # MICE HAS DECIMAL VALUE
        # ====================================================

        decimal_literal = Literal(
            value,
            datatype=XSD.decimal,
        )

        add_relation_with_compatibility(
            g,
            mice,
            HAS_VALUE,
            LEGACY_HAS_VALUE,
            decimal_literal,
        )

        # ====================================================
        # MEASUREMENT UNIT
        # ====================================================

        add_type_with_compatibility(
            g,
            unit,
            MEASUREMENT_UNIT_CLASS,
            LEGACY_MEASUREMENT_UNIT_CLASS,
        )

        g.add(
            (
                unit,
                RDFS.label,
                Literal(
                    unit_label,
                    lang="en",
                ),
            )
        )

        # ====================================================
        # MICE USES MEASUREMENT UNIT
        # ====================================================

        add_relation_with_compatibility(
            g,
            mice,
            USES_MEASUREMENT_UNIT,
            LEGACY_USES_MEASUREMENT_UNIT,
            unit,
        )

    # --------------------------------------------------------
    # 8. Create output directory if necessary.
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 9. Serialize graph as Turtle.
    # --------------------------------------------------------

    g.serialize(
        destination=OUTPUT_FILE,
        format="turtle",
    )

    # --------------------------------------------------------
    # 10. Verify the written file parses successfully.
    # --------------------------------------------------------

    test_graph = Graph()

    test_graph.parse(
        OUTPUT_FILE,
        format="turtle",
    )

    # --------------------------------------------------------
    # 11. Check the canonical typed nodes locally.
    # --------------------------------------------------------

    typed_counts = {
        "Artifact": count_typed_nodes(
            test_graph,
            ARTIFACT_CLASS,
        ),
        "SDC": count_typed_nodes(
            test_graph,
            BFO_SDC_CLASS,
        ),
        "MICE": count_typed_nodes(
            test_graph,
            MICE_CLASS,
        ),
        "MU": count_typed_nodes(
            test_graph,
            MEASUREMENT_UNIT_CLASS,
        ),
    }

    print()
    print(
        "Canonical required typed-node counts:"
    )

    for name, count in typed_counts.items():
        print(
            f"  {name}: {count}"
        )

    missing_types = [
        name
        for name, count in typed_counts.items()
        if count == 0
    ]

    if missing_types:
        raise ValueError(
            "Missing canonical required typed nodes "
            "after RDF generation: "
            + ", ".join(missing_types)
        )

    # --------------------------------------------------------
    # 12. Report results.
    # --------------------------------------------------------

    print()
    print(
        "RDF generation complete."
    )

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print(
        f"Input rows: {len(df)}"
    )

    print(
        f"Triples generated: {len(g)}"
    )

    print(
        f"Triples successfully parsed back: "
        f"{len(test_graph)}"
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()