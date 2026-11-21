from pathlib import Path
import argparse
import json
import os
import re
import time

import pandas as pd

from openai import OpenAI

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = PROJECT_ROOT / "generated"

MODULE_FILE = SRC_DIR / "module.ttl"

DEFINITIONS_FILE = (
    DATA_DIR
    / "definitions_enriched.csv"
)

OUTPUT_TTL = (
    GENERATED_DIR
    / "candidate_el.ttl"
)

OUTPUT_CSV = (
    GENERATED_DIR
    / "candidate_el_candidates.csv"
)

PROGRESS_CSV = (
    GENERATED_DIR
    / "candidate_generation_progress.csv"
)


# ============================================================
# XAI SETTINGS
# ============================================================

XAI_BASE_URL = "https://api.x.ai/v1"

MODEL = os.getenv(
    "XAI_MODEL",
    "grok-4.7",
)

MAX_RETRIES = 5

RETRY_DELAY_SECONDS = 5


# ============================================================
# CANDIDATE GENERATION SETTINGS
# ============================================================

# Increased from 30.
#
# Grok will see the 50 highest-ranked possible
# superclasses for each class.
CANDIDATE_POOL_SIZE = 50


# Increased from 3.
#
# These are only hypotheses at this stage.
# Step 6 will perform stricter hybrid filtering.
MAX_CANDIDATES_PER_CLASS = 5


# Number of superclass options printed to the terminal.
DIAGNOSTIC_OPTIONS_TO_SHOW = 10


# ============================================================
# OUTPUT COLUMNS
# ============================================================

AUDIT_COLUMNS = [
    "subclass_iri",
    "subclass_label",
    "superclass_iri",
    "superclass_label",
    "reason",
]


PROGRESS_COLUMNS = [
    "subclass_iri",
    "subclass_label",
    "status",
    "candidate_count",
]


# ============================================================
# BASIC TEXT PROCESSING
# ============================================================

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "which",
    "with",
}


def tokenize(text):

    tokens = re.findall(
        r"[A-Za-z0-9]+",
        str(text).lower(),
    )

    return {
        token
        for token in tokens
        if (
            token not in STOPWORDS
            and len(token) > 1
        )
    }


def local_name(iri):

    text = str(iri)

    if "#" in text:
        return text.rsplit(
            "#",
            1,
        )[1]

    return (
        text
        .rstrip("/")
        .rsplit("/", 1)[-1]
    )


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Generate OWL 2 EL candidate subclass "
            "axioms from enriched ontology definitions "
            "using the xAI Grok API."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N classes. "
            "Useful for testing."
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Discard previous candidate-generation "
            "progress and start over."
        ),
    )

    return parser.parse_args()


# ============================================================
# CREATE XAI CLIENT
# ============================================================

def create_client():

    api_key = os.getenv(
        "XAI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "\nXAI_API_KEY is not set.\n\n"
            "In PowerShell run:\n\n"
            '$env:XAI_API_KEY="your-xai-key"\n'
        )

    return OpenAI(
        api_key=api_key,
        base_url=XAI_BASE_URL,
    )


# ============================================================
# LOAD ENRICHED DEFINITIONS
# ============================================================

def load_definitions():

    if not DEFINITIONS_FILE.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{DEFINITIONS_FILE}\n\n"
            "Run preprocess_definitions_llm.py first."
        )

    df = pd.read_csv(
        DEFINITIONS_FILE,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = [
        "IRI",
        "label",
        "definition",
        "definition_enriched",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "definitions_enriched.csv "
            "is missing columns: "
            + ", ".join(missing)
        )

    for column in required_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # Only rows that have an IRI and an enriched
    # definition can be used.
    df = df[
        (df["IRI"] != "")
        &
        (df["definition_enriched"] != "")
    ].copy()

    df = df.drop_duplicates(
        subset=["IRI"]
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# LOAD ONTOLOGY MODULE
# ============================================================

def load_module():

    if not MODULE_FILE.exists():

        raise FileNotFoundError(
            f"Could not find:\n{MODULE_FILE}"
        )

    graph = Graph()

    graph.parse(
        MODULE_FILE,
        format="turtle",
    )

    return graph


# ============================================================
# DISCOVER CANDIDATE CLASSES
# ============================================================

def get_candidate_classes(graph):
    """
    Return named classes that may serve as subclasses
    or superclasses.

    This is intentionally broader than checking only:

        ?x rdf:type owl:Class

    because a CCO module may reference a class whose formal
    declaration exists in BFO or another imported CCO module.
    """

    classes = set()

    # --------------------------------------------------------
    # Explicit OWL class declarations
    # --------------------------------------------------------

    classes.update(
        graph.subjects(
            RDF.type,
            OWL.Class,
        )
    )

    # --------------------------------------------------------
    # Explicit RDFS class declarations
    # --------------------------------------------------------

    classes.update(
        graph.subjects(
            RDF.type,
            RDFS.Class,
        )
    )

    # --------------------------------------------------------
    # Named subjects of subclass axioms
    # --------------------------------------------------------

    for subclass in graph.subjects(
        RDFS.subClassOf,
        None,
    ):

        if isinstance(
            subclass,
            URIRef,
        ):

            classes.add(
                subclass
            )

    # --------------------------------------------------------
    # Named objects of subclass axioms
    #
    # This captures referenced BFO / CCO classes whose
    # declarations may reside outside module.ttl.
    # --------------------------------------------------------

    for superclass in graph.objects(
        None,
        RDFS.subClassOf,
    ):

        if isinstance(
            superclass,
            URIRef,
        ):

            classes.add(
                superclass
            )

    return {
        iri
        for iri in classes
        if isinstance(
            iri,
            URIRef,
        )
    }


# ============================================================
# CLASS LABELS
# ============================================================

def get_labels(
    graph,
    classes,
):

    labels = {}

    for iri in classes:

        label_values = list(
            graph.objects(
                iri,
                RDFS.label,
            )
        )

        if label_values:

            labels[iri] = str(
                label_values[0]
            )

        else:

            labels[iri] = local_name(
                iri
            )

    return labels


# ============================================================
# EXISTING NAMED SUBCLASS AXIOMS
# ============================================================

def get_named_subclass_pairs(
    graph,
    classes,
):

    pairs = set()

    for (
        subclass,
        superclass,
    ) in graph.subject_objects(
        RDFS.subClassOf
    ):

        if (
            isinstance(
                subclass,
                URIRef,
            )
            and
            isinstance(
                superclass,
                URIRef,
            )
            and
            subclass in classes
            and
            superclass in classes
        ):

            pairs.add(
                (
                    subclass,
                    superclass,
                )
            )

    return pairs


# ============================================================
# EXISTING HIERARCHY
# ============================================================

def build_parent_map(
    subclass_pairs,
):

    parent_map = {}

    for (
        child,
        parent,
    ) in subclass_pairs:

        parent_map.setdefault(
            child,
            set(),
        ).add(
            parent
        )

    return parent_map


def get_ancestors(
    iri,
    parent_map,
):
    """
    Return every named superclass reachable through the
    existing subclass hierarchy.
    """

    ancestors = set()

    stack = list(
        parent_map.get(
            iri,
            set(),
        )
    )

    while stack:

        current = stack.pop()

        if current in ancestors:
            continue

        ancestors.add(
            current
        )

        stack.extend(
            parent_map.get(
                current,
                set(),
            )
        )

    return ancestors


# ============================================================
# DEFINITION LOOKUP
# ============================================================

def build_definition_lookup(
    definitions_df,
):

    lookup = {}

    for _, row in definitions_df.iterrows():

        iri = URIRef(
            row["IRI"]
        )

        lookup[iri] = (
            row["definition_enriched"]
        )

    return lookup


# ============================================================
# RANK POSSIBLE SUPERCLASSES
# ============================================================

def rank_superclass_options(
    target_iri,
    target_definition,
    classes,
    labels,
    definition_lookup,
    parent_map,
):
    """
    Rank possible superclasses using lightweight lexical
    signals before giving the best candidates to Grok.

    This is candidate retrieval, NOT final classification.
    """

    target_tokens = tokenize(
        target_definition
    )

    target_definition_lower = (
        target_definition
        .lower()
    )

    # Existing direct + indirect superclasses.
    already_entailed = get_ancestors(
        target_iri,
        parent_map,
    )

    scored = []

    for candidate_iri in classes:

        # ----------------------------------------------------
        # Do not create A subclassOf A
        # ----------------------------------------------------

        if candidate_iri == target_iri:
            continue

        # ----------------------------------------------------
        # Do not propose something already directly or
        # indirectly true.
        # ----------------------------------------------------

        if candidate_iri in already_entailed:
            continue

        # ----------------------------------------------------
        # Avoid an obvious cycle.
        #
        # If target is already an ancestor of candidate,
        # adding target subclassOf candidate would create:
        #
        # target -> candidate -> ... -> target
        # ----------------------------------------------------

        candidate_ancestors = (
            get_ancestors(
                candidate_iri,
                parent_map,
            )
        )

        if target_iri in candidate_ancestors:
            continue

        candidate_label = labels.get(
            candidate_iri,
            local_name(
                candidate_iri
            ),
        )

        label_tokens = tokenize(
            candidate_label
        )

        candidate_definition = (
            definition_lookup.get(
                candidate_iri,
                "",
            )
        )

        candidate_definition_tokens = tokenize(
            candidate_definition
        )

        # ----------------------------------------------------
        # SIGNAL 1:
        #
        # Candidate label occurs directly in the target
        # definition.
        #
        # This is useful with genus-differentia definitions.
        # ----------------------------------------------------

        phrase_score = 0.0

        normalized_label = (
            candidate_label
            .strip()
            .lower()
        )

        if (
            normalized_label
            and
            normalized_label
            in target_definition_lower
        ):

            phrase_score = 10.0

        # ----------------------------------------------------
        # SIGNAL 2:
        #
        # Token overlap between superclass label and
        # definition.
        # ----------------------------------------------------

        label_overlap = len(
            target_tokens
            &
            label_tokens
        )

        # ----------------------------------------------------
        # SIGNAL 3:
        #
        # Jaccard similarity between enriched definitions.
        #
        # This is only a retrieval heuristic.
        # mOWL handles semantic scoring later.
        # ----------------------------------------------------

        token_union = (
            target_tokens
            |
            candidate_definition_tokens
        )

        token_intersection = (
            target_tokens
            &
            candidate_definition_tokens
        )

        if token_union:

            definition_similarity = (
                len(
                    token_intersection
                )
                /
                len(
                    token_union
                )
            )

        else:

            definition_similarity = 0.0

        # ----------------------------------------------------
        # Combined retrieval score
        # ----------------------------------------------------

        score = (
            phrase_score
            +
            (2.0 * label_overlap)
            +
            definition_similarity
        )

        scored.append(
            (
                score,
                str(
                    candidate_iri
                ),
                candidate_label,
            )
        )

    scored.sort(
        key=lambda item: (
            -item[0],
            item[2].lower(),
        )
    )

    return scored[
        :CANDIDATE_POOL_SIZE
    ]


# ============================================================
# STRUCTURED OUTPUT SCHEMA
# ============================================================

CANDIDATE_SCHEMA = {

    "type": "object",

    "properties": {

        "candidates": {

            "type": "array",

            "maxItems": (
                MAX_CANDIDATES_PER_CLASS
            ),

            "items": {

                "type": "object",

                "properties": {

                    "superclass_iri": {
                        "type": "string",
                    },

                    "reason": {
                        "type": "string",
                    },
                },

                "required": [
                    "superclass_iri",
                    "reason",
                ],

                "additionalProperties": False,
            },
        },
    },

    "required": [
        "candidates",
    ],

    "additionalProperties": False,
}


# ============================================================
# BUILD LLM PROMPT
# ============================================================

def build_prompt(
    subclass_iri,
    subclass_label,
    enriched_definition,
    superclass_options,
):

    option_lines = []

    for (
        score,
        iri,
        label,
    ) in superclass_options:

        option_lines.append(
            f"- {iri} | {label}"
        )

    options_text = "\n".join(
        option_lines
    )

    return f"""
You are performing candidate axiom discovery for a
Common Core Ontologies module.

This is a CANDIDATE GENERATION stage.

The candidates produced here will later be evaluated using
ontology embeddings and another LLM plausibility score.
Therefore, your job is to identify reasonable hypotheses,
not to make the final decision about whether an axiom should
be added to the ontology.


TARGET CLASS

IRI:
{subclass_iri}

Label:
{subclass_label}

Enriched definition:
{enriched_definition}


POSSIBLE SUPERCLASSES

{options_text}


TASK

Identify plausible missing subclass relationships between
the target class and the supplied possible superclasses.

Each candidate axiom has the logical form:

    TARGET_CLASS rdfs:subClassOf SUPERCLASS


RULES

1. Select ONLY superclass IRIs from the supplied list.

2. Return no more than
   {MAX_CANDIDATES_PER_CLASS} candidates.

3. A candidate should be reasonably supported by the
   enriched definition.

4. The relationship does NOT need to be certain at this
   stage. These candidates will later be evaluated using
   mOWL cosine similarity and another LLM plausibility score.

5. Prefer genus/type relationships indicated directly or
   indirectly by the definition.

6. Consider semantic relationships rather than relying only
   on exact lexical matches.

7. Do not invent classes.

8. Do not invent IRIs.

9. Do not generate object-property restrictions.

10. Do not generate cardinality restrictions.

11. Do not generate negation.

12. Do not generate unions.

13. Do not generate complements.

14. Do not generate disjunctions.

15. If none of the supplied superclasses is reasonably
    plausible, return an empty candidates list.

These are candidate hypotheses, NOT accepted ontology
axioms.

Favor plausible recall over excessive conservatism while
remaining semantically defensible.
""".strip()


# ============================================================
# CALL XAI / GROK
# ============================================================

def generate_for_class(
    client,
    subclass_iri,
    subclass_label,
    enriched_definition,
    superclass_options,
):

    if not superclass_options:

        return []

    prompt = build_prompt(
        subclass_iri=subclass_iri,
        subclass_label=subclass_label,
        enriched_definition=enriched_definition,
        superclass_options=superclass_options,
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = (
                client.responses.create(

                    model=MODEL,

                    input=[
                        {
                            "role": "system",
                            "content": (
                                "You are an ontology engineer "
                                "performing candidate axiom "
                                "discovery. Generate plausible "
                                "hypotheses that can later be "
                                "evaluated by machine-learning "
                                "and semantic filtering."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],

                    text={
                        "format": {
                            "type": (
                                "json_schema"
                            ),
                            "name": (
                                "candidate_axioms"
                            ),
                            "schema": (
                                CANDIDATE_SCHEMA
                            ),
                            "strict": True,
                        }
                    },

                    store=False,
                )
            )

            data = json.loads(
                response.output_text
            )

            return data[
                "candidates"
            ]

        except Exception as error:

            last_error = error

            status = getattr(
                error,
                "status_code",
                None,
            )

            print()

            print(
                f"    xAI attempt "
                f"{attempt}/"
                f"{MAX_RETRIES} failed:"
            )

            print(
                f"    {error}"
            )

            # Authentication, billing, and permission
            # errors should not be retried.
            if status in (
                401,
                402,
                403,
            ):

                raise

            if attempt < MAX_RETRIES:

                print(
                    f"    Retrying in "
                    f"{RETRY_DELAY_SECONDS} "
                    "seconds..."
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

    raise RuntimeError(
        "Unable to generate candidates "
        f"for {subclass_label}"
    ) from last_error


# ============================================================
# LOAD EXISTING CANDIDATE RESULTS
# ============================================================

def load_existing_results(
    fresh,
):

    if (
        fresh
        or not OUTPUT_CSV.exists()
    ):

        return pd.DataFrame(
            columns=AUDIT_COLUMNS
        )

    existing = pd.read_csv(
        OUTPUT_CSV,
        dtype=str,
        keep_default_na=False,
    )

    missing = [
        column
        for column in AUDIT_COLUMNS
        if column not in existing.columns
    ]

    if missing:

        raise ValueError(
            "Existing candidate CSV "
            "is missing columns: "
            + ", ".join(missing)
        )

    return existing[
        AUDIT_COLUMNS
    ].copy()


# ============================================================
# LOAD PROCESSING PROGRESS
# ============================================================

def load_progress(
    fresh,
):
    """
    Progress is stored separately from accepted candidate
    rows so that classes producing ZERO candidates are still
    remembered as completed.
    """

    if (
        fresh
        or not PROGRESS_CSV.exists()
    ):

        return pd.DataFrame(
            columns=PROGRESS_COLUMNS
        )

    progress = pd.read_csv(
        PROGRESS_CSV,
        dtype=str,
        keep_default_na=False,
    )

    missing = [
        column
        for column in PROGRESS_COLUMNS
        if column not in progress.columns
    ]

    if missing:

        raise ValueError(
            "Existing progress CSV "
            "is missing columns: "
            + ", ".join(missing)
        )

    return progress[
        PROGRESS_COLUMNS
    ].copy()


# ============================================================
# SAVE AUDIT RESULTS
# ============================================================

def save_audit(
    rows,
):

    GENERATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(
        rows,
        columns=AUDIT_COLUMNS,
    )

    try:

        df.to_csv(
            OUTPUT_CSV,
            index=False,
            encoding="utf-8",
        )

    except PermissionError as error:

        raise PermissionError(
            f"\nCannot write to:\n"
            f"{OUTPUT_CSV}\n\n"
            "Close the CSV in Excel or another "
            "program and rerun the script."
        ) from error


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_progress(
    rows,
):

    GENERATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(
        rows,
        columns=PROGRESS_COLUMNS,
    )

    try:

        df.to_csv(
            PROGRESS_CSV,
            index=False,
            encoding="utf-8",
        )

    except PermissionError as error:

        raise PermissionError(
            f"\nCannot write to:\n"
            f"{PROGRESS_CSV}\n\n"
            "Close the CSV in Excel or another "
            "program and rerun the script."
        ) from error


# ============================================================
# WRITE OWL 2 EL TURTLE
# ============================================================

def write_candidate_ttl(
    rows,
):
    """
    Write only named-class SubClassOf axioms.

    Every logical candidate therefore has the form:

        A rdfs:subClassOf B

    where A and B are named classes.

    This keeps candidate output inside OWL 2 EL.
    """

    GENERATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph = Graph()

    graph.bind(
        "owl",
        OWL,
    )

    graph.bind(
        "rdfs",
        RDFS,
    )

    ontology_iri = URIRef(
        "urn:project5:candidate-el"
    )

    graph.add(
        (
            ontology_iri,
            RDF.type,
            OWL.Ontology,
        )
    )

    added = set()

    for row in rows:

        subclass = URIRef(
            row["subclass_iri"]
        )

        superclass = URIRef(
            row["superclass_iri"]
        )

        pair = (
            subclass,
            superclass,
        )

        if pair in added:
            continue

        graph.add(
            (
                subclass,
                RDFS.subClassOf,
                superclass,
            )
        )

        added.add(
            pair
        )

    try:

        graph.serialize(
            destination=OUTPUT_TTL,
            format="turtle",
        )

    except PermissionError as error:

        raise PermissionError(
            f"\nCannot write to:\n"
            f"{OUTPUT_TTL}\n\n"
            "Close the file in another program "
            "and rerun the script."
        ) from error

    return len(
        added
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    print("=" * 72)

    print(
        "PROJECT 5 - LLM CANDIDATE AXIOM GENERATION"
    )

    print("=" * 72)

    print()

    print(
        f"Module          : {MODULE_FILE}"
    )

    print(
        f"Definitions     : {DEFINITIONS_FILE}"
    )

    print(
        f"Candidate TTL   : {OUTPUT_TTL}"
    )

    print(
        f"Audit CSV       : {OUTPUT_CSV}"
    )

    print(
        f"Progress CSV    : {PROGRESS_CSV}"
    )

    print(
        f"xAI model       : {MODEL}"
    )

    print(
        f"Candidate pool  : "
        f"{CANDIDATE_POOL_SIZE}"
    )

    print(
        f"Max per class   : "
        f"{MAX_CANDIDATES_PER_CLASS}"
    )

    print()

    # ========================================================
    # LOAD DATA
    # ========================================================

    definitions_df = (
        load_definitions()
    )

    module_graph = (
        load_module()
    )

    # ========================================================
    # DISCOVER CLASSES
    # ========================================================

    classes = (
        get_candidate_classes(
            module_graph
        )
    )

    labels = get_labels(
        module_graph,
        classes,
    )

    existing_pairs = (
        get_named_subclass_pairs(
            module_graph,
            classes,
        )
    )

    parent_map = (
        build_parent_map(
            existing_pairs
        )
    )

    definition_lookup = (
        build_definition_lookup(
            definitions_df
        )
    )

    print(
        f"Enriched definitions       : "
        f"{len(definitions_df)}"
    )

    print(
        f"Candidate named classes    : "
        f"{len(classes)}"
    )

    print(
        f"Existing named subclass "
        f"axioms : "
        f"{len(existing_pairs)}"
    )

    # ========================================================
    # ONLY PROCESS DEFINITIONS WHOSE IRI APPEARS AS A
    # RECOGNIZED NAMED CLASS
    # ========================================================

    definitions_df = definitions_df[
        definitions_df["IRI"].apply(
            lambda value: (
                URIRef(value)
                in classes
            )
        )
    ].copy()

    definitions_df = (
        definitions_df
        .reset_index(
            drop=True
        )
    )

    print(
        f"Eligible target classes    : "
        f"{len(definitions_df)}"
    )

    # ========================================================
    # OPTIONAL TEST LIMIT
    # ========================================================

    if args.limit is not None:

        if args.limit <= 0:

            raise ValueError(
                "--limit must be greater than zero."
            )

        definitions_df = (
            definitions_df
            .head(
                args.limit
            )
            .copy()
        )

        print()

        print(
            "TEST MODE: processing "
            f"{len(definitions_df)} classes."
        )

    print()

    # ========================================================
    # LOAD EXISTING OUTPUT + PROGRESS
    # ========================================================

    existing_results = (
        load_existing_results(
            args.fresh
        )
    )

    progress_df = (
        load_progress(
            args.fresh
        )
    )

    output_rows = (
        existing_results
        .to_dict(
            orient="records"
        )
    )

    progress_rows = (
        progress_df
        .to_dict(
            orient="records"
        )
    )

    completed_classes = set(
        progress_df[
            "subclass_iri"
        ]
    )

    if completed_classes:

        print(
            f"Previously completed classes: "
            f"{len(completed_classes)}"
        )

        print()

    # ========================================================
    # CREATE API CLIENT
    # ========================================================

    client = create_client()

    total = len(
        definitions_df
    )

    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for index, row in definitions_df.iterrows():

        subclass_iri = (
            row["IRI"]
        )

        subclass_label = (
            row["label"]
        )

        if not subclass_label:

            subclass_label = labels.get(
                URIRef(
                    subclass_iri
                ),
                local_name(
                    subclass_iri
                ),
            )

        enriched_definition = (
            row[
                "definition_enriched"
            ]
        )

        # ----------------------------------------------------
        # RESUME SUPPORT
        # ----------------------------------------------------

        if subclass_iri in completed_classes:

            print(
                f"[{index + 1}/{total}] "
                f"Already complete: "
                f"{subclass_label}"
            )

            continue

        print(
            f"[{index + 1}/{total}] "
            f"Processing: "
            f"{subclass_label}"
        )

        target_iri = URIRef(
            subclass_iri
        )

        # ----------------------------------------------------
        # RETRIEVE SUPERCLASS OPTIONS
        # ----------------------------------------------------

        superclass_options = (
            rank_superclass_options(

                target_iri=(
                    target_iri
                ),

                target_definition=(
                    enriched_definition
                ),

                classes=(
                    classes
                ),

                labels=(
                    labels
                ),

                definition_lookup=(
                    definition_lookup
                ),

                parent_map=(
                    parent_map
                ),
            )
        )

        # ----------------------------------------------------
        # DIAGNOSTIC OUTPUT
        #
        # This lets us see whether candidate retrieval is
        # providing Grok with useful superclass options.
        # ----------------------------------------------------

        print(
            "    Top superclass options:"
        )

        for (
            option_number,
            (
                score,
                iri,
                label,
            ),
        ) in enumerate(
            superclass_options[
                :DIAGNOSTIC_OPTIONS_TO_SHOW
            ],
            start=1,
        ):

            print(
                f"      "
                f"{option_number:>2}. "
                f"{score:>7.3f} | "
                f"{label} | "
                f"{iri}"
            )

        # ----------------------------------------------------
        # ASK GROK FOR CANDIDATES
        # ----------------------------------------------------

        candidates = (
            generate_for_class(

                client=client,

                subclass_iri=(
                    subclass_iri
                ),

                subclass_label=(
                    subclass_label
                ),

                enriched_definition=(
                    enriched_definition
                ),

                superclass_options=(
                    superclass_options
                ),
            )
        )

        # Every IRI Grok is permitted to return.
        allowed_iris = {
            iri
            for (
                _,
                iri,
                _,
            ) in superclass_options
        }

        accepted_here = 0

        # ----------------------------------------------------
        # HARD VALIDATION
        # ----------------------------------------------------

        for candidate in candidates:

            superclass_iri = (
                candidate[
                    "superclass_iri"
                ]
                .strip()
            )

            reason = (
                candidate[
                    "reason"
                ]
                .strip()
            )

            # ------------------------------------------------
            # Reject hallucinated / unapproved IRIs
            # ------------------------------------------------

            if (
                superclass_iri
                not in allowed_iris
            ):

                print(
                    "    Rejected unknown "
                    "superclass: "
                    f"{superclass_iri}"
                )

                continue

            superclass = URIRef(
                superclass_iri
            )

            # ------------------------------------------------
            # Reject:
            #
            # A subclassOf A
            # ------------------------------------------------

            if superclass == target_iri:

                print(
                    "    Rejected "
                    "self-subclass candidate."
                )

                continue

            # ------------------------------------------------
            # Reject an already direct/indirect superclass
            # ------------------------------------------------

            if superclass in get_ancestors(
                target_iri,
                parent_map,
            ):

                print(
                    "    Rejected already-entailed "
                    f"superclass: "
                    f"{labels.get(superclass, superclass)}"
                )

                continue

            # ------------------------------------------------
            # Reject obvious cycle:
            #
            # A -> B
            # B -> ... -> A
            # ------------------------------------------------

            if target_iri in get_ancestors(
                superclass,
                parent_map,
            ):

                print(
                    "    Rejected cycle-producing "
                    f"candidate: "
                    f"{labels.get(superclass, superclass)}"
                )

                continue

            # ------------------------------------------------
            # Avoid duplicate candidate rows
            # ------------------------------------------------

            duplicate = any(
                existing[
                    "subclass_iri"
                ]
                == subclass_iri
                and
                existing[
                    "superclass_iri"
                ]
                == superclass_iri
                for existing
                in output_rows
            )

            if duplicate:
                continue

            output_rows.append(
                {
                    "subclass_iri": (
                        subclass_iri
                    ),

                    "subclass_label": (
                        subclass_label
                    ),

                    "superclass_iri": (
                        superclass_iri
                    ),

                    "superclass_label": (
                        labels.get(
                            superclass,
                            local_name(
                                superclass
                            ),
                        )
                    ),

                    "reason": (
                        reason
                    ),
                }
            )

            accepted_here += 1

            print(
                "    Candidate accepted: "
                f"{subclass_label} "
                "⊑ "
                f"{labels.get(superclass, local_name(superclass))}"
            )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Record completion even when the class produces
        # ZERO candidates.
        # ----------------------------------------------------

        progress_rows.append(
            {
                "subclass_iri": (
                    subclass_iri
                ),

                "subclass_label": (
                    subclass_label
                ),

                "status": (
                    "complete"
                ),

                "candidate_count": (
                    str(
                        accepted_here
                    )
                ),
            }
        )

        completed_classes.add(
            subclass_iri
        )

        print(
            f"    Candidate axioms: "
            f"{accepted_here}"
        )

        print()

        # ----------------------------------------------------
        # CHECKPOINT AFTER EVERY CLASS
        # ----------------------------------------------------

        save_audit(
            output_rows
        )

        save_progress(
            progress_rows
        )

        write_candidate_ttl(
            output_rows
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    final_count = (
        write_candidate_ttl(
            output_rows
        )
    )

    save_audit(
        output_rows
    )

    save_progress(
        progress_rows
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    completed_count = len(
        progress_rows
    )

    zero_candidate_count = sum(
        1
        for row in progress_rows
        if str(
            row[
                "candidate_count"
            ]
        ) == "0"
    )

    print("=" * 72)

    print(
        "CANDIDATE GENERATION COMPLETE"
    )

    print("=" * 72)

    print()

    print(
        f"Classes processed : "
        f"{completed_count}"
    )

    print(
        f"Classes with zero "
        f"candidates : "
        f"{zero_candidate_count}"
    )

    print(
        f"Candidate axioms  : "
        f"{final_count}"
    )

    print()

    print(
        f"OWL 2 EL file     : "
        f"{OUTPUT_TTL}"
    )

    print(
        f"Audit CSV         : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Progress CSV      : "
        f"{PROGRESS_CSV}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()