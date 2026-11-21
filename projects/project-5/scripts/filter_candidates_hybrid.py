from pathlib import Path
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS

import mowl

from openai import OpenAI
from pydantic import BaseModel, Field


# ============================================================
# VERSION
# ============================================================

SCRIPT_VERSION = "hybrid-mowl-llm-filter-v2-semantic-floor-reciprocal-guard"


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = PROJECT_ROOT / "generated"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = SRC_DIR / "train.ttl"
VALID_FILE = SRC_DIR / "valid.ttl"

CANDIDATE_FILE = GENERATED_DIR / "candidate_el.ttl"
DEFINITIONS_FILE = DATA_DIR / "definitions_enriched.csv"

ACCEPTED_FILE = GENERATED_DIR / "accepted_el.ttl"

SCORES_FILE = (
    GENERATED_DIR
    / "hybrid_candidate_scores.csv"
)

LLM_CACHE_FILE = (
    GENERATED_DIR
    / "hybrid_llm_cache.jsonl"
)

RECIPROCAL_REVIEW_FILE = (
    GENERATED_DIR
    / "hybrid_reciprocal_review.csv"
)


# ============================================================
# FROZEN V8 MODEL ARTIFACTS
# ============================================================

NATIVE_MODEL_FILE = (
    REPORTS_DIR
    / "elembeddings_v8_native.pt"
)

PROJECTION_FILE = (
    REPORTS_DIR
    / "mowl_v8_residual_projection_best.pt"
)

BEST_CONFIG_FILE = (
    REPORTS_DIR
    / "mowl_v8_best_config.json"
)

V8_VALIDATION_FILE = (
    REPORTS_DIR
    / "mowl_v8_best_validation_pairs.csv"
)


# ============================================================
# MOWL CONFIG
#
# Must match train_mowl.py v8.
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

EMBED_DIM = 50

MARGIN = 0.1
REG_NORM = 1.0

NATIVE_LR = 0.001
BATCH_SIZE = 128


# ============================================================
# HYBRID CONFIG
# ============================================================

DEFAULT_COSINE_WEIGHT = 0.70
DEFAULT_LLM_WEIGHT = 0.30

DEFAULT_THRESHOLD = 0.60

# NEW IN V2:
#
# Even if mOWL strongly favors a candidate, Grok must rate
# the relationship at least semantically plausible.
DEFAULT_LLM_FLOOR = 0.50


# ============================================================
# XAI / GROK CONFIG
# ============================================================

XAI_BASE_URL = "https://api.x.ai/v1"

DEFAULT_LLM_MODEL = os.getenv(
    "XAI_MODEL",
    "grok-4.7",
)

MAX_LLM_RETRIES = 3

MAX_CONTEXT_CHARS = 6000


# ============================================================
# STRUCTURED LLM OUTPUT
# ============================================================

class PlausibilityRating(BaseModel):

    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Semantic plausibility of the proposed "
            "OWL subclass axiom."
        ),
    )

    rationale: str = Field(
        description=(
            "Brief explanation of why the subclass "
            "relationship is or is not semantically plausible."
        ),
    )


# ============================================================
# V8 RESIDUAL PROJECTION
# ============================================================

class ResidualProjection(nn.Module):

    def __init__(
        self,
        dimension,
        hidden_size,
        residual_scale,
    ):

        super().__init__()

        self.residual_scale = residual_scale

        self.fc1 = nn.Linear(
            dimension,
            hidden_size,
        )

        self.fc2 = nn.Linear(
            hidden_size,
            dimension,
        )

        self.activation = nn.GELU()

    def forward(
        self,
        embeddings,
    ):

        residual = self.fc1(
            embeddings
        )

        residual = self.activation(
            residual
        )

        residual = self.fc2(
            residual
        )

        output = (
            embeddings
            +
            self.residual_scale
            *
            residual
        )

        return F.normalize(
            output,
            p=2,
            dim=1,
        )


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Filter candidate OWL subclass axioms using "
            "frozen v8 mOWL cosine + Grok semantic "
            "plausibility, an LLM semantic floor, "
            "and reciprocal-edge protection."
        )
    )

    parser.add_argument(
        "--candidate-file",
        type=Path,
        default=CANDIDATE_FILE,
    )

    parser.add_argument(
        "--definitions-file",
        type=Path,
        default=DEFINITIONS_FILE,
    )

    parser.add_argument(
        "--accepted-file",
        type=Path,
        default=ACCEPTED_FILE,
    )

    parser.add_argument(
        "--scores-file",
        type=Path,
        default=SCORES_FILE,
    )

    parser.add_argument(
        "--cache-file",
        type=Path,
        default=LLM_CACHE_FILE,
    )

    parser.add_argument(
        "--reciprocal-review-file",
        type=Path,
        default=RECIPROCAL_REVIEW_FILE,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--llm-floor",
        type=float,
        default=DEFAULT_LLM_FLOOR,
    )

    parser.add_argument(
        "--cosine-weight",
        type=float,
        default=DEFAULT_COSINE_WEIGHT,
    )

    parser.add_argument(
        "--llm-weight",
        type=float,
        default=DEFAULT_LLM_WEIGHT,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
    )

    parser.add_argument(
        "--jvm-memory",
        default="2g",
    )

    parser.add_argument(
        "--force-llm",
        action="store_true",
        help=(
            "Ignore cached LLM scores and query "
            "the LLM again."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


# ============================================================
# BASIC HELPERS
# ============================================================

def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n  {path}"
        )


def load_graph(path):

    require_file(
        path
    )

    graph = Graph()

    graph.parse(
        str(path),
        format="turtle",
    )

    return graph


def copy_namespaces(
    source,
    target,
):

    for prefix, namespace in source.namespaces():

        target.bind(
            prefix,
            namespace,
        )


# ============================================================
# CANDIDATE EXTRACTION
# ============================================================

def extract_candidate_pairs(graph):

    pairs = set()

    skipped = 0

    for child, parent in graph.subject_objects(
        RDFS.subClassOf
    ):

        if not isinstance(
            child,
            URIRef,
        ):

            skipped += 1
            continue

        if not isinstance(
            parent,
            URIRef,
        ):

            skipped += 1
            continue

        if child == parent:

            skipped += 1
            continue

        pairs.add(
            (
                str(child),
                str(parent),
            )
        )

    return (
        sorted(
            pairs
        ),
        skipped,
    )


# ============================================================
# CLASS INDEX
# ============================================================

def normalize_class_index(
    index_dict,
):

    return {
        str(iri):
            int(index)

        for iri, index
        in index_dict.items()
    }


# ============================================================
# SAFE TORCH LOAD
# ============================================================

def safe_torch_load(
    path,
    device,
):

    try:

        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

    except TypeError:

        return torch.load(
            path,
            map_location=device,
        )


# ============================================================
# DEFINITIONS
# ============================================================

def load_definitions(path):

    require_file(
        path
    )

    definitions = {}

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        if not reader.fieldnames:

            raise RuntimeError(
                "definitions_enriched.csv has no header."
            )

        iri_column = None

        for column in reader.fieldnames:

            normalized = (
                column
                .strip()
                .lower()
            )

            if normalized in {
                "iri",
                "class_iri",
                "class iri",
                "uri",
            }:

                iri_column = column
                break

        if iri_column is None:

            raise RuntimeError(
                "Could not find an IRI column in "
                "definitions_enriched.csv."
            )

        for row in reader:

            iri = (
                row.get(
                    iri_column,
                    ""
                )
                or ""
            ).strip()

            if iri:

                definitions[
                    iri
                ] = row

    return definitions


# ============================================================
# LABEL HELPERS
# ============================================================

def graph_label(
    iri,
    *graphs,
):

    node = URIRef(
        iri
    )

    for graph in graphs:

        for label in graph.objects(
            node,
            RDFS.label,
        ):

            return str(
                label
            )

    if "#" in iri:

        return iri.rsplit(
            "#",
            1,
        )[-1]

    return iri.rstrip(
        "/"
    ).rsplit(
        "/",
        1,
    )[-1]


# ============================================================
# LLM DEFINITION CONTEXT
# ============================================================

def definition_context(
    iri,
    definitions,
    fallback_label,
):

    row = definitions.get(
        iri
    )

    if row is None:

        return (
            f"IRI: {iri}\n"
            f"Label: {fallback_label}\n"
            "Definition information: unavailable"
        )

    preferred_order = [
        "label",
        "preferred_label",
        "definition",
        "normalized_definition",
        "enriched_definition",
        "description",
        "synonyms",
        "parent",
        "suggested_parent",
        "notes",
    ]

    normalized_columns = {
        key.strip().lower():
            key

        for key in row.keys()

        if key is not None
    }

    output = [
        f"IRI: {iri}"
    ]

    used_columns = set()

    for preferred in preferred_order:

        actual = normalized_columns.get(
            preferred
        )

        if actual is None:
            continue

        value = (
            row.get(
                actual,
                ""
            )
            or ""
        ).strip()

        if not value:
            continue

        output.append(
            f"{actual}: {value}"
        )

        used_columns.add(
            actual
        )

    for column, value in row.items():

        if column is None:

            continue

        if column in used_columns:

            continue

        normalized = (
            column
            .strip()
            .lower()
        )

        if normalized in {
            "iri",
            "class_iri",
            "class iri",
            "uri",
        }:

            continue

        value = (
            value
            or ""
        ).strip()

        if not value:

            continue

        output.append(
            f"{column}: {value}"
        )

    return "\n".join(
        output
    )[:MAX_CONTEXT_CHARS]


# ============================================================
# LLM CACHE
# ============================================================

def make_cache_key(
    model,
    subclass,
    superclass,
):

    return (
        f"{model}\t"
        f"{subclass}\t"
        f"{superclass}"
    )


def load_llm_cache(path):

    cache = {}

    if not path.exists():

        return cache

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:

                continue

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError:

                continue

            key = record.get(
                "cache_key"
            )

            if key:

                cache[
                    key
                ] = record

    return cache


def append_llm_cache(
    path,
    record,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as handle:

        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )

        handle.write(
            "\n"
        )


# ============================================================
# LLM PROMPTS
# ============================================================

LLM_SYSTEM_PROMPT = """
You are evaluating candidate OWL subclass axioms.

For the proposed axiom

    SubClassOf(Subclass, Superclass)

judge whether the relationship is semantically plausible.

Interpret subclasshood strictly:

    Every instance of the subclass must necessarily also
    be an instance of the superclass.

Do NOT give a high score merely because the concepts are:
- associated,
- related,
- frequently co-occurring,
- part-whole related,
- participants in the same process,
- spatially related,
- causally related,
- or otherwise conceptually similar.

Use only the supplied ontology context.

Scoring:

0.00-0.19:
    Contradictory or category/type mismatch.

0.20-0.39:
    Very weak or probably incorrect.

0.40-0.59:
    Uncertain or only partially supported.

0.60-0.79:
    Plausible subclass relationship.

0.80-0.94:
    Strongly supported subclass relationship.

0.95-1.00:
    Essentially definitional or extremely clear subsumption.

Return a score from 0 to 1 and a concise rationale.
""".strip()


def build_llm_prompt(
    subclass,
    superclass,
    subclass_label,
    superclass_label,
    subclass_context,
    superclass_context,
):

    return f"""
Evaluate this proposed OWL subclass axiom:

SUBCLASS:
{subclass_label}

SUPERCLASS:
{superclass_label}

PROPOSED AXIOM:
<{subclass}> rdfs:subClassOf <{superclass}>

SUBCLASS ONTOLOGY INFORMATION:
{subclass_context}

SUPERCLASS ONTOLOGY INFORMATION:
{superclass_context}

Question:

Is it semantically plausible that EVERY instance of
"{subclass_label}" is necessarily also an instance of
"{superclass_label}"?

Return a plausibility score from 0 to 1.
""".strip()


# ============================================================
# QUERY LLM
# ============================================================

def query_llm(
    client,
    model,
    prompt,
):

    last_error = None

    for attempt in range(
        1,
        MAX_LLM_RETRIES + 1,
    ):

        try:

            response = (
                client
                .responses
                .parse(
                    model=model,

                    input=[
                        {
                            "role":
                                "system",

                            "content":
                                LLM_SYSTEM_PROMPT,
                        },
                        {
                            "role":
                                "user",

                            "content":
                                prompt,
                        },
                    ],

                    text_format=(
                        PlausibilityRating
                    ),

                    store=False,
                )
            )

            parsed = (
                response.output_parsed
            )

            if parsed is None:

                parsed = PlausibilityRating(
                    **json.loads(
                        response.output_text
                    )
                )

            return {
                "score":
                    float(
                        parsed.score
                    ),

                "rationale":
                    parsed.rationale.strip(),
            }

        except Exception as exc:

            last_error = exc

            if attempt < MAX_LLM_RETRIES:

                time.sleep(
                    2 ** (
                        attempt - 1
                    )
                )

    raise RuntimeError(
        "LLM request failed after "
        f"{MAX_LLM_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# REBUILD EXACT V8 DATASET
# ============================================================

def build_v8_dataset(
    train_file,
    jvm_memory,
):

    print(
        "Initializing mOWL JVM..."
    )

    mowl.init_jvm(
        jvm_memory
    )

    from mowl.datasets import (
        PathDataset,
        Dataset,
    )

    from org.semanticweb.owlapi.apibinding import (
        OWLManager,
    )

    from java.util import HashSet

    print(
        "JVM initialized."
    )

    print()

    print(
        "Rebuilding frozen v8 training ontology..."
    )

    path_dataset = PathDataset(
        str(
            train_file.resolve()
        )
    )

    root_ontology = (
        path_dataset.ontology
    )

    import_closure = list(
        root_ontology.getImportsClosure()
    )

    root_axioms = (
        root_ontology.getAxioms()
    )

    manager = (
        OWLManager
        .createOWLOntologyManager()
    )

    data_factory = (
        manager.getOWLDataFactory()
    )

    training_axioms = HashSet()

    training_axioms.addAll(
        root_axioms
    )

    broad_signature = HashSet()

    for ontology in import_closure:

        broad_signature.addAll(
            ontology.getSignature()
        )

    declarations_added = 0

    for entity in broad_signature:

        declaration = (
            data_factory
            .getOWLDeclarationAxiom(
                entity
            )
        )

        if not training_axioms.contains(
            declaration
        ):

            training_axioms.add(
                declaration
            )

            declarations_added += 1

    training_ontology = (
        manager.createOntology(
            training_axioms
        )
    )

    dataset = Dataset(
        training_ontology
    )

    print(
        f"Import closure          : "
        f"{len(import_closure)}"
    )

    print(
        f"Root axioms             : "
        f"{root_axioms.size()}"
    )

    print(
        f"Broad signature entities: "
        f"{broad_signature.size()}"
    )

    print(
        f"Declarations added      : "
        f"{declarations_added}"
    )

    print(
        f"Indexed classes         : "
        f"{len(dataset.classes.as_str)}"
    )

    print(
        "Imported logical axioms: NOT INCLUDED"
    )

    print()

    if len(
        dataset.classes.as_str
    ) < 600:

        raise RuntimeError(
            "Frozen v8 dataset reconstruction failed."
        )

    return dataset


# ============================================================
# LOAD FROZEN V8
# ============================================================

def load_frozen_v8(
    dataset,
):

    from mowl.models import ELEmbeddings

    require_file(
        NATIVE_MODEL_FILE
    )

    require_file(
        PROJECTION_FILE
    )

    require_file(
        BEST_CONFIG_FILE
    )

    print(
        "Loading frozen v8 native ELEmbeddings..."
    )

    model = ELEmbeddings(

        dataset,

        embed_dim=EMBED_DIM,

        margin=MARGIN,

        reg_norm=REG_NORM,

        learning_rate=NATIVE_LR,

        batch_size=BATCH_SIZE,

        model_filepath=None,

        device=DEVICE,
    )

    native_state = safe_torch_load(
        NATIVE_MODEL_FILE,
        DEVICE,
    )

    model.module.load_state_dict(
        native_state
    )

    model.module.eval()

    class_index = normalize_class_index(
        model.class_index_dict
    )

    base_embeddings = (
        model
        .module
        .class_embed
        .weight
        .detach()
        .clone()
        .to(
            DEVICE
        )
    )

    print(
        "Native ELEmbeddings loaded."
    )

    print()

    checkpoint = safe_torch_load(
        PROJECTION_FILE,
        DEVICE,
    )

    with open(
        BEST_CONFIG_FILE,
        "r",
        encoding="utf-8",
    ) as handle:

        config_json = json.load(
            handle
        )

    checkpoint_config = checkpoint.get(
        "best_config",
        {}
    )

    config = {
        **config_json,
        **checkpoint_config,
    }

    if config.get(
        "stage"
    ) != "residual_mlp":

        raise RuntimeError(
            "Selected v8 stage is not residual_mlp."
        )

    hidden_size = int(
        config[
            "hidden_size"
        ]
    )

    residual_scale = float(
        config[
            "residual_scale"
        ]
    )

    projection = ResidualProjection(

        dimension=EMBED_DIM,

        hidden_size=hidden_size,

        residual_scale=residual_scale,

    ).to(
        DEVICE
    )

    projection_state = checkpoint.get(
        "projection_state_dict"
    )

    if projection_state is None:

        raise RuntimeError(
            "No v8 projection state found."
        )

    projection.load_state_dict(
        projection_state
    )

    projection.eval()

    with torch.no_grad():

        projected_embeddings = projection(
            base_embeddings
        )

    print(
        "Frozen v8 residual MLP loaded."
    )

    print(
        f"  hidden size           : "
        f"{hidden_size}"
    )

    print(
        f"  residual scale        : "
        f"{residual_scale}"
    )

    print(
        f"  validation mean       : "
        f"{config_json.get('validation_mean', 'unknown')}"
    )

    print()

    return (
        class_index,
        projected_embeddings,
    )


# ============================================================
# VERIFY V8 RECONSTRUCTION
# ============================================================

def verify_v8_reconstruction(
    class_index,
    projected_embeddings,
):

    if not V8_VALIDATION_FILE.exists():

        print(
            "WARNING: v8 validation CSV not found."
        )

        return

    expected = {}

    with open(
        V8_VALIDATION_FILE,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        for row in reader:

            subclass = (
                row.get(
                    "subclass",
                    ""
                )
                or ""
            ).strip()

            superclass = (
                row.get(
                    "superclass",
                    ""
                )
                or ""
            ).strip()

            if subclass and superclass:

                expected[
                    (
                        subclass,
                        superclass,
                    )
                ] = float(
                    row[
                        "cosine"
                    ]
                )

    differences = []

    for (
        subclass,
        superclass,
    ), expected_cosine in expected.items():

        if (
            subclass not in class_index
            or
            superclass not in class_index
        ):

            raise RuntimeError(
                "Validation class missing during "
                "frozen-model verification."
            )

        actual = float(
            torch.dot(
                projected_embeddings[
                    class_index[
                        subclass
                    ]
                ],
                projected_embeddings[
                    class_index[
                        superclass
                    ]
                ],
            ).item()
        )

        differences.append(
            abs(
                actual
                -
                expected_cosine
            )
        )

    maximum_difference = max(
        differences
    )

    print(
        "Frozen v8 reconstruction check:"
    )

    print(
        f"  validation pairs      : "
        f"{len(differences)}"
    )

    print(
        f"  max cosine difference : "
        f"{maximum_difference:.8f}"
    )

    if maximum_difference > 1e-4:

        raise RuntimeError(
            "Frozen v8 reconstruction FAILED."
        )

    print(
        "  reconstruction        : PASS"
    )

    print()


# ============================================================
# MOWL COSINE
# ============================================================

def mowl_cosine(
    subclass,
    superclass,
    class_index,
    projected_embeddings,
):

    if (
        subclass not in class_index
        or
        superclass not in class_index
    ):

        return None

    return float(
        torch.dot(
            projected_embeddings[
                class_index[
                    subclass
                ]
            ],
            projected_embeddings[
                class_index[
                    superclass
                ]
            ],
        ).item()
    )


# ============================================================
# RECIPROCAL DETECTION
#
# If:
#
#     A subclassOf B
#
# and
#
#     B subclassOf A
#
# both independently pass hybrid + semantic thresholds,
# neither is automatically added.
#
# Both become REVIEW_RECIPROCAL.
# ============================================================

def find_reciprocal_passes(
    rows,
):

    preliminary = {
        (
            row[
                "subclass"
            ],
            row[
                "superclass"
            ]
        )

        for row in rows

        if row[
            "preliminary_pass"
        ]
    }

    reciprocal_pairs = set()

    for child, parent in preliminary:

        reverse = (
            parent,
            child,
        )

        if reverse not in preliminary:

            continue

        canonical = tuple(
            sorted(
                [
                    child,
                    parent,
                ]
            )
        )

        reciprocal_pairs.add(
            canonical
        )

    return reciprocal_pairs


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    GENERATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        TRAIN_FILE,
        VALID_FILE,
        args.candidate_file,
        args.definitions_file,
    ]:

        require_file(
            path
        )

    if not (
        0.0
        <=
        args.threshold
        <=
        1.0
    ):

        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    if not (
        0.0
        <=
        args.llm_floor
        <=
        1.0
    ):

        raise ValueError(
            "--llm-floor must be between 0 and 1."
        )

    weight_sum = (
        args.cosine_weight
        +
        args.llm_weight
    )

    if weight_sum <= 0:

        raise ValueError(
            "Hybrid weights must sum to a "
            "positive number."
        )

    cosine_weight = (
        args.cosine_weight
        /
        weight_sum
    )

    llm_weight = (
        args.llm_weight
        /
        weight_sum
    )

    print(
        f"HYBRID_FILTER_VERSION  : "
        f"{SCRIPT_VERSION}"
    )

    print()

    print("=" * 78)

    print(
        "PROJECT 5 - HYBRID MOWL + LLM FILTER V2"
    )

    print("=" * 78)

    print()

    print(
        f"Candidate ontology      : "
        f"{args.candidate_file}"
    )

    print(
        f"Accepted ontology       : "
        f"{args.accepted_file}"
    )

    print()

    print(
        f"mOWL weight             : "
        f"{cosine_weight:.2f}"
    )

    print(
        f"LLM weight              : "
        f"{llm_weight:.2f}"
    )

    print(
        f"Hybrid threshold        : "
        f"{args.threshold:.2f}"
    )

    print(
        f"LLM semantic floor      : "
        f"{args.llm_floor:.2f}"
    )

    print(
        "Reciprocal guard        : ENABLED"
    )

    print()

    # ========================================================
    # API CLIENT
    #
    # Even when all values are cached, retain the client setup.
    # ========================================================

    api_key = os.getenv(
        "XAI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "XAI_API_KEY is not set."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=XAI_BASE_URL,
    )

    # ========================================================
    # INPUT DATA
    # ========================================================

    candidate_graph = load_graph(
        args.candidate_file
    )

    train_graph = load_graph(
        TRAIN_FILE
    )

    valid_graph = load_graph(
        VALID_FILE
    )

    (
        candidate_pairs,
        skipped,
    ) = extract_candidate_pairs(
        candidate_graph
    )

    if args.limit is not None:

        candidate_pairs = (
            candidate_pairs[
                :args.limit
            ]
        )

    print(
        f"Named candidate pairs   : "
        f"{len(candidate_pairs)}"
    )

    print(
        f"Skipped candidates      : "
        f"{skipped}"
    )

    definitions = load_definitions(
        args.definitions_file
    )

    print(
        f"Definition records      : "
        f"{len(definitions)}"
    )

    print()

    # ========================================================
    # LOAD V8
    # ========================================================

    dataset = build_v8_dataset(
        TRAIN_FILE,
        args.jvm_memory,
    )

    (
        class_index,
        projected_embeddings,
    ) = load_frozen_v8(
        dataset
    )

    verify_v8_reconstruction(
        class_index,
        projected_embeddings,
    )

    # ========================================================
    # CACHE
    # ========================================================

    llm_cache = load_llm_cache(
        args.cache_file
    )

    print(
        f"Cached LLM judgments    : "
        f"{len(llm_cache)}"
    )

    print()

    # ========================================================
    # PASS 1:
    #
    # mOWL + LLM scoring and preliminary decision
    # ========================================================

    results = []

    cache_hits = 0
    llm_requests = 0
    llm_failures = 0
    missing_mowl = 0

    total_candidates = len(
        candidate_pairs
    )

    print("=" * 78)

    print(
        "PASS 1 - HYBRID + SEMANTIC FLOOR"
    )

    print("=" * 78)

    print()

    for index, (
        subclass,
        superclass,
    ) in enumerate(
        candidate_pairs,
        start=1,
    ):

        subclass_label = graph_label(
            subclass,
            candidate_graph,
            train_graph,
            valid_graph,
        )

        superclass_label = graph_label(
            superclass,
            candidate_graph,
            train_graph,
            valid_graph,
        )

        cosine = mowl_cosine(
            subclass,
            superclass,
            class_index,
            projected_embeddings,
        )

        if cosine is None:

            missing_mowl += 1

            results.append(
                {
                    "subclass":
                        subclass,

                    "subclass_label":
                        subclass_label,

                    "superclass":
                        superclass,

                    "superclass_label":
                        superclass_label,

                    "mowl_cosine":
                        "",

                    "llm_score":
                        "",

                    "hybrid_score":
                        "",

                    "threshold":
                        args.threshold,

                    "llm_floor":
                        args.llm_floor,

                    "preliminary_pass":
                        False,

                    "decision":
                        "REJECT",

                    "llm_source":
                        "not_queried",

                    "llm_rationale":
                        "",

                    "error":
                        "missing_mowl_entity",
                }
            )

            continue

        cache_key = make_cache_key(
            args.model,
            subclass,
            superclass,
        )

        llm_score = None
        llm_rationale = ""
        llm_source = ""
        error = ""

        if (
            not args.force_llm
            and
            cache_key in llm_cache
        ):

            cached = llm_cache[
                cache_key
            ]

            llm_score = float(
                cached[
                    "score"
                ]
            )

            llm_rationale = (
                cached.get(
                    "rationale",
                    ""
                )
            )

            llm_source = "cache"

            cache_hits += 1

        else:

            subclass_context = (
                definition_context(
                    subclass,
                    definitions,
                    subclass_label,
                )
            )

            superclass_context = (
                definition_context(
                    superclass,
                    definitions,
                    superclass_label,
                )
            )

            prompt = build_llm_prompt(

                subclass,
                superclass,

                subclass_label,
                superclass_label,

                subclass_context,
                superclass_context,
            )

            try:

                rating = query_llm(
                    client,
                    args.model,
                    prompt,
                )

                llm_score = (
                    rating[
                        "score"
                    ]
                )

                llm_rationale = (
                    rating[
                        "rationale"
                    ]
                )

                llm_source = "api"

                llm_requests += 1

                cache_record = {
                    "cache_key":
                        cache_key,

                    "model":
                        args.model,

                    "subclass":
                        subclass,

                    "superclass":
                        superclass,

                    "score":
                        llm_score,

                    "rationale":
                        llm_rationale,
                }

                append_llm_cache(
                    args.cache_file,
                    cache_record,
                )

                llm_cache[
                    cache_key
                ] = cache_record

                if args.sleep > 0:

                    time.sleep(
                        args.sleep
                    )

            except Exception as exc:

                llm_failures += 1

                error = str(
                    exc
                )

                results.append(
                    {
                        "subclass":
                            subclass,

                        "subclass_label":
                            subclass_label,

                        "superclass":
                            superclass,

                        "superclass_label":
                            superclass_label,

                        "mowl_cosine":
                            cosine,

                        "llm_score":
                            "",

                        "hybrid_score":
                            "",

                        "threshold":
                            args.threshold,

                        "llm_floor":
                            args.llm_floor,

                        "preliminary_pass":
                            False,

                        "decision":
                            "REJECT",

                        "llm_source":
                            "error",

                        "llm_rationale":
                            "",

                        "error":
                            error,
                    }
                )

                continue

        hybrid_score = (
            cosine_weight
            *
            cosine
            +
            llm_weight
            *
            llm_score
        )

        passes_hybrid = (
            hybrid_score
            >=
            args.threshold
        )

        passes_semantic = (
            llm_score
            >=
            args.llm_floor
        )

        preliminary_pass = (
            passes_hybrid
            and
            passes_semantic
        )

        if not passes_hybrid:

            initial_decision = (
                "REJECT_HYBRID"
            )

        elif not passes_semantic:

            initial_decision = (
                "REJECT_LLM_FLOOR"
            )

        else:

            initial_decision = (
                "PRELIMINARY_ACCEPT"
            )

        results.append(
            {
                "subclass":
                    subclass,

                "subclass_label":
                    subclass_label,

                "superclass":
                    superclass,

                "superclass_label":
                    superclass_label,

                "mowl_cosine":
                    cosine,

                "llm_score":
                    llm_score,

                "hybrid_score":
                    hybrid_score,

                "threshold":
                    args.threshold,

                "llm_floor":
                    args.llm_floor,

                "preliminary_pass":
                    preliminary_pass,

                "decision":
                    initial_decision,

                "llm_source":
                    llm_source,

                "llm_rationale":
                    llm_rationale,

                "error":
                    error,
            }
        )

        print(
            f"[{index:03d}/{total_candidates}] "
            f"{initial_decision:<20} "
            f"cos={cosine:+.4f} "
            f"llm={llm_score:.4f} "
            f"hybrid={hybrid_score:.4f}  "
            f"{subclass_label} -> "
            f"{superclass_label}"
        )

    # ========================================================
    # PASS 2:
    #
    # RECIPROCAL GUARD
    # ========================================================

    reciprocal_pairs = find_reciprocal_passes(
        results
    )

    reciprocal_edges = set()

    for left, right in reciprocal_pairs:

        reciprocal_edges.add(
            (
                left,
                right,
            )
        )

        reciprocal_edges.add(
            (
                right,
                left,
            )
        )

    accepted_graph = Graph()

    copy_namespaces(
        candidate_graph,
        accepted_graph,
    )

    accepted_count = 0
    rejected_hybrid = 0
    rejected_llm_floor = 0
    reciprocal_review_count = 0
    generic_rejected = 0

    reciprocal_review_rows = []

    for row in results:

        pair = (
            row[
                "subclass"
            ],
            row[
                "superclass"
            ],
        )

        if (
            row[
                "preliminary_pass"
            ]
            and
            pair in reciprocal_edges
        ):

            row[
                "decision"
            ] = "REVIEW_RECIPROCAL"

            reciprocal_review_count += 1

            reciprocal_review_rows.append(
                row.copy()
            )

            continue

        if row[
            "preliminary_pass"
        ]:

            row[
                "decision"
            ] = "ACCEPT"

            accepted_count += 1

            accepted_graph.add(
                (
                    URIRef(
                        row[
                            "subclass"
                        ]
                    ),
                    RDFS.subClassOf,
                    URIRef(
                        row[
                            "superclass"
                        ]
                    ),
                )
            )

            continue

        if row[
            "decision"
        ] == "REJECT_HYBRID":

            rejected_hybrid += 1

        elif row[
            "decision"
        ] == "REJECT_LLM_FLOOR":

            rejected_llm_floor += 1

        else:

            generic_rejected += 1

    # ========================================================
    # WRITE ACCEPTED TTL
    # ========================================================

    accepted_graph.serialize(
        destination=str(
            args.accepted_file
        ),
        format="turtle",
    )

    # ========================================================
    # WRITE FULL AUDIT CSV
    # ========================================================

    fields = [
        "subclass",
        "subclass_label",
        "superclass",
        "superclass_label",
        "mowl_cosine",
        "llm_score",
        "hybrid_score",
        "threshold",
        "llm_floor",
        "preliminary_pass",
        "decision",
        "llm_source",
        "llm_rationale",
        "error",
    ]

    with open(
        args.scores_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ========================================================
    # WRITE RECIPROCAL REVIEW CSV
    # ========================================================

    with open(
        args.reciprocal_review_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            reciprocal_review_rows
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    hybrid_scores = [
        float(
            row[
                "hybrid_score"
            ]
        )

        for row in results

        if row[
            "hybrid_score"
        ] != ""
    ]

    llm_scores = [
        float(
            row[
                "llm_score"
            ]
        )

        for row in results

        if row[
            "llm_score"
        ] != ""
    ]

    cosine_scores = [
        float(
            row[
                "mowl_cosine"
            ]
        )

        for row in results

        if row[
            "mowl_cosine"
        ] != ""
    ]

    print()

    print("=" * 78)

    print(
        "HYBRID FILTER V2 RESULTS"
    )

    print("=" * 78)

    print()

    print(
        f"Candidate pairs         : "
        f"{total_candidates}"
    )

    print(
        f"Fully scored pairs      : "
        f"{len(hybrid_scores)}"
    )

    print(
        f"Accepted axioms         : "
        f"{accepted_count}"
    )

    print(
        f"Rejected by hybrid      : "
        f"{rejected_hybrid}"
    )

    print(
        f"Rejected by LLM floor   : "
        f"{rejected_llm_floor}"
    )

    print(
        f"Reciprocal review edges : "
        f"{reciprocal_review_count}"
    )

    print(
        f"Reciprocal pairs        : "
        f"{len(reciprocal_pairs)}"
    )

    print(
        f"Other rejected/errors   : "
        f"{generic_rejected}"
    )

    print()

    print(
        f"Missing mOWL entities   : "
        f"{missing_mowl}"
    )

    print(
        f"LLM cache hits          : "
        f"{cache_hits}"
    )

    print(
        f"New LLM API requests    : "
        f"{llm_requests}"
    )

    print(
        f"LLM failures            : "
        f"{llm_failures}"
    )

    print()

    if cosine_scores:

        print(
            f"Mean mOWL cosine        : "
            f"{np.mean(cosine_scores):.4f}"
        )

    if llm_scores:

        print(
            f"Mean LLM score          : "
            f"{np.mean(llm_scores):.4f}"
        )

    if hybrid_scores:

        print(
            f"Mean hybrid score       : "
            f"{np.mean(hybrid_scores):.4f}"
        )

    print()

    print(
        f"Hybrid formula          : "
        f"{cosine_weight:.2f} × cosine "
        f"+ {llm_weight:.2f} × LLM"
    )

    print(
        f"Hybrid threshold        : "
        f"{args.threshold:.2f}"
    )

    print(
        f"LLM semantic floor      : "
        f"{args.llm_floor:.2f}"
    )

    print(
        "Reciprocal guard        : ENABLED"
    )

    print()

    if accepted_count > 0:

        print(
            "Hybrid acceptance       : PASS"
        )

    else:

        print(
            "Hybrid acceptance       : WARNING"
        )

    print()

    print(
        "Files written:"
    )

    print(
        f"  Accepted axioms:\n"
        f"    {args.accepted_file}"
    )

    print()

    print(
        f"  Full audit:\n"
        f"    {args.scores_file}"
    )

    print()

    print(
        f"  Reciprocal review:\n"
        f"    {args.reciprocal_review_file}"
    )

    print()

    print(
        f"  LLM cache:\n"
        f"    {args.cache_file}"
    )


if __name__ == "__main__":

    main()