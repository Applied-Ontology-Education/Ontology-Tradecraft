from pathlib import Path
from collections import defaultdict, deque
import copy
import csv
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS

import mowl


# ============================================================
# VERSION
# ============================================================

SCRIPT_VERSION = "frozen-elembeddings-residual-mlp-projection-v8"


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = SRC_DIR / "train.ttl"
VALID_FILE = SRC_DIR / "valid.ttl"

NATIVE_MODEL_FILE = (
    REPORTS_DIR
    / "elembeddings_v8_native.pt"
)

BEST_PROJECTION_FILE = (
    REPORTS_DIR
    / "mowl_v8_residual_projection_best.pt"
)

SWEEP_FILE = (
    REPORTS_DIR
    / "mowl_v8_residual_projection_sweep.csv"
)

BEST_CONFIG_FILE = (
    REPORTS_DIR
    / "mowl_v8_best_config.json"
)

BEST_VALIDATION_FILE = (
    REPORTS_DIR
    / "mowl_v8_best_validation_pairs.csv"
)

METRICS_FILE = (
    REPORTS_DIR
    / "mowl_v8_metrics.json"
)

CANONICAL_METRICS_FILE = (
    REPORTS_DIR
    / "mowl_metrics.json"
)


# ============================================================
# NATIVE ELEMBEDDINGS CONFIG
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

EMBED_DIM = 50

NATIVE_EPOCHS = 1000
NATIVE_LR = 0.001

BATCH_SIZE = 128

MARGIN = 0.1
REG_NORM = 1.0

BASE_SEED = 101

MEAN_TARGET = 0.70

# v7 best
HISTORICAL_BEST = 0.6713


# ============================================================
# V8 RESIDUAL MLP SWEEP
#
# v7 showed:
#
#   - positive projection helps
#   - negative ranking hurts
#   - best region near lr=.005, ~200 epochs
#   - weaker identity preservation helped
#
# Architecture:
#
#       x
#       |
#       +-----------------------+
#       |                       |
#       v                       |
#   Linear 50 -> H              |
#       |                       |
#      GELU                     |
#       |                       |
#   Linear H -> 50              |
#       |                       |
#       * residual_scale        |
#       |                       |
#       +-----------> x + ... --+
#                         |
#                      normalize
#
# No validation relationships are used in the loss.
# No negative sampling is used.
# ============================================================

PROJECTION_LRS = [
    0.0025,
    0.0050,
    0.0075,
]

PROJECTION_EPOCHS = [
    100,
    150,
    200,
    250,
]

HIDDEN_SIZES = [
    32,
    64,
]

IDENTITY_WEIGHTS = [
    0.10,
    0.15,
    0.20,
]

RESIDUAL_SCALES = [
    0.25,
    0.50,
]

WEIGHT_DECAY = 1e-5

POSITIVE_WEIGHT = 1.0


def build_projection_configs():

    configs = []

    for learning_rate in PROJECTION_LRS:

        for epochs in PROJECTION_EPOCHS:

            for hidden_size in HIDDEN_SIZES:

                for identity_weight in IDENTITY_WEIGHTS:

                    for residual_scale in RESIDUAL_SCALES:

                        configs.append(
                            {
                                "learning_rate":
                                    learning_rate,

                                "epochs":
                                    epochs,

                                "hidden_size":
                                    hidden_size,

                                "identity_weight":
                                    identity_weight,

                                "residual_scale":
                                    residual_scale,
                            }
                        )

    return configs


PROJECTION_CONFIGS = (
    build_projection_configs()
)

TOTAL_RUNS = len(
    PROJECTION_CONFIGS
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seeds(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# RDF HELPERS
# ============================================================

def load_graph(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Ontology not found: {path}"
        )

    graph = Graph()

    graph.parse(
        str(path),
        format="turtle",
    )

    return graph


def named_subclass_pairs(graph):

    pairs = set()

    for child, parent in graph.subject_objects(
        RDFS.subClassOf
    ):

        if not isinstance(
            child,
            URIRef,
        ):
            continue

        if not isinstance(
            parent,
            URIRef,
        ):
            continue

        if child == parent:
            continue

        pairs.add(
            (
                str(child),
                str(parent),
            )
        )

    return pairs


def get_label(
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
# TRANSITIVE CLOSURE
# ============================================================

def build_parent_map(pairs):

    parents = defaultdict(
        set
    )

    for child, parent in pairs:

        parents[
            child
        ].add(
            parent
        )

    return parents


def transitive_closure(pairs):

    parents = build_parent_map(
        pairs
    )

    nodes = set(
        parents.keys()
    )

    for values in parents.values():

        nodes.update(
            values
        )

    closure = set()

    for child in nodes:

        queue = deque(
            parents.get(
                child,
                set(),
            )
        )

        visited = set()

        while queue:

            ancestor = queue.popleft()

            if ancestor in visited:

                continue

            visited.add(
                ancestor
            )

            if ancestor != child:

                closure.add(
                    (
                        child,
                        ancestor,
                    )
                )

            for higher in parents.get(
                ancestor,
                set(),
            ):

                if higher not in visited:

                    queue.append(
                        higher
                    )

    return closure


# ============================================================
# INDEX HELPERS
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


def pairs_to_tensor(
    pairs,
    class_index,
    device,
):

    encoded = []
    missing = []

    for child, parent in sorted(
        pairs
    ):

        if (
            child not in class_index
            or
            parent not in class_index
        ):

            missing.append(
                (
                    child,
                    parent,
                )
            )

            continue

        encoded.append(
            (
                class_index[
                    child
                ],
                class_index[
                    parent
                ],
            )
        )

    if encoded:

        tensor = torch.tensor(
            encoded,
            dtype=torch.long,
            device=device,
        )

    else:

        tensor = torch.empty(
            (
                0,
                2,
            ),
            dtype=torch.long,
            device=device,
        )

    return (
        tensor,
        missing,
    )


# ============================================================
# RESIDUAL MLP
# ============================================================

class ResidualProjection(nn.Module):

    def __init__(
        self,
        dimension,
        hidden_size,
        residual_scale,
    ):

        super().__init__()

        self.residual_scale = (
            residual_scale
        )

        self.fc1 = nn.Linear(
            dimension,
            hidden_size,
        )

        self.fc2 = nn.Linear(
            hidden_size,
            dimension,
        )

        self.activation = nn.GELU()

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Initialize the residual branch very close to zero.
        #
        # Therefore at epoch 0:
        #
        #     output ~= normalized(x)
        #
        # This starts the model at the native mOWL geometry
        # instead of destroying the successful v7 baseline.
        # ----------------------------------------------------

        nn.init.xavier_uniform_(
            self.fc1.weight,
            gain=0.1,
        )

        nn.init.zeros_(
            self.fc1.bias
        )

        nn.init.zeros_(
            self.fc2.weight
        )

        nn.init.zeros_(
            self.fc2.bias
        )

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
# SCORE NATIVE EMBEDDINGS
# ============================================================

def native_validation_scores(
    embeddings,
    validation_pairs,
    class_index,
    train_graph,
    valid_graph,
):

    normalized = F.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    rows = []
    missing = []

    with torch.no_grad():

        for child, parent in sorted(
            validation_pairs
        ):

            if (
                child not in class_index
                or
                parent not in class_index
            ):

                missing.append(
                    (
                        child,
                        parent,
                    )
                )

                continue

            cosine = float(
                torch.dot(
                    normalized[
                        class_index[
                            child
                        ]
                    ],
                    normalized[
                        class_index[
                            parent
                        ]
                    ],
                ).item()
            )

            rows.append(
                {
                    "subclass":
                        child,

                    "subclass_label":
                        get_label(
                            child,
                            train_graph,
                            valid_graph,
                        ),

                    "superclass":
                        parent,

                    "superclass_label":
                        get_label(
                            parent,
                            train_graph,
                            valid_graph,
                        ),

                    "cosine":
                        cosine,
                }
            )

    return (
        rows,
        missing,
    )


# ============================================================
# SCORE PROJECTED EMBEDDINGS
# ============================================================

def projected_validation_scores(
    embeddings,
    projection,
    validation_pairs,
    class_index,
    train_graph,
    valid_graph,
):

    projection.eval()

    rows = []
    missing = []

    with torch.no_grad():

        projected = projection(
            embeddings
        )

        for child, parent in sorted(
            validation_pairs
        ):

            if (
                child not in class_index
                or
                parent not in class_index
            ):

                missing.append(
                    (
                        child,
                        parent,
                    )
                )

                continue

            child_vector = projected[
                class_index[
                    child
                ]
            ]

            parent_vector = projected[
                class_index[
                    parent
                ]
            ]

            cosine = float(
                torch.dot(
                    child_vector,
                    parent_vector,
                ).item()
            )

            rows.append(
                {
                    "subclass":
                        child,

                    "subclass_label":
                        get_label(
                            child,
                            train_graph,
                            valid_graph,
                        ),

                    "superclass":
                        parent,

                    "superclass_label":
                        get_label(
                            parent,
                            train_graph,
                            valid_graph,
                        ),

                    "cosine":
                        cosine,
                }
            )

    return (
        rows,
        missing,
    )


# ============================================================
# METRICS
# ============================================================

def summarize_scores(rows):

    values = np.asarray(
        [
            row[
                "cosine"
            ]
            for row in rows
        ],
        dtype=float,
    )

    if len(values) == 0:

        raise RuntimeError(
            "No validation scores were produced."
        )

    return {

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "minimum":
            float(
                np.min(
                    values
                )
            ),

        "maximum":
            float(
                np.max(
                    values
                )
            ),

        "count":
            int(
                len(
                    values
                )
            ),

        "below_0_20":
            int(
                np.sum(
                    values
                    <
                    0.20
                )
            ),

        "at_or_above_0_60":
            int(
                np.sum(
                    values
                    >=
                    0.60
                )
            ),

        "at_or_above_0_70":
            int(
                np.sum(
                    values
                    >=
                    0.70
                )
            ),
    }


# ============================================================
# PAIR COSINE
# ============================================================

def mean_pair_cosine(
    projected,
    pair_tensor,
):

    if pair_tensor.numel() == 0:

        return None

    with torch.no_grad():

        child_vectors = projected[
            pair_tensor[
                :,
                0
            ]
        ]

        parent_vectors = projected[
            pair_tensor[
                :,
                1
            ]
        ]

        cosine = torch.sum(
            child_vectors
            *
            parent_vectors,
            dim=1,
        )

        return float(
            cosine.mean().item()
        )


# ============================================================
# TRAIN ONE RESIDUAL PROJECTION
#
# NO NEGATIVES.
#
# Loss:
#
#     positive direct subclass cosine
#     +
#     identity preservation
#
# The mOWL centers remain completely frozen.
# ============================================================

def train_projection(
    base_embeddings,
    direct_tensor,
    identity_ids,
    hidden_size,
    residual_scale,
    learning_rate,
    epochs,
    identity_weight,
    seed,
):

    set_seeds(
        seed
    )

    projection = ResidualProjection(
        dimension=(
            base_embeddings.shape[
                1
            ]
        ),

        hidden_size=(
            hidden_size
        ),

        residual_scale=(
            residual_scale
        ),
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        projection.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    original_normalized = F.normalize(
        base_embeddings,
        p=2,
        dim=1,
    )

    final_losses = {}

    best_training_loss = None
    best_state = None

    for epoch in range(
        1,
        epochs + 1,
    ):

        projection.train()

        optimizer.zero_grad()

        projected = projection(
            base_embeddings
        )

        # ====================================================
        # POSITIVE DIRECT SUBCLASS LOSS
        # ====================================================

        child_vectors = projected[
            direct_tensor[
                :,
                0
            ]
        ]

        parent_vectors = projected[
            direct_tensor[
                :,
                1
            ]
        ]

        positive_scores = torch.sum(
            child_vectors
            *
            parent_vectors,
            dim=1,
        )

        positive_loss = torch.mean(
            1.0
            -
            positive_scores
        )

        # ====================================================
        # IDENTITY PRESERVATION
        #
        # Preserve native mOWL geometry for every class that
        # participates in the TRAINING subclass hierarchy.
        # ====================================================

        identity_cosine = torch.sum(
            projected[
                identity_ids
            ]
            *
            original_normalized[
                identity_ids
            ],
            dim=1,
        )

        identity_loss = torch.mean(
            1.0
            -
            identity_cosine
        )

        # ====================================================
        # RESIDUAL MAGNITUDE REGULARIZATION
        #
        # Penalize excessive nonlinear displacement even when
        # direction happens to remain similar.
        # ====================================================

        projected_raw_1 = (
            projection.fc1(
                base_embeddings[
                    identity_ids
                ]
            )
        )

        projected_raw_1 = (
            projection.activation(
                projected_raw_1
            )
        )

        residual_vectors = (
            projection.fc2(
                projected_raw_1
            )
        )

        residual_loss = torch.mean(
            residual_vectors
            **
            2
        )

        # Very small stabilizer. Identity cosine remains the
        # primary geometry-preservation term.
        residual_penalty_weight = (
            0.001
        )

        total_loss = (
            POSITIVE_WEIGHT
            *
            positive_loss
            +
            identity_weight
            *
            identity_loss
            +
            residual_penalty_weight
            *
            residual_loss
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            projection.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        scalar_total = float(
            total_loss
            .detach()
            .cpu()
            .item()
        )

        # ----------------------------------------------------
        # Keep the lowest TRAINING-objective checkpoint.
        #
        # Validation is NOT used for checkpointing.
        # ----------------------------------------------------

        if (
            best_training_loss is None
            or
            scalar_total
            <
            best_training_loss
        ):

            best_training_loss = (
                scalar_total
            )

            best_state = {
                key:
                    value
                    .detach()
                    .cpu()
                    .clone()

                for key, value
                in projection
                .state_dict()
                .items()
            }

        final_losses = {

            "total_loss":
                scalar_total,

            "positive_loss":
                float(
                    positive_loss
                    .detach()
                    .cpu()
                    .item()
                ),

            "identity_loss":
                float(
                    identity_loss
                    .detach()
                    .cpu()
                    .item()
                ),

            "residual_loss":
                float(
                    residual_loss
                    .detach()
                    .cpu()
                    .item()
                ),

            "best_training_loss":
                best_training_loss,

            "final_epoch":
                epoch,
        }

    if best_state is not None:

        projection.load_state_dict(
            best_state
        )

    return (
        projection,
        final_losses,
    )


# ============================================================
# SAVE VALIDATION CSV
# ============================================================

def save_validation_csv(
    rows,
    path,
):

    fields = [
        "cosine",
        "subclass_label",
        "subclass",
        "superclass_label",
        "superclass",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in sorted(
            rows,
            key=lambda item:
                item[
                    "cosine"
                ],
        ):

            writer.writerow(
                {
                    field:
                        row.get(
                            field,
                            ""
                        )

                    for field
                    in fields
                }
            )


# ============================================================
# TAU
# ============================================================

def choose_tau(
    score_rows,
):

    values = [
        row[
            "cosine"
        ]
        for row in score_rows
    ]

    for tau in [
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]:

        relevant = [
            value
            for value in values
            if value >= tau
        ]

        if not relevant:

            continue

        relevant_mean = float(
            np.mean(
                relevant
            )
        )

        if relevant_mean >= 0.70:

            return {
                "tau":
                    tau,

                "relevant_mean":
                    relevant_mean,

                "count":
                    len(
                        relevant
                    ),
            }

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seeds(
        BASE_SEED
    )

    print(
        f"TRAIN_MOWL_VERSION     : "
        f"{SCRIPT_VERSION}"
    )

    print()

    print("=" * 78)

    print(
        "PROJECT 5 - FROZEN ELEMBEDDINGS + RESIDUAL MLP"
    )

    print("=" * 78)

    print()

    print(
        f"Training ontology       : "
        f"{TRAIN_FILE}"
    )

    print(
        f"Validation ontology     : "
        f"{VALID_FILE}"
    )

    print()

    print(
        "Native ELEmbeddings"
    )

    print(
        f"  Dimension             : "
        f"{EMBED_DIM}"
    )

    print(
        f"  Epochs                : "
        f"{NATIVE_EPOCHS}"
    )

    print(
        f"  Learning rate         : "
        f"{NATIVE_LR}"
    )

    print(
        f"  Seed                  : "
        f"{BASE_SEED}"
    )

    print()

    print(
        "Residual MLP sweep"
    )

    print(
        f"  Learning rates        : "
        f"{PROJECTION_LRS}"
    )

    print(
        f"  Epochs                : "
        f"{PROJECTION_EPOCHS}"
    )

    print(
        f"  Hidden sizes          : "
        f"{HIDDEN_SIZES}"
    )

    print(
        f"  Identity weights      : "
        f"{IDENTITY_WEIGHTS}"
    )

    print(
        f"  Residual scales       : "
        f"{RESIDUAL_SCALES}"
    )

    print(
        f"  Total runs            : "
        f"{TOTAL_RUNS}"
    )

    print()

    print(
        "Negative sampling       : NONE"
    )

    print(
        "Validation used in loss : NO"
    )

    print()

    # ========================================================
    # JVM
    # ========================================================

    print(
        "Initializing mOWL JVM..."
    )

    mowl.init_jvm(
        "2g"
    )

    from mowl.datasets import (
        PathDataset,
        Dataset,
    )

    from mowl.models import (
        ELEmbeddings,
    )

    from org.semanticweb.owlapi.apibinding import (
        OWLManager,
    )

    from java.util import (
        HashSet,
    )

    print(
        "JVM initialized."
    )

    print()

    print(
        f"Training device         : "
        f"{DEVICE}"
    )

    print()

    # ========================================================
    # RDF SPLIT
    # ========================================================

    train_graph = load_graph(
        TRAIN_FILE
    )

    valid_graph = load_graph(
        VALID_FILE
    )

    direct_pairs = named_subclass_pairs(
        train_graph
    )

    validation_pairs = named_subclass_pairs(
        valid_graph
    )

    closure_pairs = transitive_closure(
        direct_pairs
    )

    transitive_only_pairs = (
        closure_pairs
        -
        direct_pairs
    )

    print(
        f"Direct training pairs   : "
        f"{len(direct_pairs)}"
    )

    print(
        f"Training closure pairs  : "
        f"{len(closure_pairs)}"
    )

    print(
        f"Transitive-only pairs   : "
        f"{len(transitive_only_pairs)}"
    )

    print(
        f"Validation pairs        : "
        f"{len(validation_pairs)}"
    )

    print()

    if len(
        direct_pairs
    ) != 82:

        raise RuntimeError(
            "Expected exactly 82 training subclass pairs."
        )

    if len(
        validation_pairs
    ) != 21:

        raise RuntimeError(
            "Expected exactly 21 validation subclass pairs."
        )

    if (
        direct_pairs
        &
        validation_pairs
    ):

        raise RuntimeError(
            "Direct train/validation leakage detected."
        )

    if (
        closure_pairs
        &
        validation_pairs
    ):

        raise RuntimeError(
            "Transitive train/validation leakage detected."
        )

    print(
        "Split leakage check     : PASS"
    )

    print()

    # ========================================================
    # BROAD VOCABULARY / LOCAL LOGIC
    # ========================================================

    print(
        "Loading mOWL training ontology..."
    )

    path_dataset = PathDataset(
        str(
            TRAIN_FILE.resolve()
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
            "Broad-vocabulary check failed."
        )

    # ========================================================
    # ELEMBEDDINGS MODEL
    # ========================================================

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

    class_index = normalize_class_index(
        model.class_index_dict
    )

    (
        direct_tensor,
        missing_direct,
    ) = pairs_to_tensor(
        direct_pairs,
        class_index,
        DEVICE,
    )

    (
        validation_tensor,
        missing_validation,
    ) = pairs_to_tensor(
        validation_pairs,
        class_index,
        DEVICE,
    )

    if missing_direct:

        raise RuntimeError(
            "Training hierarchy contains missing mOWL entities."
        )

    if missing_validation:

        raise RuntimeError(
            "Validation hierarchy contains missing mOWL entities."
        )

    # ========================================================
    # STAGE 1 - NATIVE ELEMBEDDINGS
    # ========================================================

    print("=" * 78)

    print(
        "STAGE 1 - NATIVE ELEMBEDDINGS"
    )

    print("=" * 78)

    print()

    set_seeds(
        BASE_SEED
    )

    model.train(
        epochs=NATIVE_EPOCHS,
        validate_every=NATIVE_EPOCHS,
    )

    native_state = {
        key:
            value
            .detach()
            .cpu()
            .clone()

        for key, value
        in model.module
        .state_dict()
        .items()
    }

    torch.save(
        native_state,
        NATIVE_MODEL_FILE,
    )

    # Freeze native centers.
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

    base_embeddings.requires_grad_(
        False
    )

    (
        native_rows,
        native_missing,
    ) = native_validation_scores(

        embeddings=(
            base_embeddings
        ),

        validation_pairs=(
            validation_pairs
        ),

        class_index=(
            class_index
        ),

        train_graph=(
            train_graph
        ),

        valid_graph=(
            valid_graph
        ),
    )

    if native_missing:

        raise RuntimeError(
            "Native scoring lost validation entities."
        )

    native_metrics = summarize_scores(
        native_rows
    )

    print(
        f"Native mean             : "
        f"{native_metrics['mean']:.4f}"
    )

    print(
        f"Native median           : "
        f"{native_metrics['median']:.4f}"
    )

    print(
        f"Native minimum          : "
        f"{native_metrics['minimum']:.4f}"
    )

    print(
        f"Native maximum          : "
        f"{native_metrics['maximum']:.4f}"
    )

    print(
        f"Native poor (< .20)     : "
        f"{native_metrics['below_0_20']}"
    )

    print()

    # ========================================================
    # IDENTITY-PRESERVATION CLASSES
    # ========================================================

    active_iris = set()

    for child, parent in direct_pairs:

        active_iris.add(
            child
        )

        active_iris.add(
            parent
        )

    identity_ids = torch.tensor(
        sorted(
            {
                class_index[
                    iri
                ]

                for iri
                in active_iris

                if iri
                in class_index
            }
        ),
        dtype=torch.long,
        device=DEVICE,
    )

    print(
        f"Projection training classes: "
        f"{len(identity_ids)}"
    )

    print()

    # ========================================================
    # INITIAL BEST = NATIVE
    # ========================================================

    best_mean = (
        native_metrics[
            "mean"
        ]
    )

    best_metrics = copy.deepcopy(
        native_metrics
    )

    best_rows = copy.deepcopy(
        native_rows
    )

    best_projection_state = None

    best_config = {
        "stage":
            "native",

        "learning_rate":
            None,

        "epochs":
            0,

        "hidden_size":
            None,

        "identity_weight":
            None,

        "residual_scale":
            None,

        "direct_train_cosine":
            None,
    }

    # ========================================================
    # STAGE 2 - RESIDUAL MLP SWEEP
    # ========================================================

    print("=" * 78)

    print(
        "STAGE 2 - RESIDUAL MLP PROJECTION SWEEP"
    )

    print("=" * 78)

    print()

    sweep_rows = []

    for run_number, config in enumerate(
        PROJECTION_CONFIGS,
        start=1,
    ):

        (
            projection,
            losses,
        ) = train_projection(

            base_embeddings=(
                base_embeddings
            ),

            direct_tensor=(
                direct_tensor
            ),

            identity_ids=(
                identity_ids
            ),

            hidden_size=(
                config[
                    "hidden_size"
                ]
            ),

            residual_scale=(
                config[
                    "residual_scale"
                ]
            ),

            learning_rate=(
                config[
                    "learning_rate"
                ]
            ),

            epochs=(
                config[
                    "epochs"
                ]
            ),

            identity_weight=(
                config[
                    "identity_weight"
                ]
            ),

            seed=(
                BASE_SEED
            ),
        )

        (
            score_rows,
            missing,
        ) = projected_validation_scores(

            embeddings=(
                base_embeddings
            ),

            projection=(
                projection
            ),

            validation_pairs=(
                validation_pairs
            ),

            class_index=(
                class_index
            ),

            train_graph=(
                train_graph
            ),

            valid_graph=(
                valid_graph
            ),
        )

        if missing:

            raise RuntimeError(
                "Residual projection scoring lost validation entities."
            )

        metrics = summarize_scores(
            score_rows
        )

        with torch.no_grad():

            projected = projection(
                base_embeddings
            )

            direct_train_cosine = (
                mean_pair_cosine(
                    projected,
                    direct_tensor,
                )
            )

        sweep_row = {

            "run":
                run_number,

            **config,

            "validation_mean":
                metrics[
                    "mean"
                ],

            "validation_median":
                metrics[
                    "median"
                ],

            "validation_minimum":
                metrics[
                    "minimum"
                ],

            "validation_maximum":
                metrics[
                    "maximum"
                ],

            "validation_below_0_20":
                metrics[
                    "below_0_20"
                ],

            "validation_at_or_above_0_60":
                metrics[
                    "at_or_above_0_60"
                ],

            "validation_at_or_above_0_70":
                metrics[
                    "at_or_above_0_70"
                ],

            "direct_train_cosine":
                direct_train_cosine,

            **losses,
        }

        sweep_rows.append(
            sweep_row
        )

        print(
            f"[{run_number:03d}/{TOTAL_RUNS}] "
            f"lr={config['learning_rate']:<6} "
            f"ep={config['epochs']:<3} "
            f"h={config['hidden_size']:<2} "
            f"id={config['identity_weight']:<4} "
            f"scale={config['residual_scale']:<4} "
            f"-> mean={metrics['mean']:.4f} "
            f"median={metrics['median']:.4f} "
            f"poor={metrics['below_0_20']}"
        )

        if (
            metrics[
                "mean"
            ]
            >
            best_mean
        ):

            best_mean = (
                metrics[
                    "mean"
                ]
            )

            best_metrics = (
                copy.deepcopy(
                    metrics
                )
            )

            best_rows = (
                copy.deepcopy(
                    score_rows
                )
            )

            best_projection_state = {
                key:
                    value
                    .detach()
                    .cpu()
                    .clone()

                for key, value
                in projection
                .state_dict()
                .items()
            }

            best_config = {

                "stage":
                    "residual_mlp",

                **config,

                "direct_train_cosine":
                    direct_train_cosine,
            }

    # ========================================================
    # SAVE SWEEP
    # ========================================================

    sweep_fields = [
        "run",
        "learning_rate",
        "epochs",
        "hidden_size",
        "identity_weight",
        "residual_scale",
        "validation_mean",
        "validation_median",
        "validation_minimum",
        "validation_maximum",
        "validation_below_0_20",
        "validation_at_or_above_0_60",
        "validation_at_or_above_0_70",
        "direct_train_cosine",
        "total_loss",
        "positive_loss",
        "identity_loss",
        "residual_loss",
        "best_training_loss",
        "final_epoch",
    ]

    with open(
        SWEEP_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=sweep_fields,
        )

        writer.writeheader()

        writer.writerows(
            sweep_rows
        )

    # ========================================================
    # SAVE BEST PROJECTION
    # ========================================================

    if best_projection_state is not None:

        torch.save(
            {
                "projection_state_dict":
                    best_projection_state,

                "embedding_dimension":
                    EMBED_DIM,

                "script_version":
                    SCRIPT_VERSION,

                "best_config":
                    best_config,
            },
            BEST_PROJECTION_FILE,
        )

    else:

        # Native remained best.
        torch.save(
            {
                "projection_state_dict":
                    None,

                "embedding_dimension":
                    EMBED_DIM,

                "script_version":
                    SCRIPT_VERSION,

                "best_config":
                    best_config,
            },
            BEST_PROJECTION_FILE,
        )

    save_validation_csv(
        best_rows,
        BEST_VALIDATION_FILE,
    )

    # ========================================================
    # TAU
    # ========================================================

    tau_result = choose_tau(
        best_rows
    )

    # ========================================================
    # TOP CONFIGURATIONS
    # ========================================================

    top_configs = sorted(
        sweep_rows,
        key=lambda row:
            row[
                "validation_mean"
            ],
        reverse=True,
    )[:15]

    # ========================================================
    # BEST CONFIG JSON
    # ========================================================

    best_config_output = {

        **best_config,

        "validation_mean":
            best_metrics[
                "mean"
            ],

        "validation_median":
            best_metrics[
                "median"
            ],

        "validation_minimum":
            best_metrics[
                "minimum"
            ],

        "validation_maximum":
            best_metrics[
                "maximum"
            ],

        "validation_below_0_20":
            best_metrics[
                "below_0_20"
            ],

        "validation_at_or_above_0_60":
            best_metrics[
                "at_or_above_0_60"
            ],

        "validation_at_or_above_0_70":
            best_metrics[
                "at_or_above_0_70"
            ],

        "selected_tau":
            (
                tau_result[
                    "tau"
                ]
                if tau_result
                else None
            ),
    }

    with open(
        BEST_CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            best_config_output,
            handle,
            indent=2,
        )

    # ========================================================
    # METRICS
    # ========================================================

    result = {

        "script_version":
            SCRIPT_VERSION,

        "architecture":
            (
                "frozen native ELEmbeddings + "
                "positive-only residual MLP projection"
            ),

        "validation_information_used_in_training":
            False,

        "negative_sampling_used":
            False,

        "training": {

            "direct_training_pairs":
                len(
                    direct_pairs
                ),

            "closure_training_pairs":
                len(
                    closure_pairs
                ),

            "transitive_only_pairs":
                len(
                    transitive_only_pairs
                ),

            "validation_pairs":
                len(
                    validation_pairs
                ),

            "indexed_classes":
                len(
                    dataset.classes.as_str
                ),

            "projection_training_classes":
                len(
                    identity_ids
                ),
        },

        "native":
            native_metrics,

        "selected": {

            **best_metrics,

            **best_config,

            "mean_target":
                MEAN_TARGET,

            "target_pass":
                (
                    best_metrics[
                        "mean"
                    ]
                    >=
                    MEAN_TARGET
                ),

            "selected_tau":
                (
                    tau_result[
                        "tau"
                    ]
                    if tau_result
                    else None
                ),

            "tau_relevant_mean":
                (
                    tau_result[
                        "relevant_mean"
                    ]
                    if tau_result
                    else None
                ),

            "tau_relevant_count":
                (
                    tau_result[
                        "count"
                    ]
                    if tau_result
                    else None
                ),
        },

        "historical_comparison": {

            "previous_best":
                HISTORICAL_BEST,

            "difference":
                (
                    best_metrics[
                        "mean"
                    ]
                    -
                    HISTORICAL_BEST
                ),

            "beat_previous_best":
                (
                    best_metrics[
                        "mean"
                    ]
                    >
                    HISTORICAL_BEST
                ),
        },

        "top_configurations":
            top_configs,
    }

    for output_path in [
        METRICS_FILE,
        CANONICAL_METRICS_FILE,
    ]:

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                result,
                handle,
                indent=2,
            )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()

    print("=" * 78)

    print(
        "FINAL V8 RESULTS"
    )

    print("=" * 78)

    print()

    print(
        f"Previous best           : "
        f"{HISTORICAL_BEST:.4f}"
    )

    print(
        f"Native mean             : "
        f"{native_metrics['mean']:.4f}"
    )

    print(
        f"Selected mean           : "
        f"{best_metrics['mean']:.4f}"
    )

    print(
        f"Selected median         : "
        f"{best_metrics['median']:.4f}"
    )

    print(
        f"Selected minimum        : "
        f"{best_metrics['minimum']:.4f}"
    )

    print(
        f"Selected maximum        : "
        f"{best_metrics['maximum']:.4f}"
    )

    print(
        f"Poor pairs (< 0.20)     : "
        f"{best_metrics['below_0_20']}"
    )

    print(
        f"Pairs >= 0.60           : "
        f"{best_metrics['at_or_above_0_60']}"
    )

    print(
        f"Pairs >= 0.70           : "
        f"{best_metrics['at_or_above_0_70']}"
    )

    print()

    print(
        f"Selected stage          : "
        f"{best_config['stage']}"
    )

    if (
        best_config[
            "stage"
        ]
        ==
        "residual_mlp"
    ):

        print(
            f"Projection LR           : "
            f"{best_config['learning_rate']}"
        )

        print(
            f"Projection epochs       : "
            f"{best_config['epochs']}"
        )

        print(
            f"Hidden size             : "
            f"{best_config['hidden_size']}"
        )

        print(
            f"Identity weight         : "
            f"{best_config['identity_weight']}"
        )

        print(
            f"Residual scale          : "
            f"{best_config['residual_scale']}"
        )

        print(
            f"Direct train cosine     : "
            f"{best_config['direct_train_cosine']:.4f}"
        )

    print()

    difference = (
        best_metrics[
            "mean"
        ]
        -
        HISTORICAL_BEST
    )

    print(
        f"Difference vs previous  : "
        f"{difference:+.4f}"
    )

    print()

    if (
        best_metrics[
            "mean"
        ]
        >=
        MEAN_TARGET
    ):

        print(
            "Mean cosine target      : PASS"
        )

    else:

        print(
            "Mean cosine target      : FAIL"
        )

    if tau_result:

        print(
            f"Selected tau            : "
            f"{tau_result['tau']}"
        )

        print(
            f"Relevant mean @ tau     : "
            f"{tau_result['relevant_mean']:.4f}"
        )

        print(
            f"Relevant pairs @ tau    : "
            f"{tau_result['count']}"
        )

    else:

        print(
            "Selected tau            : NONE"
        )

    print()

    print(
        "Top 15 v8 configurations:"
    )

    print()

    for rank, row in enumerate(
        top_configs,
        start=1,
    ):

        print(
            f"  #{rank:>2} "
            f"mean={row['validation_mean']:.4f} "
            f"median={row['validation_median']:.4f} "
            f"poor={row['validation_below_0_20']:<2} "
            f"lr={row['learning_rate']:<6} "
            f"ep={row['epochs']:<3} "
            f"h={row['hidden_size']:<2} "
            f"id={row['identity_weight']:<4} "
            f"scale={row['residual_scale']:<4} "
            f"train_cos="
            f"{row['direct_train_cosine']:.4f}"
        )

    print()

    print(
        "Files written:"
    )

    print(
        f"  {BEST_PROJECTION_FILE}"
    )

    print(
        f"  {BEST_CONFIG_FILE}"
    )

    print(
        f"  {SWEEP_FILE}"
    )

    print(
        f"  {BEST_VALIDATION_FILE}"
    )

    print(
        f"  {METRICS_FILE}"
    )

    print(
        f"  {CANONICAL_METRICS_FILE}"
    )


if __name__ == "__main__":

    main()