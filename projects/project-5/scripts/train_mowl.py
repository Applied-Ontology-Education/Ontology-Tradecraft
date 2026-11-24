from pathlib import Path
import json
import random
import tempfile

import numpy as np
import torch
from rdflib import Graph, OWL, RDFS

# Initialize the Java environment before importing mOWL components.
from mowl import init_jvm

init_jvm("4g")

from mowl.datasets import PathDataset
from mowl.models import ELEmbeddings


# ============================================================
# PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_DIR / "src" / "train.ttl"
VALID_PATH = PROJECT_DIR / "src" / "valid.ttl"

MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_PATH = MODEL_DIR / "quality_ele_model.pt"
METRICS_PATH = REPORT_DIR / "mowl_metrics.json"

EMBEDDING_DIMENSION = 50
EPOCHS = 500
LEARNING_RATE = 0.001

MARGIN = 0.1
REGULARIZATION_NORM = 1.0

RANDOM_SEED = 42

COSINE_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
TARGET_MEAN_COSINE = 0.70

# Number of random unrelated pairs used as a diagnostic baseline.
RANDOM_CONTROL_COUNT = 1000


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seeds():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# TEMPORARY ONTOLOGY COPIES
# ============================================================

def create_import_free_copy(source_path, destination_path):
    """
    Create a temporary copy of an ontology without owl:imports.

    This avoids mOWL attempting to load imported ontology classes
    that are not represented in the project's indexed class set.

    The original train.ttl and valid.ttl files are not modified.
    """

    graph = Graph()
    graph.parse(str(source_path), format="turtle")

    import_count = len(list(graph.triples((None, OWL.imports, None))))
    graph.remove((None, OWL.imports, None))

    graph.serialize(destination_path, format="turtle")

    return {
        "triples": len(graph),
        "imports_removed": import_count,
    }


# ============================================================
# LOAD DATASET AND MODEL
# ============================================================

def load_dataset():
    """
    Load the training and validation ontologies.

    The training ontology contains 92 subclass relationships.
    The validation ontology contains 18 held-out relationships.
    """

    temporary_directory = tempfile.TemporaryDirectory()
    temporary_path = Path(temporary_directory.name)

    temporary_train_path = temporary_path / "train.ttl"
    temporary_valid_path = temporary_path / "valid.ttl"

    train_info = create_import_free_copy(
        TRAIN_PATH,
        temporary_train_path,
    )

    valid_info = create_import_free_copy(
        VALID_PATH,
        temporary_valid_path,
    )

    print("\nTEMPORARY ONTOLOGY COPIES")
    print("---------------------------------------------")
    print(f"Training triples:   {train_info['triples']}")
    print(f"Training imports removed: {train_info['imports_removed']}")
    print(f"Validation triples: {valid_info['triples']}")
    print(f"Validation imports removed: {valid_info['imports_removed']}")

    dataset = PathDataset(
        str(temporary_train_path),
        validation_path=str(temporary_valid_path),
    )

    model = ELEmbeddings(
        dataset,
        embed_dim=EMBEDDING_DIMENSION,
        margin=MARGIN,
        reg_norm=REGULARIZATION_NORM,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device="cpu",
    )

    training_data = model.training_datasets["gci0"][:]
    validation_data = model.validation_datasets["gci0"][:]

    training_count = int(training_data.shape[0])
    validation_count = int(validation_data.shape[0])

    print("\nMOWL DATASET SUMMARY")
    print("---------------------------------------------")
    print(f"Training GCI0 relationships:   {training_count}")
    print(f"Validation GCI0 relationships: {validation_count}")
    print(f"Indexed classes:                {len(model.class_index_dict)}")


    return dataset, model, temporary_directory


# ============================================================
# REGULARIZATION
# ============================================================

def regularization_loss(model):
    """
    Apply the class-center norm regularization used by ELEmbeddings.
    """

    class_weights = model.module.class_embed.weight

    norms = torch.linalg.vector_norm(
        class_weights,
        ord=2,
        dim=1,
    )

    return torch.abs(norms - REGULARIZATION_NORM).mean()


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(model):
    """
    Train ELEmbeddings using the training GCI0 relationships.

    The validation relationships are not used for gradient updates.
    """

    training_data = model.training_datasets["gci0"][:]

    optimizer = torch.optim.Adam(
        model.module.parameters(),
        lr=LEARNING_RATE,
    )

    print("\nMODEL TRAINING")
    print("---------------------------------------------")
    print(f"Training relationships: {training_data.shape[0]}")
    print(f"Embedding dimension:    {EMBEDDING_DIMENSION}")
    print(f"Epochs:                 {EPOCHS}")
    print(f"Learning rate:          {LEARNING_RATE}")

    model.module.train()

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()

        gci_loss = model.module(training_data, "gci0")
        reg_loss = regularization_loss(model)

        loss = gci_loss.mean() + reg_loss   

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:3d}/{EPOCHS} - "
                f"Loss: {loss.item():.6f}"
            )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": model.module.state_dict(),
        "class_iris": list(model.class_index_dict.keys()),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "margin": MARGIN,
        "regularization_norm": REGULARIZATION_NORM,
        "random_seed": RANDOM_SEED,
    }

    torch.save(checkpoint, MODEL_PATH)

    print(f"\nModel checkpoint saved to: {MODEL_PATH}")


# ============================================================
# ONTOLOGY RELATIONSHIP EXTRACTION
# ============================================================

def get_subclass_pairs(ontology_path):
    """
    Extract asserted named-class rdfs:subClassOf relationships.

    Returns pairs as (subclass IRI, superclass IRI) strings.
    """

    graph = Graph()
    graph.parse(str(ontology_path), format="turtle")

    pairs = []

    for subclass, _, superclass in graph.triples(
        (None, RDFS.subClassOf, None)
    ):
        if subclass.startswith("http") and superclass.startswith("http"):
            pairs.append((str(subclass), str(superclass)))

    return pairs


# ============================================================
# EMBEDDING AND COSINE UTILITIES
# ============================================================

def normalize_embedding_dictionary(model):
    """
    Return class embeddings keyed by their IRI strings.
    """

    raw_embeddings = model.class_embeddings

    embeddings = {}

    for iri, vector in raw_embeddings.items():
        embeddings[str(iri)] = np.asarray(vector, dtype=float)

    return embeddings


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    """

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return None

    return float(
        np.dot(vector_a, vector_b) / (norm_a * norm_b)
    )


def score_pairs(pairs, embeddings):
    """
    Calculate cosine similarity for each pair with available embeddings.
    """

    results = []
    missing = []

    for subclass_iri, superclass_iri in pairs:
        if (
            subclass_iri not in embeddings
            or superclass_iri not in embeddings
        ):
            missing.append({
                "subclass_iri": subclass_iri,
                "superclass_iri": superclass_iri,
            })
            continue

        score = cosine_similarity(
            embeddings[subclass_iri],
            embeddings[superclass_iri],
        )

        if score is None:
            missing.append({
                "subclass_iri": subclass_iri,
                "superclass_iri": superclass_iri,
            })
            continue

        results.append({
            "subclass_iri": subclass_iri,
            "superclass_iri": superclass_iri,
            "cosine_similarity": score,
        })

    return results, missing


# ============================================================
# RANDOM UNRELATED-PAIR CONTROLS
# ============================================================

def get_transitive_relationships(train_pairs, valid_pairs):
    """
    Build a directed graph of asserted subclass relationships
    and calculate transitive reachability.

    A pair is considered related if either class is an ancestor
    or descendant of the other in the combined asserted graph.
    """

    children = {}

    for subclass, superclass in train_pairs + valid_pairs:
        children.setdefault(subclass, set()).add(superclass)

    reachability = {}

    for start in children:
        visited = set()
        stack = list(children.get(start, set()))

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            stack.extend(children.get(current, set()))

        reachability[start] = visited

    return reachability


def generate_random_unrelated_pairs(
    embeddings,
    train_pairs,
    valid_pairs,
):
    """
    Generate deterministic random class pairs that are not
    related by an asserted subclass or transitive relationship.

    These are diagnostic controls, not formally declared
    negative examples in the ontology.
    """

    classes = sorted(embeddings.keys())

    reachability = get_transitive_relationships(
        train_pairs,
        valid_pairs,
    )

    candidates = []

    for subclass in classes:
        for superclass in classes:
            if subclass == superclass:
                continue

            subclass_is_ancestor = (
                superclass in reachability.get(subclass, set())
            )

            superclass_is_ancestor = (
                subclass in reachability.get(superclass, set())
            )

            if subclass_is_ancestor or superclass_is_ancestor:
                continue

            candidates.append((subclass, superclass))

    rng = random.Random(RANDOM_SEED)

    sample_size = min(
        RANDOM_CONTROL_COUNT,
        len(candidates),
    )

    if sample_size == 0:
        return []

    return rng.sample(candidates, sample_size)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model):
    """
    Evaluate class-center cosine similarities for:

    1. Training subclass-superclass pairs.
    2. Held-out validation subclass-superclass pairs.
    3. Random unrelated-pair diagnostic controls.

    Also evaluate the required validation cosine thresholds.
    """

    print("\nMODEL EVALUATION")
    print("---------------------------------------------")

        # Evaluate the actual ELEmbeddings GCI0 objective.
    model.module.eval()

    with torch.no_grad():
        training_data = model.training_datasets["gci0"][:]
        validation_data = model.validation_datasets["gci0"][:]

        training_gci0_scores = model.module(training_data, "gci0")
        validation_gci0_scores = model.module(validation_data, "gci0")

        training_el_loss = training_gci0_scores.mean().item()
        validation_el_loss = validation_gci0_scores.mean().item()

        validation_satisfied = (
            validation_gci0_scores <= 0
        ).sum().item()

        validation_total = validation_gci0_scores.numel()

    print("\nGCI0 GEOMETRIC EVALUATION")
    print("---------------------------------------------")
    print(f"Training mean GCI0 loss:   {training_el_loss:.6f}")
    print(f"Validation mean GCI0 loss: {validation_el_loss:.6f}")
    print(
        f"Validation relationships satisfying margin: "
        f"{validation_satisfied}/{validation_total}"
    )
    print(
        f"Validation margin satisfaction: "
        f"{validation_satisfied / validation_total:.2%}"
    )

    embeddings = normalize_embedding_dictionary(model)

    training_pairs = get_subclass_pairs(TRAIN_PATH)
    validation_pairs = get_subclass_pairs(VALID_PATH)

    training_results, training_missing = score_pairs(
        training_pairs,
        embeddings,
    )

    validation_results, validation_missing = score_pairs(
        validation_pairs,
        embeddings,
    )

    random_pairs = generate_random_unrelated_pairs(
        embeddings,
        training_pairs,
        validation_pairs,
    )

    random_results, random_missing = score_pairs(
        random_pairs,
        embeddings,
    )

    def mean_score(results):
        scores = [
            item["cosine_similarity"]
            for item in results
        ]

        return float(np.mean(scores)) if scores else None

    training_mean = mean_score(training_results)
    validation_mean = mean_score(validation_results)
    random_mean = mean_score(random_results)

    print("\nCOSINE SIMILARITY DIAGNOSTICS")
    print("---------------------------------------------")

    print(
        f"Training pairs: {len(training_results)} scored, "
        f"{len(training_missing)} missing"
    )

    print(
        f"Training mean cosine: "
        f"{training_mean if training_mean is not None else 'None'}"
    )

    print(
        f"Validation pairs: {len(validation_results)} scored, "
        f"{len(validation_missing)} missing"
    )

    print(
        f"Validation mean cosine: "
        f"{validation_mean if validation_mean is not None else 'None'}"
    )

    print(
        f"Random unrelated pairs: {len(random_results)} scored, "
        f"{len(random_missing)} missing"
    )

    print(
        f"Random-control mean cosine: "
        f"{random_mean if random_mean is not None else 'None'}"
    )

    if validation_mean is not None and random_mean is not None:
        print(
            f"Validation minus random-control mean: "
            f"{validation_mean - random_mean:.4f}"
        )

    # --------------------------------------------------------
    # Validation threshold analysis
    # --------------------------------------------------------

    threshold_results = []
    selected_threshold = None

    print("\nVALIDATION THRESHOLD ANALYSIS")
    print("---------------------------------------------")
    print(f"Validation pairs: {len(validation_results)}")

    for threshold in COSINE_THRESHOLDS:
        retained = [
            item["cosine_similarity"]
            for item in validation_results
            if item["cosine_similarity"] >= threshold
        ]

        retained_count = len(retained)

        coverage = (
            retained_count / len(validation_results)
            if validation_results
            else 0.0
        )

        retained_mean = (
            float(np.mean(retained))
            if retained
            else None
        )

        threshold_result = {
            "threshold": threshold,
            "retained_pairs": retained_count,
            "coverage": coverage,
            "mean_cosine": retained_mean,
        }

        threshold_results.append(threshold_result)

        print(
            f"Threshold {threshold:.2f}: "
            f"{retained_count}/{len(validation_results)} retained, "
            f"coverage={coverage:.2%}, "
            f"mean cosine={retained_mean}"
        )

        if (
            selected_threshold is None
            and retained_mean is not None
            and retained_mean >= TARGET_MEAN_COSINE
        ):
            selected_threshold = threshold

    print(f"\nSelected threshold: {selected_threshold}")

    # --------------------------------------------------------
    # Individual validation-pair results
    # --------------------------------------------------------

    print("\nINDIVIDUAL VALIDATION PAIR SCORES")
    print("---------------------------------------------")

    for item in validation_results:
        subclass_short = item["subclass_iri"].rsplit("/", 1)[-1]
        superclass_short = item["superclass_iri"].rsplit("/", 1)[-1]

        print(
            f"{item['cosine_similarity']: .4f} "
            f"{subclass_short} -> {superclass_short}"
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    metrics = {
        "training_pair_count": len(training_pairs),
        "training_scored_pairs": len(training_results),
        "training_missing_pairs": len(training_missing),
        "training_mean_cosine": training_mean,
        "training_pair_results": training_results,

        "validation_pair_count": len(validation_pairs),
        "validation_scored_pairs": len(validation_results),
        "validation_missing_pairs": len(validation_missing),
        "validation_mean_cosine": validation_mean,
        "validation_pair_results": validation_results,

        "random_control_pair_count": len(random_pairs),
        "random_control_scored_pairs": len(random_results),
        "random_control_missing_pairs": len(random_missing),
        "random_control_mean_cosine": random_mean,
        "random_control_pair_results": random_results,

        "validation_minus_random_control_mean": (
            validation_mean - random_mean
            if validation_mean is not None and random_mean is not None
            else None
        ),

        "thresholds": threshold_results,
        "target_mean_cosine": TARGET_MEAN_COSINE,
        "selected_threshold": selected_threshold,
    }

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    set_random_seeds()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataset, model, temporary_directory = load_dataset()

    try:
        train_model(model)

        metrics = evaluate_model(model)

        report = {
            "ontology": "Quality Ontology",
            "model": "mOWL ELEmbeddings",
            "configuration": {
                "embedding_dimension": EMBEDDING_DIMENSION,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "margin": MARGIN,
                "regularization_norm": REGULARIZATION_NORM,
                "random_seed": RANDOM_SEED,
            },
            "metrics": metrics,
            "model_checkpoint": str(MODEL_PATH),
        }

        with open(METRICS_PATH, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        print("\nTRAINING AND EVALUATION COMPLETE")
        print("---------------------------------------------")
        print(f"Metrics report: {METRICS_PATH}")
        print(f"Model checkpoint: {MODEL_PATH}")

    finally:
        temporary_directory.cleanup()


if __name__ == "__main__":
    main()
