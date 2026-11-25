import csv
import json
import os
import re
import sys
from pathlib import Path

import torch
from google import genai
from rdflib import Graph, OWL, RDF, RDFS, URIRef


# ============================================================
# PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_DIR / "src" / "train.ttl"
VALID_PATH = PROJECT_DIR / "src" / "valid.ttl"

CHECKPOINT_PATH = PROJECT_DIR / "models" / "quality_ele_model.pt"
CANDIDATE_PATH = PROJECT_DIR / "generated" / "candidate_el.ttl"
DEFINITIONS_PATH = PROJECT_DIR / "data" / "definitions_enriched.csv"

CACHE_PATH = PROJECT_DIR / "generated" / "gemini_candidate_scores.json"
REPORT_PATH = PROJECT_DIR / "reports" / "hybrid_filter_report.json"
ACCEPTED_PATH = PROJECT_DIR / "generated" / "accepted_el.ttl"
METRICS_PATH = PROJECT_DIR / "reports" / "mowl_metrics.json"

GEMINI_MODEL = "gemini-3.5-flash-lite"

MOWL_WEIGHT = 0.70
GEMINI_WEIGHT = 0.30

# The hybrid score uses the RAW MOWL cosine (no rescaling to [0, 1]).
# Rescaling with (cos + 1) / 2 lifts even chance-level similarity
# (random unrelated pairs average ~0.37 cosine) above the threshold,
# so the MOWL component could no longer reject anything.


def load_learned_threshold():
    """
    Read the threshold tau selected by train_mowl.py, i.e. the smallest
    tau in {0.60 ... 0.80} whose validation mean cosine is >= 0.70.
    """
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"MOWL metrics not found: {METRICS_PATH}. Run train_mowl.py first."
        )

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["metrics"]
    tau = metrics.get("selected_threshold")

    if tau is None:
        raise ValueError(
            "train_mowl.py did not select a threshold (validation mean "
            "cosine target not met). Fix training before filtering."
        )

    return float(tau), metrics.get("validation_mean_cosine")


ACCEPTANCE_THRESHOLD, VALIDATION_MEAN_COSINE = load_learned_threshold()


# ============================================================
# IMPORT EXISTING MOWL UTILITIES
# ============================================================

sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from train_mowl import (
    load_dataset,
    normalize_embedding_dictionary,
    cosine_similarity,
)


# ============================================================
# LOAD DEFINITIONS
# ============================================================

def load_definitions():
    definitions = {}

    with open(
        DEFINITIONS_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "IRI",
            "label",
            "original_definition",
            "definition",
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "The enriched definitions CSV is missing required columns."
            )

        for row in reader:
            definitions[row["IRI"]] = row

    return definitions


# ============================================================
# EXTRACT NAMED SUBCLASS CANDIDATES
# ============================================================

def load_candidates():
    graph = Graph()
    graph.parse(CANDIDATE_PATH, format="turtle")

    pairs = []

    for subclass, superclass in graph.subject_objects(RDFS.subClassOf):
        if (
            isinstance(subclass, URIRef)
            and isinstance(superclass, URIRef)
        ):
            pairs.append({
                "subclass_iri": str(subclass),
                "superclass_iri": str(superclass),
            })

    pairs.sort(
        key=lambda pair: (
            pair["subclass_iri"],
            pair["superclass_iri"],
        )
    )

    if not pairs:
        raise ValueError("No named subclass candidates were found.")

    return graph, pairs


# ============================================================
# RESTORE TRAINED MOWL MODEL
# ============================================================

def restore_mowl_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    dataset, model, temporary_directory = load_dataset()

    try:
        saved_classes = checkpoint["class_iris"]
        current_classes = list(model.class_index_dict.keys())

        if saved_classes != current_classes:
            raise ValueError(
                "Checkpoint class ordering does not match "
                "the reconstructed MOWL model."
            )

        model.module.load_state_dict(
            checkpoint["state_dict"],
            strict=True,
        )

        model.module.eval()

        embeddings = normalize_embedding_dictionary(model)

        print("\nMOWL CHECKPOINT RESTORED")
        print("---------------------------------------------")
        print(f"Classes: {len(saved_classes)}")
        print(f"Embedding dimension: {checkpoint['embedding_dimension']}")
        print("Class ordering: exact match")
        print("Training: skipped; saved weights restored")

        return model, embeddings, temporary_directory

    except Exception:
        temporary_directory.cleanup()
        raise


# ============================================================
# SCORE CANDIDATES WITH MOWL
# ============================================================

def score_mowl_candidates(pairs, embeddings):
    results = []

    for pair in pairs:
        subclass = pair["subclass_iri"]
        superclass = pair["superclass_iri"]

        if subclass not in embeddings or superclass not in embeddings:
            raise ValueError(
                "Candidate class missing from MOWL embeddings: "
                f"{subclass} -> {superclass}"
            )

        raw_cosine = cosine_similarity(
            embeddings[subclass],
            embeddings[superclass],
        )

        if raw_cosine is None:
            raise ValueError(
                f"Could not calculate cosine for {subclass} -> {superclass}"
            )

        results.append({
            **pair,
            "mowl_cosine": raw_cosine,
        })

    return results


# ============================================================
# BUILD GEMINI INPUT
# ============================================================

def build_gemini_prompt(scored_pairs, definitions):
    candidate_descriptions = []

    for index, pair in enumerate(scored_pairs):
        subclass = definitions.get(pair["subclass_iri"])
        superclass = definitions.get(pair["superclass_iri"])

        if subclass is None or superclass is None:
            raise ValueError(
                "Missing enriched definition for a candidate class."
            )

        candidate_descriptions.append({
            "index": index,
            "proposed_axiom": (
                f"{subclass['label']} SubClassOf {superclass['label']}"
            ),
            "subclass": {
                "label": subclass["label"],
                "original_definition": subclass["original_definition"],
                "normalized_definition": subclass["definition"],
            },
            "superclass": {
                "label": superclass["label"],
                "original_definition": superclass["original_definition"],
                "normalized_definition": superclass["definition"],
            },
        })

    prompt = f"""
You are evaluating proposed OWL subclass axioms for a quality ontology.

For each proposed axiom, assess whether the meaning of the subclass
is genuinely subsumed by the meaning of the superclass.

Use the definitions as the primary evidence.

Evaluation guidance:
- A subclass must necessarily be an instance of the superclass.
- Similar wording or topical association alone is insufficient.
- Do not assume that a proposed axiom is true merely because it is
  plausible or because the classes are related.
- Distinguish necessary subsumption from typical, contingent, or
  merely associated characteristics.
- If the definitions do not establish the relationship, use a
  conservative score.
- Do not invent facts that are absent from the definitions.

Return one plausibility score from 0.0 to 1.0 for each candidate:
- 0.0 = clearly unsupported or contradictory
- 0.5 = uncertain or only weakly supported
- 1.0 = strongly supported as a necessary subclass relationship

These scores are semantic judgments, not statistical probabilities.

Return ONLY valid JSON in this exact structure:
{{
  "scores": [
    {{
      "index": 0,
      "plausibility": 0.0,
      "rationale": "Brief explanation grounded in the definitions."
    }}
  ]
}}

Evaluate every candidate exactly once. Preserve its index.

CANDIDATES:
{json.dumps(candidate_descriptions, indent=2)}
"""

    return prompt


# ============================================================
# GEMINI RESPONSE AND CACHE
# ============================================================

def load_cache():
    if not CACHE_PATH.exists():
        return {}

    with open(CACHE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CACHE_PATH, "w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2)


def extract_json_response(response_text):
    cleaned = response_text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")

    if start == -1:
        raise ValueError("Gemini response did not contain a JSON object.")

    result, _ = json.JSONDecoder().raw_decode(cleaned[start:])

    return result


def get_gemini_scores(scored_pairs, definitions):
    cache = load_cache()

    cache_key = json.dumps(
        [
            {
                "subclass_iri": pair["subclass_iri"],
                "superclass_iri": pair["superclass_iri"],
                "subclass_definition": definitions[pair["subclass_iri"]]["definition"],
                "superclass_definition": definitions[pair["superclass_iri"]]["definition"],
            }
            for pair in scored_pairs
        ],
        sort_keys=True,
    )

    if cache_key in cache:
        print("\nUsing cached Gemini assessment.")
        return cache[cache_key]

    # Only a cache miss needs the API key, so cached reruns work without one.
    if not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GEMINI_API_KEY is not configured in this terminal."
        )

    print("\nRequesting Gemini semantic assessment...")
    print(f"Candidates in request: {len(scored_pairs)}")

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    try:
        response = client.interactions.create(
            model=GEMINI_MODEL,
            input=build_gemini_prompt(scored_pairs, definitions),
        )

        response_text = response.output_text

    finally:
        client.close()

    parsed = extract_json_response(response_text)

    if "scores" not in parsed:
        raise ValueError("Gemini response is missing the scores array.")

    scores = parsed["scores"]

    expected_indices = set(range(len(scored_pairs)))
    received_indices = set()

    validated = {}

    for item in scores:
        index = item.get("index")
        plausibility = item.get("plausibility")
        rationale = item.get("rationale")

        if not isinstance(index, int) or index not in expected_indices:
            raise ValueError(f"Invalid Gemini candidate index: {index}")

        if index in received_indices:
            raise ValueError(f"Duplicate Gemini candidate index: {index}")

        if (
            not isinstance(plausibility, (int, float))
            or not 0.0 <= plausibility <= 1.0
        ):
            raise ValueError(
                f"Invalid plausibility score for candidate {index}."
            )

        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(
                f"Missing rationale for candidate {index}."
            )

        received_indices.add(index)

        validated[index] = {
            "gemini_plausibility": float(plausibility),
            "gemini_rationale": rationale.strip(),
        }

    if received_indices != expected_indices:
        missing = sorted(expected_indices - received_indices)
        raise ValueError(
            f"Gemini omitted candidate indices: {missing}"
        )

    ordered_scores = [
        validated[index]
        for index in range(len(scored_pairs))
    ]

    cache[cache_key] = ordered_scores
    save_cache(cache)

    return ordered_scores

# ============================================================
# HYBRID SCORING
# ============================================================

def calculate_hybrid_scores(scored_pairs, gemini_scores):
    results = []

    for pair, gemini in zip(scored_pairs, gemini_scores):
        mowl_part = MOWL_WEIGHT * pair["mowl_cosine"]
        gemini_part = GEMINI_WEIGHT * gemini["gemini_plausibility"]
        hybrid_score = mowl_part + gemini_part

        accepted = hybrid_score >= ACCEPTANCE_THRESHOLD
        # What MOWL alone would have decided at the same threshold.
        mowl_only_accepted = pair["mowl_cosine"] >= ACCEPTANCE_THRESHOLD

        result = {
            **pair,
            **gemini,
            "mowl_weight": MOWL_WEIGHT,
            "gemini_weight": GEMINI_WEIGHT,
            "hybrid_score": hybrid_score,
            "gemini_share_of_score": (
                gemini_part / hybrid_score if hybrid_score > 0 else None
            ),
            "mowl_only_accepted": mowl_only_accepted,
            "decision_changed_by_llm": accepted != mowl_only_accepted,
            "accepted": accepted,
        }

        results.append(result)

    return results


def calculate_llm_contribution(results):
    """
    LLM contribution rate = share of candidates whose accept/reject
    decision differs from the MOWL-only decision at the same threshold.
    Also reports the mean share of the hybrid score supplied by the LLM.
    """
    changed = [item for item in results if item["decision_changed_by_llm"]]
    shares = [
        item["gemini_share_of_score"]
        for item in results
        if item["gemini_share_of_score"] is not None
    ]

    return {
        "decisions_changed_by_llm": len(changed),
        "llm_contribution_rate": len(changed) / len(results) if results else 0.0,
        "promoted_by_llm": sum(
            1 for item in changed if item["accepted"]
        ),
        "demoted_by_llm": sum(
            1 for item in changed if not item["accepted"]
        ),
        "mean_llm_share_of_hybrid_score": (
            sum(shares) / len(shares) if shares else None
        ),
    }


# ============================================================
# WRITE ACCEPTED TURTLE AXIOMS
# ============================================================

def write_accepted_axioms(candidate_graph, results):
    accepted_graph = Graph()

    for prefix, namespace in candidate_graph.namespaces():
        accepted_graph.bind(prefix, namespace)

    ontology_iri = URIRef(
        "https://www.commoncoreontologies.org/QualityOntology/accepted"
    )

    accepted_graph.add((ontology_iri, RDF.type, OWL.Ontology))

    accepted_graph.add((
        ontology_iri,
        RDFS.comment,
        URIRef("https://www.commoncoreontologies.org/QualityOntology/candidates"),
    ))

    for result in results:
        if result["accepted"]:
            accepted_graph.add((
                URIRef(result["subclass_iri"]),
                RDFS.subClassOf,
                URIRef(result["superclass_iri"]),
            ))

    ACCEPTED_PATH.parent.mkdir(parents=True, exist_ok=True)

    accepted_graph.serialize(
        destination=str(ACCEPTED_PATH),
        format="turtle",
    )

    return sum(1 for item in results if item["accepted"])


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(results, accepted_count, contribution):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "ontology": "Quality Ontology",
        "filter": "MOWL ELEmbeddings + Gemini",
        "gemini_model": GEMINI_MODEL,
        "candidate_type": "Named subclass axioms",
        "candidate_count": len(results),
        "mowl_weight": MOWL_WEIGHT,
        "gemini_weight": GEMINI_WEIGHT,
        "hybrid_formula": "mowl_weight * raw_cosine + gemini_weight * plausibility",
        "acceptance_threshold": ACCEPTANCE_THRESHOLD,
        "acceptance_threshold_source": "selected_threshold in reports/mowl_metrics.json",
        "validation_mean_cosine": VALIDATION_MEAN_COSINE,
        "accepted_count": accepted_count,
        "rejected_count": len(results) - accepted_count,
        "llm_contribution": contribution,
        "note": (
            "Hybrid scores are ranking scores, not calibrated probabilities. "
            "Restriction and disjointness candidates in candidate_el.ttl are "
            "not scored; only named (sub, super) pairs are filtered."
        ),
        "results": results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    metrics_report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    metrics_report["hybrid_settings"] = {
        "mowl_weight": MOWL_WEIGHT,
        "llm_weight": GEMINI_WEIGHT,
        "llm_model": GEMINI_MODEL,
        "formula": "mowl_weight * raw_cosine + llm_weight * plausibility",
        "acceptance_threshold": ACCEPTANCE_THRESHOLD,
        "accepted_count": accepted_count,
        "llm_contribution_rate": contribution["llm_contribution_rate"],
    }
    METRICS_PATH.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")



# ============================================================
# MAIN
# ============================================================

def main():
    print("\nHYBRID CANDIDATE FILTER")
    print("=============================================")

    definitions = load_definitions()
    candidate_graph, pairs = load_candidates()

    print(f"Named subclass candidates: {len(pairs)}")
    print(f"Definitions loaded: {len(definitions)}")

    for pair in pairs:
        if (
            pair["subclass_iri"] not in definitions
            or pair["superclass_iri"] not in definitions
        ):
            raise ValueError(
                "A candidate class is missing an enriched definition."
            )

    model, embeddings, temporary_directory = restore_mowl_model()

    try:
        mowl_results = score_mowl_candidates(
            pairs,
            embeddings,
        )

        gemini_results = get_gemini_scores(
            mowl_results,
            definitions,
        )

        results = calculate_hybrid_scores(
            mowl_results,
            gemini_results,
        )

        contribution = calculate_llm_contribution(results)

        accepted_count = write_accepted_axioms(
            candidate_graph,
            results,
        )

        save_report(results, accepted_count, contribution)

        print("\nHYBRID FILTER RESULTS")
        print("---------------------------------------------")

        for item in results:
            subclass = definitions[item["subclass_iri"]]["label"]
            superclass = definitions[item["superclass_iri"]]["label"]

            status = "ACCEPTED" if item["accepted"] else "REJECTED"
            flag = " *LLM changed decision*" if item["decision_changed_by_llm"] else ""

            print(
                f"{status:8} "
                f"{subclass} -> {superclass} | "
                f"MOWL={item['mowl_cosine']:.3f} | "
                f"Gemini={item['gemini_plausibility']:.3f} | "
                f"Hybrid={item['hybrid_score']:.3f}{flag}"
            )

        print("\nFILTER COMPLETE")
        print("---------------------------------------------")
        print(f"Candidates evaluated: {len(results)}")
        print(f"Accepted: {accepted_count}")
        print(f"Rejected: {len(results) - accepted_count}")
        print(f"Acceptance threshold (learned tau): {ACCEPTANCE_THRESHOLD:.2f}")
        print(
            f"LLM contribution rate: {contribution['llm_contribution_rate']:.1%} "
            f"({contribution['decisions_changed_by_llm']} decisions changed: "
            f"{contribution['promoted_by_llm']} promoted, "
            f"{contribution['demoted_by_llm']} demoted)"
        )
        print(
            f"Mean LLM share of hybrid score: "
            f"{contribution['mean_llm_share_of_hybrid_score']:.1%}"
        )
        print(f"Detailed report: {REPORT_PATH}")
        print(f"Accepted axioms: {ACCEPTED_PATH}")
        print(f"Gemini cache: {CACHE_PATH}")

    finally:
        temporary_directory.cleanup()


if __name__ == "__main__":
    main()