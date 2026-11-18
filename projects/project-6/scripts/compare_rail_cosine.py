"""Project 6: independent 80/20 held-out cosine experiment.

Compare existing mOWL ELEmbeddings class-center cosine with an independently
trained PyTorch cosine-contrastive baseline, on the SAME frozen Step 5A split.

This is a research diagnostic, not a replacement for ELEmbeddings or OWL
reasoning, and it never modifies source, original vector files, or Step 5B
reports. It cannot guarantee a 0.70 held-out mean.

Usage (from project-6 root, after the 80/20 Step 5A and Step 5B runs):
  python -u scripts/compare_rail_cosine.py --project-root . --epochs 300 --dim 50
  python -u scripts/compare_rail_cosine.py --project-root . --strict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL

VERSION = "project6-independent-cosine-contrastive-experiment-v1"
TRAIN_TTL = "generated/mowl_rail_train.ttl"
VALID_TTL = "generated/mowl_rail_valid.ttl"
SPLIT_JSON = "reports/step5_split_report.json"
BASELINE_JSON = "reports/step5_cosine_scores.json"
BASELINE_NPZ = "generated/mowl_rail_class_vectors.npz"
BASELINE_MANIFEST = "reports/step5_model_manifest.json"
OUTPUT_JSON = "reports/step5_cosine_comparison.json"
OUTPUT_NPZ = "generated/cosine_contrastive_class_vectors.npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def subclass_pairs(g: Graph) -> set[tuple[str, str]]:
    return {(str(s), str(o)) for s, o in g.subject_objects(RDFS.subClassOf)
            if isinstance(s, URIRef) and isinstance(o, URIRef) and s != o}


def classes(g: Graph) -> set[str]:
    return {str(s) for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}


def closure(edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    parents: dict[str, set[str]] = {}
    for child, parent in edges:
        parents.setdefault(child, set()).add(parent)
    result = set(edges)
    for child in parents:
        todo = list(parents[child]); seen: set[str] = set()
        while todo:
            ancestor = todo.pop()
            if ancestor in seen:
                continue
            seen.add(ancestor)
            if child != ancestor:
                result.add((child, ancestor))
            todo.extend(parents.get(ancestor, ()))
    return result


def read_inputs(root: Path) -> dict:
    root = root.resolve()
    paths = {name: root / rel for name, rel in {
        "train": TRAIN_TTL, "valid": VALID_TTL, "split": SPLIT_JSON,
        "baseline": BASELINE_JSON, "baseline_npz": BASELINE_NPZ,
        "baseline_manifest": BASELINE_MANIFEST, "source": "src/module.ttl",
        "refined": "generated/rail_axioms_refined.ttl",
    }.items()}
    for key, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing {key}: {p}; run Step 5A and 5B first")
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    base = json.loads(paths["baseline"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["baseline_manifest"].read_text(encoding="utf-8"))
    for name, key in (("train", "train_sha256"), ("valid", "valid_sha256"),
                      ("source", "source_sha256"), ("refined", "refined_sha256")):
        expected = split.get(key)
        if not expected or sha256(paths[name]) != expected:
            raise ValueError(f"Step 5A provenance mismatch for {name}: rerun Step 5A/5B")
    if any(base.get(k) != split.get(k) for k in ("train_sha256", "valid_sha256")):
        raise ValueError("Baseline mOWL cosine report is not from this exact 80/20 split")
    if (base.get("model", {}).get("source_sha256") != split["source_sha256"]
            or base.get("model", {}).get("refined_sha256") != split["refined_sha256"]):
        raise ValueError("Baseline mOWL scores do not match the source/refined ontology")
    if (manifest.get("train_sha256") != split["train_sha256"]
            or manifest.get("valid_sha256") != split["valid_sha256"]
            or sha256(paths["baseline_npz"]) != manifest.get("vectors_sha256")):
        raise ValueError("Baseline mOWL class vectors are missing, stale, or modified")

    train_g, valid_g = Graph(), Graph()
    train_g.parse(paths["train"], format="turtle")
    valid_g.parse(paths["valid"], format="turtle")
    train, valid = subclass_pairs(train_g), subclass_pairs(valid_g)
    declared = classes(train_g)
    if not train or not valid or train & valid:
        raise ValueError("Expected nonempty, disjoint train/validation subclass edges")
    if valid != {(r["child"], r["parent"]) for r in split.get("holdout_pairs", [])}:
        raise ValueError("Validation TTL is not equal to the saved holdout-pair manifest")
    if len(train) != split.get("train_axiom_pairs") or len(valid) != split.get("valid_axiom_pairs"):
        raise ValueError("Train/validation counts changed since Step 5A")
    if valid & closure(train):
        raise ValueError("Transitive subclass leakage detected in Step 5A split")
    nodes = {n for pair in train for n in pair}
    if any(c not in nodes or p not in nodes for c, p in valid):
        raise ValueError("At least one held-out endpoint has no training-edge support")
    if not (nodes <= declared):
        raise ValueError("Training class declarations are incomplete")
    rail = {(r["child"], r["parent"]) for r in split.get("candidate_pairs", [])
            if r.get("evaluation") == "HELD_OUT"}
    if not rail or not rail <= valid or len(rail) != split.get("rail_candidate_holdout_pairs"):
        raise ValueError("The independently held-out rail set is missing or inconsistent")
    if len(rail) < 2:
        raise ValueError("This comparison requires at least two independent rail holdouts")
    score_pairs = {(r["child"], r["parent"]) for r in base.get("scores", [])
                   if r.get("status") == "HELD_OUT_COSINE"}
    if score_pairs != valid:
        raise ValueError("mOWL baseline scores do not cover the exact held-out set")
    if base.get("heldout_rail_pairs") != len(rail):
        raise ValueError("mOWL baseline rail count does not match 80/20 holdouts")
    return {"paths": paths, "split": split, "baseline": base,
            "train": train, "valid": valid, "rail": rail, "nodes": sorted(nodes)}


def choose_controls(valid: set[tuple[str, str]], nodes: list[str],
                    all_known_edges: set[tuple[str, str]], *, seed: int,
                    per_positive: int = 3) -> list[dict]:
    """Same-child unasserted controls; NOT logically proven negative axioms.

    The true validation edges are masked from control generation; these labels
    never become training pairs or positive training loss.
    """
    if per_positive < 1:
        raise ValueError("per_positive must be >=1")
    reach = closure(all_known_edges)
    node_set = set(nodes)
    if any(c not in node_set or p not in node_set for c, p in valid):
        raise ValueError("Unindexed endpoint in held-out validation set")
    rng = random.Random(seed + 1009)
    controls: list[dict] = []
    for child, parent in sorted(valid):
        alternatives = [candidate for candidate in nodes
                        if candidate != child and candidate != parent
                        and (child, candidate) not in reach
                        and (candidate, child) not in reach]
        if len(alternatives) < per_positive:
            raise ValueError(f"Insufficient unasserted same-child controls for {child}")
        for alternative in rng.sample(alternatives, per_positive):
            controls.append({"child": child, "parent": alternative,
                             "control_for": [child, parent]})
    return controls


def sample_training_controls(train: set[tuple[str, str]], nodes: list[str],
                             *, seed: int, negatives_per_positive: int,
                             known_holdouts: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Training-only negative sampler, with known validation positives excluded.

    Holdouts are used ONLY as an exclusion mask against false negative labels;
    there is no training loss on held-out positive edges.
    Unobserved edges are weak negatives, not OWL disjointness statements.
    """
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive")
    rng = random.Random(seed + 17)
    truth = closure(train | known_holdouts)
    all_nodes = sorted(nodes)
    result: list[tuple[str, str]] = []
    for child, parent in sorted(train):
        for i in range(negatives_per_positive):
            # Sample hard-ish negatives sharing one endpoint of an asserted edge.
            # A cosine score is symmetric, so both endpoint substitutions matter.
            candidates = ([(child, other) for other in all_nodes] if i % 2 == 0
                          else [(other, parent) for other in all_nodes])
            candidates = [(a, b) for a, b in candidates
                          if a != b and (a, b) not in truth and (b, a) not in truth]
            if not candidates:
                raise ValueError(f"No unasserted negative training pair for {(child, parent)}")
            result.append(candidates[rng.randrange(len(candidates))])
    if set(result) & known_holdouts:
        raise AssertionError("Held-out positives leaked into training negatives")
    return result


def center_cosine(vectors: np.ndarray, index: dict[str, int], pair: tuple[str, str]) -> float:
    a = np.asarray(vectors[index[pair[0]]], dtype=np.float64)
    b = np.asarray(vectors[index[pair[1]]], dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError(f"Missing valid vector norm for {pair}")
    score = float(np.dot(a, b) / denominator)
    if not np.isfinite(score):
        raise ValueError(f"Nonfinite cosine for {pair}")
    return float(np.clip(score, -1, 1))


def train_contrastive(train: set[tuple[str, str]],
                      negatives: list[tuple[str, str]], nodes: list[str], *,
                      dim: int = 50, epochs: int = 300, lr: float = .015,
                      seed: int = 42, temperature: float = .2) -> tuple[dict, np.ndarray, dict]:
    """Cosine-contrastive class vectors, distinct from mOWL ELEmbeddings.

    Fixed, nonleaking edge split. Positive asserted subclass pairs and
    weakly-negative unobserved pairs are trained with logistic contrastive loss.
    Cosine is symmetric and does NOT encode subclass direction; this is a
    diagnostic comparison, not an ontology reasoner or an OWL model.
    """
    if dim < 2 or epochs < 1 or not (0 < lr <= .1) or not (0 < temperature <= 1):
        raise ValueError("Invalid training hyperparameters")
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("Install PyTorch in your existing .venv-mowl environment") from exc
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    order = sorted(nodes)
    index = {node: i for i, node in enumerate(order)}
    pos = torch.tensor([[index[a], index[b]] for a, b in sorted(train)], dtype=torch.long)
    neg = torch.tensor([[index[a], index[b]] for a, b in negatives], dtype=torch.long)
    embedding = torch.nn.Embedding(len(order), dim)
    torch.nn.init.normal_(embedding.weight, std=1.0 / dim ** .5)
    opt = torch.optim.Adam(embedding.parameters(), lr=lr)
    final_loss = None
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        v = F.normalize(embedding.weight, dim=1)
        positive = (v[pos[:, 0]] * v[pos[:, 1]]).sum(dim=1) / temperature
        negative = (v[neg[:, 0]] * v[neg[:, 1]]).sum(dim=1) / temperature
        loss = F.softplus(-positive).mean() + F.softplus(negative).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite cosine training loss")
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().item())
    vectors = embedding.weight.detach().cpu().numpy().astype("float32").copy()
    if not np.isfinite(vectors).all():
        raise RuntimeError("Nonfinite trained vectors")
    return index, vectors, {"backend": "PyTorch cosine-contrastive diagnostic",
                            "torch_version": torch.__version__,
                            "training_loss": round(final_loss, 6),
                            "positive_training_pairs": len(train),
                            "weak_negative_training_pairs": len(negatives)}


def rank_auc(positives: list[float], controls: list[float]) -> float:
    """Pairwise concordance against unasserted controls, ties count half."""
    if not positives or not controls:
        raise ValueError("Cannot calculate AUC on empty sample")
    p = np.asarray(positives, dtype=float)[:, None]
    n = np.asarray(controls, dtype=float)[None, :]
    return float(np.mean((p > n).astype(float) + 0.5 * (p == n)))


def score_group(vectors: np.ndarray, index: dict[str, int],
                valid: set[tuple[str, str]], rail: set[tuple[str, str]],
                controls: list[dict]) -> dict:
    rows = []
    for pair in sorted(valid):
        rows.append({"child": pair[0], "parent": pair[1],
                     "scope": "RAIL_NEW_DOMAIN" if pair in rail else "SOURCE_IMPORT_REFERENCE",
                     "cosine": round(center_cosine(vectors, index, pair), 6)})
    negative_rows = [{"child": c["child"], "parent": c["parent"],
                      "control_for": c["control_for"],
                      "cosine": round(center_cosine(vectors, index, (c["child"], c["parent"])), 6)}
                     for c in controls]
    rail_pos = [r["cosine"] for r in rows if r["scope"] == "RAIL_NEW_DOMAIN"]
    source_pos = [r["cosine"] for r in rows if r["scope"] == "SOURCE_IMPORT_REFERENCE"]
    rail_children = {p[0] for p in rail}
    rail_neg = [r["cosine"] for r in negative_rows if r["control_for"][0] in rail_children
                and tuple(r["control_for"]) in rail]
    source_neg = [r["cosine"] for r in negative_rows if tuple(r["control_for"]) not in rail]
    return {
        "heldout_rail_count": len(rail_pos),
        "heldout_rail_mean_cosine": round(float(np.mean(rail_pos)), 6),
        "heldout_reference_count": len(source_pos),
        "heldout_reference_mean_cosine": round(float(np.mean(source_pos)), 6),
        "rail_control_mean_cosine": round(float(np.mean(rail_neg)), 6),
        "reference_control_mean_cosine": round(float(np.mean(source_neg)), 6),
        "rail_control_auc": round(rank_auc(rail_pos, rail_neg), 6),
        "reference_control_auc": round(rank_auc(source_pos, source_neg), 6),
        "rail_positive_minus_control": round(float(np.mean(rail_pos) - np.mean(rail_neg)), 6),
        "rail_pair_scores": [r for r in rows if r["scope"] == "RAIL_NEW_DOMAIN"],
        "validation_scores": rows,
        "control_scores": negative_rows,
    }


def read_baseline_vectors(data: dict) -> tuple[dict[str, int], np.ndarray]:
    path = data["paths"]["baseline_npz"]
    with np.load(path, allow_pickle=False) as z:
        names = [str(s) for s in z["iris"]]
        vectors = np.asarray(z["vectors"], dtype=np.float32)
    if len(names) != len(set(names)) or len(names) != len(vectors):
        raise ValueError("Invalid mOWL vector index")
    if any(n not in names for n in data["nodes"]):
        raise ValueError("mOWL baseline vectors omit a class involved in the split")
    return {n: i for i, n in enumerate(names)}, vectors


def run(root: Path, *, epochs: int = 300, dim: int = 50, seed: int = 42,
        lr: float = .015, temperature: float = .2,
        negatives_per_positive: int = 2, controls_per_positive: int = 3,
        threshold: float = .70) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("Cosine threshold must be in [0,1]")
    d = read_inputs(root)
    split = d["split"]
    if seed != split.get("seed"):
        raise ValueError("Use Step 5A's saved seed; don't tune the split against validation")
    all_known = d["train"] | d["valid"]
    negative_train = sample_training_controls(d["train"], d["nodes"], seed=seed,
                                              negatives_per_positive=negatives_per_positive,
                                              known_holdouts=d["valid"])
    controls = choose_controls(d["valid"], d["nodes"], all_known,
                               seed=seed, per_positive=controls_per_positive)
    b_index, b_vectors = read_baseline_vectors(d)
    baseline = score_group(b_vectors, b_index, d["valid"], d["rail"], controls)
    for field, expected in (("heldout_rail_mean_cosine", "mean_heldout_rail_cosine"),
                            ("heldout_reference_mean_cosine", "mean_heldout_source_import_cosine")):
        if abs(baseline[field] - d["baseline"][expected]) > 1e-5:
            raise ValueError("Baseline vector cosine differs from recorded mOWL report")
    index, vectors, details = train_contrastive(
        d["train"], negative_train, d["nodes"], epochs=epochs, dim=dim,
        seed=seed, lr=lr, temperature=temperature)
    experimental = score_group(vectors, index, d["valid"], d["rail"], controls)
    cosine_met = experimental["heldout_rail_mean_cosine"] >= threshold
    controls_pass = (experimental["rail_control_auc"] >= .70
                     and experimental["rail_positive_minus_control"] >= .15)
    source_before = split["source_sha256"]
    if (sha256(d["paths"]["source"]) != source_before
            or sha256(d["paths"]["train"]) != split["train_sha256"]
            or sha256(d["paths"]["valid"]) != split["valid_sha256"]):
        raise RuntimeError("An upstream input changed during the experiment")
    output_npz = root / OUTPUT_NPZ
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_npz.with_name(output_npz.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, iris=np.array(sorted(index), dtype="U"),
                            vectors=np.array([vectors[index[n]] for n in sorted(index)], dtype=np.float32))
    tmp.replace(output_npz)
    report = {
        "version": VERSION,
        "status": "PASS_COSINE_REQUIREMENT_AND_CONTROLS" if cosine_met and controls_pass
                  else "FAIL_COSINE_REQUIREMENT" if not cosine_met else "FAIL_NEGATIVE_CONTROL_GUARDS",
        "cosine_threshold": threshold,
        "cosine_requirement_met": bool(cosine_met),
        "negative_control_guards_met": bool(controls_pass),
        "independently_heldout_rail_pairs": len(d["rail"]),
        "independently_heldout_source_import_pairs": len(d["valid"] - d["rail"]),
        "training_pairs": len(d["train"]),
        "validation_pairs": len(d["valid"]),
        "split_seed": seed,
        "training_config": {"model": "cosine-contrastive diagnostic (not ELEmbeddings)",
                            "epochs": epochs, "embedding_dim": dim,
                            "learning_rate": lr, "temperature": temperature,
                            "negatives_per_positive": negatives_per_positive,
                            "controls_per_positive": controls_per_positive},
        "train_sha256": split["train_sha256"],
        "valid_sha256": split["valid_sha256"],
        "source_sha256": source_before,
        "refined_sha256": split["refined_sha256"],
        "baseline_mowl": baseline,
        "experiment_cosine_aware": experimental,
        "experimental_training_details": details,
        "vectors": OUTPUT_NPZ,
        "vectors_sha256": sha256(output_npz),
        "source_unchanged": True,
        "limitations": [
            "Raw class-center cosine is not the native ELEmbeddings axiom score.",
            "A symmetric cosine cannot establish subclass direction or logical validity.",
            "Unasserted control pairs are not guaranteed logically false under open-world semantics.",
            "Rail candidate evaluation has only two held-out relationships; report small-sample uncertainty.",
            "Validation labels are used to mask known positives from weak-negative sampling, not as positive training targets.",
            "Do not tune hyperparameters repeatedly against the same held-out validation set.",
            "This stand-alone experiment does not change the run_all.py pipeline's pass/fail behavior."
        ],
    }
    output_json = root / OUTPUT_JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = output_json.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp_json.replace(output_json)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--dim", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=.015)
    p.add_argument("--temperature", type=float, default=.2)
    p.add_argument("--negatives-per-positive", type=int, default=2)
    p.add_argument("--controls-per-positive", type=int, default=3)
    p.add_argument("--threshold", type=float, default=.70)
    p.add_argument("--strict", action="store_true", help="Return exit 1 unless cosine AND control guards pass")
    a = p.parse_args()
    try:
        report = run(a.project_root, epochs=a.epochs, dim=a.dim, seed=a.seed,
                     lr=a.lr, temperature=a.temperature,
                     negatives_per_positive=a.negatives_per_positive,
                     controls_per_positive=a.controls_per_positive, threshold=a.threshold)
    except Exception as exc:
        print("STEP 5 COSINE COMPARISON - ERROR: " + str(exc), file=sys.stderr)
        return 2
    print("STEP 5 COSINE COMPARISON - " + report["status"])
    print(json.dumps({
        "baseline_rail_mean": report["baseline_mowl"]["heldout_rail_mean_cosine"],
        "experiment_rail_mean": report["experiment_cosine_aware"]["heldout_rail_mean_cosine"],
        "experiment_rail_auc": report["experiment_cosine_aware"]["rail_control_auc"],
        "experiment_rail_positive_minus_control": report["experiment_cosine_aware"]["rail_positive_minus_control"],
        "threshold": report["cosine_threshold"],
        "cosine_requirement_met": report["cosine_requirement_met"],
        "negative_control_guards_met": report["negative_control_guards_met"],
        "report": OUTPUT_JSON,
    }, indent=2))
    return 1 if a.strict and report["status"] != "PASS_COSINE_REQUIREMENT_AND_CONTROLS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
