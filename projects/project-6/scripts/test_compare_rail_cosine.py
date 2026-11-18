"""Offline regression tests for independent 80/20 cosine experiment."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL

import compare_rail_cosine as c


def dump(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def make_fixture(root: Path):
    ex = "https://example.org/test/cosine/"
    iri = lambda n: ex + n
    source_train = {(iri(f"C{i}"), iri(f"C{i % 4}")) for i in range(4, 20)}
    source_train |= {(iri(f"C{i}"), iri(f"C{(i + 1) % 4}")) for i in range(4, 20)}
    source_valid = {(iri(f"C{i}"), iri(f"C{(i + 2) % 4}")) for i in range(4, 12)}
    rail_train = {(iri("R0"), iri("C0")), (iri("R0"), iri("C1")),
                  (iri("R0"), iri("C2")), (iri("R1"), iri("C1")),
                  (iri("R1"), iri("C2")), (iri("R1"), iri("C3")),
                  (iri("R2"), iri("C0"))}
    rail_valid = {(iri("R0"), iri("C3")), (iri("R1"), iri("C0"))}
    train = source_train | rail_train
    valid = source_valid | rail_valid
    nodes = {n for pair in train | valid for n in pair}
    train_g, valid_g = Graph(), Graph()
    for node in nodes:
        train_g.add((URIRef(node), RDF.type, OWL.Class))
    for a, b in train:
        train_g.add((URIRef(a), RDFS.subClassOf, URIRef(b)))
    for a, b in valid:
        valid_g.add((URIRef(a), RDFS.subClassOf, URIRef(b)))
    for rel, graph in ((c.TRAIN_TTL, train_g), (c.VALID_TTL, valid_g)):
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        graph.serialize(dest, format="turtle")
    source = root / "src/module.ttl"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# original source untouched\n", encoding="utf-8")
    refined = root / "generated/rail_axioms_refined.ttl"
    refined.write_text("# candidate untouched\n", encoding="utf-8")
    split = {
        "status": "PASS_SPLIT_READY_FOR_MOWL", "seed": 42,
        "train_sha256": c.sha256(root / c.TRAIN_TTL),
        "valid_sha256": c.sha256(root / c.VALID_TTL),
        "source_sha256": c.sha256(source),
        "refined_sha256": c.sha256(refined),
        "train_axiom_pairs": len(train),
        "valid_axiom_pairs": len(valid),
        "rail_candidate_holdout_pairs": 2,
        "holdout_pairs": [{"child": a, "parent": b} for a, b in sorted(valid)],
        "candidate_pairs": ([{"child": a, "parent": b, "evaluation": "HELD_OUT"}
                             for a, b in sorted(rail_valid)] +
                            [{"child": a, "parent": b,
                              "evaluation": "IN_TRAINING_UNSCORABLE_INDEPENDENTLY"}
                             for a, b in sorted(rail_train)])
    }
    dump(root / c.SPLIT_JSON, split)
    rng = np.random.default_rng(42)
    order = sorted(nodes)
    vectors = rng.normal(size=(len(order), 16)).astype("float32")
    idx = {node: i for i, node in enumerate(order)}
    vec_path = root / c.BASELINE_NPZ
    with vec_path.open("wb") as f:
        np.savez_compressed(f, iris=np.array(order, dtype="U"), vectors=vectors)
    baseline_scores = [{"child": a, "parent": b,
                        "scope": "RAIL_NEW_DOMAIN" if (a, b) in rail_valid
                                 else "SOURCE_IMPORT_REFERENCE",
                        "status": "HELD_OUT_COSINE",
                        "cosine": round(c.center_cosine(vectors, idx, (a, b)), 6)}
                       for a, b in sorted(valid)]
    base = {"model": {"source_sha256": split["source_sha256"],
                      "refined_sha256": split["refined_sha256"]},
            "train_sha256": split["train_sha256"],
            "valid_sha256": split["valid_sha256"],
            "scores": baseline_scores,
            "heldout_rail_pairs": 2,
            "mean_heldout_rail_cosine": round(float(np.mean([
                r["cosine"] for r in baseline_scores if r["scope"] == "RAIL_NEW_DOMAIN"])), 6),
            "mean_heldout_source_import_cosine": round(float(np.mean([
                r["cosine"] for r in baseline_scores if r["scope"] == "SOURCE_IMPORT_REFERENCE"])), 6)}
    dump(root / c.BASELINE_JSON, base)
    dump(root / c.BASELINE_MANIFEST, {"train_sha256": split["train_sha256"],
                                          "valid_sha256": split["valid_sha256"],
                                          "vectors_sha256": c.sha256(vec_path)})
    return split, train, valid, rail_valid


class CosineComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.split, self.train, self.valid, self.rail = make_fixture(self.root)

    def test_split_provenance_and_leakage(self):
        d = c.read_inputs(self.root)
        self.assertEqual(d["train"], self.train)
        self.assertEqual(d["valid"], self.valid)
        self.assertEqual(d["rail"], self.rail)
        self.assertFalse(self.valid & c.closure(self.train))

    def test_same_split_baseline_and_experiment(self):
        original = c.sha256(self.root / "src/module.ttl")
        model_before = c.sha256(self.root / c.BASELINE_NPZ)
        result = c.run(self.root, epochs=15, dim=16)
        self.assertEqual(result["independently_heldout_rail_pairs"], 2)
        self.assertEqual(result["validation_pairs"], 10)
        self.assertEqual(result["training_pairs"], 39)
        self.assertEqual(result["baseline_mowl"]["heldout_rail_count"], 2)
        self.assertEqual(len(result["experiment_cosine_aware"]["rail_pair_scores"]), 2)
        self.assertTrue(-1 <= result["experiment_cosine_aware"]["heldout_rail_mean_cosine"] <= 1)
        self.assertEqual(original, c.sha256(self.root / "src/module.ttl"))
        self.assertEqual(model_before, c.sha256(self.root / c.BASELINE_NPZ))
        self.assertTrue((self.root / c.OUTPUT_NPZ).exists())
        self.assertEqual(result["source_unchanged"], True)

    def test_reproducible_fixed_seed(self):
        a = c.run(self.root, epochs=12, dim=12)
        b = c.run(self.root, epochs=12, dim=12)
        self.assertEqual(a["experiment_cosine_aware"]["validation_scores"],
                         b["experiment_cosine_aware"]["validation_scores"])
        self.assertEqual(a["experiment_cosine_aware"]["control_scores"],
                         b["experiment_cosine_aware"]["control_scores"])

    def test_training_negatives_exclude_heldout(self):
        d = c.read_inputs(self.root)
        negatives = c.sample_training_controls(d["train"], d["nodes"], seed=42,
                                               negatives_per_positive=2,
                                               known_holdouts=d["valid"])
        self.assertFalse(set(negatives) & d["valid"])
        self.assertTrue(len(negatives) > len(d["train"]))

    def test_controls_never_overlap_positive_or_transitive_axioms(self):
        d = c.read_inputs(self.root)
        controls = c.choose_controls(d["valid"], d["nodes"], d["train"] | d["valid"], seed=42)
        closure_all = c.closure(d["train"] | d["valid"])
        self.assertEqual(len(controls), 3 * len(d["valid"]))
        for row in controls:
            self.assertNotIn((row["child"], row["parent"]), closure_all)

    def test_changed_train_file_rejected(self):
        with (self.root / c.TRAIN_TTL).open("a") as f:
            f.write("# unexpected edit\n")
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            c.read_inputs(self.root)

    def test_changed_baseline_vectors_rejected(self):
        with (self.root / c.BASELINE_NPZ).open("ab") as f:
            f.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "stale, or modified"):
            c.read_inputs(self.root)

    def test_malformed_holdout_manifest_rejected(self):
        p = self.root / c.SPLIT_JSON
        split = json.loads(p.read_text(encoding="utf-8"))
        split["holdout_pairs"] = split["holdout_pairs"][:-1]
        dump(p, split)
        with self.assertRaisesRegex(ValueError, "holdout-pair manifest"):
            c.read_inputs(self.root)

    def test_unknown_heldout_endpoint_rejected(self):
        d = c.read_inputs(self.root)
        unsupported = {("https://example.org/missing/child", "https://example.org/missing/parent")}
        with self.assertRaisesRegex(ValueError, "Unindexed endpoint"):
            c.choose_controls(unsupported, d["nodes"], d["train"], seed=42)

    def test_auc_and_no_collapse_guard(self):
        self.assertEqual(c.rank_auc([.8, .9], [.1, .2]), 1)
        self.assertEqual(c.rank_auc([.1, .2], [.8, .9]), 0)
        self.assertEqual(c.rank_auc([.5], [.5]), .5)
        self.assertLess(.8 - .79, .15)  # control gap prevents trivial high scores

    def test_reject_invalid_training_hyperparameters(self):
        d = c.read_inputs(self.root)
        with self.assertRaisesRegex(ValueError, "hyperparameters"):
            c.train_contrastive(d["train"], [(a, b) for a, b in d["train"]],
                                d["nodes"], dim=1)


if __name__ == "__main__":
    unittest.main()
