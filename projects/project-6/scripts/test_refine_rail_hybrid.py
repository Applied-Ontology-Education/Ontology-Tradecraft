"""Step 5C offline regression tests; no Java, network, API token or LLM needed."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import RDFS

import evaluate_rail_mowl as eva
import prepare_rail_mowl as pre
import refine_rail_hybrid as v2
from test_rail_mowl import make_fixture


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Step5CGroundingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source, refined, proposal, hashes = make_fixture(self.root)
        patcher = patch.object(pre, "require_inputs", return_value=(source, Graph(), refined,
                                                                   proposal, {}, {}, hashes))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.fake_train = self._fake_train
        eva.evaluate(self.root, epochs=200, with_llm=True,
                     train_function=self.fake_train, llm_function=self._old_llm)
        self.before_hashes = {p.name: sha(p) for p in (self.root / "reports").glob("step5_*.json")}
        self.source_hash = pre.file_hash(self.root / "src/module.ttl")

    def _fake_train(self, path, *, epochs, dim, seed):
        from rdflib.namespace import RDF, OWL
        g = pre.load_ontology(path)
        classes = sorted({str(x) for x in g.subjects(RDF.type, OWL.Class)})
        vec = np.random.default_rng(seed).standard_normal((len(classes), dim)).astype("float32")
        return {iri:i for i, iri in enumerate(classes)}, vec, {"test_backend": "no_java_no_network"}

    @staticmethod
    def _old_llm(req):
        return {"ratings": [{"id": item["id"],
                            "plausibility": .34 if item["parent_iri"] == pre.AGENT_NS + "passenger" else .9,
                            "rationale": "The relationship could be reasonable for the class.",
                            "concern": "Passenger may not be a role" if item["parent_iri"].endswith("/passenger") else "none"}
                            for item in req["candidates"]]}

    @staticmethod
    def _grounded_llm(req):
        return {"ratings": [{"id": item["id"],
                            "plausibility": .95 if item["parent_iri"] == pre.AGENT_NS + "passenger" else .90,
                            "rationale": "Verified ontology axioms indicate compatible parent class typing.",
                            "concern": "Human review and OWL reasoning are still required."}
                            for item in req["candidates"]]}

    def test_explicit_role_axiom_in_request(self):
        split = pre.plan(self.root)
        req = v2.grounded_request(self.root, split, "mock-grok")
        role = next(r for r in req["candidates"] if r["parent_iri"] == pre.AGENT_NS + "passenger")
        self.assertTrue(role["verified_parent_is_bfo_role_or_subclass"])
        self.assertEqual(role["verified_parent_role_ancestry_path"],
                         [pre.AGENT_NS + "passenger", v2.BFO_ROLE])
        proof = role["verified_parent_direct_superclasses"]
        self.assertTrue(any(p["object"] == v2.BFO_ROLE and "SOURCE_MODULE" in p["asserted_in"] for p in proof))
        # Unverified proposed parents are clearly distinguished from facts.
        self.assertNotIn("OTHER_PROPOSED_CANDIDATE_AXIOM", [p["verification"] for p in proof])

    def test_false_role_claim_cannot_come_from_label(self):
        source = Graph()
        source.add((URIRef("https://example.org/labelled"), RDFS.label,
                    __import__("rdflib").Literal("Passenger Role")))
        self.assertIsNone(v2.verified_ancestry(source, "https://example.org/labelled", v2.BFO_ROLE))

    def test_grounded_request_is_prepared_without_api_or_retraining(self):
        with patch.object(v2, "call_grounded_llm", side_effect=AssertionError("do not call API")), \
             patch.object(v2.old, "train_mowl", side_effect=AssertionError("do not retrain")):
            result = v2.execute(self.root, with_llm=False)
        self.assertEqual(result["status"], "PASS_GROUNDED_REQUEST_PREPARED")
        self.assertFalse(result["llm_executed"])
        self.assertEqual(result["epoch_model_reused"], 200)
        self.assertTrue((self.root / "generated/step5_v2_grounded_llm_request.json").is_file())
        self.assertEqual(self.source_hash, pre.file_hash(self.root / "src/module.ttl"))

    def test_joint_review_v2_has_separate_evidence_and_no_fake_hybrid_score(self):
        result = v2.execute(self.root, with_llm=True, llm_function=self._grounded_llm)
        self.assertEqual(result["status"], "PASS_GROUNDED_HYBRID_REVIEW_ONLY")
        self.assertEqual(result["llm_evaluated_pairs"], 9)
        self.assertEqual(result["joint_evidence_pairs"], 2)
        self.assertEqual(result["hybrid_confidence_scores_generated"], 0)
        report = json.loads((self.root / "reports/step5_v2_grounded_review.json").read_text())
        self.assertEqual(sum(x["cosine"] is not None for x in report["candidate_relationships"]), 2)
        self.assertEqual(sum(x["llm_plausibility"] is not None for x in report["candidate_relationships"]), 9)
        self.assertTrue(all(x["hybrid_confidence_score"] is None for x in report["candidate_relationships"]))
        role = next(x for x in report["candidate_relationships"] if x["parent"] == pre.AGENT_NS+"passenger")
        self.assertEqual(role["previous_llm_plausibility"], .34)
        self.assertEqual(role["llm_plausibility"], .95)
        self.assertIsNotNone(role["cosine_reference_percentile_descriptive"])
        self.assertEqual(role["cosine_evidence"], "HELD_OUT_INDEPENDENT")
        self.assertTrue(all(not x["approved"] for x in report["candidate_relationships"]))
        self.assertEqual(self.source_hash, pre.file_hash(self.root / "src/module.ttl"))

    def test_low_score_versus_verified_role_is_flagged_not_rewritten(self):
        v2.execute(self.root, with_llm=True, llm_function=self._old_llm)
        report = json.loads((self.root / "reports/step5_v2_grounded_review.json").read_text())
        role = next(x for x in report["candidate_relationships"] if x["parent"] == pre.AGENT_NS+"passenger")
        self.assertEqual(role["llm_plausibility"], .34)
        self.assertIn("LOW_LLM_SCORE_DESPITE_VERIFIED_PARENT_ROLE_TYPE_REVIEW_RATIONALE", role["grounding_flags"])

    def test_v1_reports_are_preserved(self):
        v2.execute(self.root, with_llm=True, llm_function=self._grounded_llm)
        after = {p.name: sha(p) for p in (self.root / "reports").glob("step5_*.json")}
        self.assertEqual({k: after[k] for k in self.before_hashes}, self.before_hashes)

    def test_vectors_mutation_rejected(self):
        p = self.root / "generated/mowl_rail_class_vectors.npz"
        with p.open("ab") as f:
            f.write(b"oops")
        with self.assertRaisesRegex(ValueError, "stale"):
            v2.execute(self.root, with_llm=False)

    def test_import_change_rejected(self):
        (self.root / "imports/fake.ttl").write_text("# modified", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Local imports"):
            v2.execute(self.root, with_llm=False)


if __name__ == "__main__":
    unittest.main()
