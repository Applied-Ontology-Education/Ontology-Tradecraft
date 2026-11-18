"""Offline Step 2 tests. No API key, remote network, or new individuals needed."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inventory_ontology import sha256_file
from propose_rail_patterns_llm import BASE_TYPES, DEFAULT_NAMESPACE, run, validate_proposal, build_request

SOURCE = '''
@prefix : <https://example.org/test/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
:ontology a owl:Ontology ; owl:imports <https://example.org/ref> .
:Airplane a owl:Class .
:person a owl:Class .
:ticket a owl:Class .
:passenger a owl:Class .
:person_01 a owl:NamedIndividual, :person .
[] a owl:Class ; rdfs:subClassOf :passenger .
'''
IMPORT = '''
@prefix : <https://example.org/ref/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
<https://example.org/ref> a owl:Ontology .
''' + '\n'.join(f'<{iri}> a owl:Class .' for iri in BASE_TYPES.values()) + '\n'


def fake_proposal(payload, model):
    ns = payload["namespace"]
    names = {"train": "train", "passenger train": "passenger_train",
             "railway station": "railway_station", "rail journey": "rail_journey",
             "rail journey plan": "rail_journey_plan", "rail ticket": "rail_ticket",
             "rail reservation information": "rail_reservation_information",
             "rail passenger role": "rail_passenger_role"}
    parents = {"train": BASE_TYPES["object"], "passenger train": ns + "train",
               "railway station": BASE_TYPES["object"], "rail journey": BASE_TYPES["process"],
               "rail journey plan": BASE_TYPES["information_content_entity"],
               "rail ticket": BASE_TYPES["material_information_bearer"],
               "rail reservation information": BASE_TYPES["information_content_entity"],
               "rail passenger role": BASE_TYPES["role"]}
    proposals = []
    for target in payload["pattern_targets"]:
        concept = target["concept"]
        matches = target["source_matches"] + target["import_matches"]
        reuse = bool(matches)
        proposals.append({
            "target_concept": concept, "decision": "REUSE" if reuse else "CREATE",
            "local_name": "" if reuse else names[concept], "label": concept,
            "definition": "" if reuse else f"A rail-domain class candidate for {concept} in an information-centric travel ontology.",
            "parent_iri": "" if reuse else parents[concept],
            "reused_iri": matches[0]["iri"] if reuse else "",
            "analogue_iris": [m["iri"] for m in target["air_pattern_analogs"]],
            "rationale": f"Maps rail concept {concept} to an appropriate source structure.",
            "review_notes": ["Human review required before OWL conversion."],
        })
    return {"rail_classes": proposals,
            "object_properties": [{"local_name": "uses_train", "label": "uses train",
                                   "definition": "A rail journey uses a particular train vehicle.",
                                   "domain_iri": ns + "rail_journey", "range_iri": ns + "train",
                                   "reused_analogue_iri": "",
                                   "rationale": "Connects process to physical vehicle in rail design."}],
            "data_properties": [], "design_warnings": ["A candidate proposal, not a validated ontology."]}


class RailProposalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "src").mkdir()
        (self.root / "imports").mkdir()
        (self.root / "src/module.ttl").write_text(SOURCE, encoding="utf-8")
        (self.root / "imports/ref.ttl").write_text(IMPORT, encoding="utf-8")

    def test_dry_run_preserves_source_and_avoids_api(self):
        before = sha256_file(self.root / "src/module.ttl")
        output = run(self.root, model="test-model", namespace=DEFAULT_NAMESPACE, dry_run=True)
        self.assertEqual(output["status"], "DRY_RUN_PASS")
        self.assertEqual(before, sha256_file(self.root / "src/module.ttl"))
        self.assertTrue((self.root / "generated/rail_design_request.json").exists())
        self.assertFalse((self.root / "generated/rail_pattern_proposal.json").exists())

    def test_offline_mock_proposal_preserves_source(self):
        before = sha256_file(self.root / "src/module.ttl")
        output = run(self.root, model="mock-model", namespace=DEFAULT_NAMESPACE, llm_function=fake_proposal)
        self.assertEqual(output["status"], "PASS_PROPOSAL_ONLY")
        self.assertEqual(output["targets"], 8)
        self.assertEqual(before, sha256_file(self.root / "src/module.ttl"))
        proposal = json.loads((self.root / "generated/rail_pattern_proposal.json").read_text())
        self.assertEqual(len(proposal["proposal"]["rail_classes"]), 8)
        self.assertFalse(list(self.root.glob("generated/*.ttl")))

    def test_missing_declared_import_fails(self):
        (self.root / "imports/ref.ttl").unlink()
        with self.assertRaisesRegex(ValueError, "Declared imports"):
            run(self.root, model="test-model", namespace=DEFAULT_NAMESPACE, dry_run=True)

    def test_unknown_parent_fails(self):
        payload, classes, properties, individuals = build_request(self.root, DEFAULT_NAMESPACE, "test")
        candidate = fake_proposal(payload, "test")
        candidate["rail_classes"][0]["parent_iri"] = "https://fake.example/Nonexistent"
        with self.assertRaisesRegex(ValueError, "Ungrounded/self parent"):
            validate_proposal(candidate, payload, classes, properties, individuals)

    def test_role_as_person_fails(self):
        payload, classes, properties, individuals = build_request(self.root, DEFAULT_NAMESPACE, "test")
        candidate = fake_proposal(payload, "test")
        role = next(x for x in candidate["rail_classes"] if x["target_concept"] == "rail passenger role")
        role["parent_iri"] = "https://example.org/test/person"
        with self.assertRaisesRegex(ValueError, "BFO Role"):
            validate_proposal(candidate, payload, classes, properties, individuals)

    def test_duplicate_class_iri_fails(self):
        payload, classes, properties, individuals = build_request(self.root, DEFAULT_NAMESPACE, "test")
        candidate = fake_proposal(payload, "test")
        candidate["rail_classes"][1]["local_name"] = "train"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_proposal(candidate, payload, classes, properties, individuals)


if __name__ == "__main__":
    unittest.main()
