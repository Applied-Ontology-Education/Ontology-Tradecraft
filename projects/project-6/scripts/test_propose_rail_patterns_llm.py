"""Offline Step 2 tests. No API key, remote network, or new individuals needed."""
from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path

from inventory_ontology import sha256_file
from generate_rail_axioms import generate
from refine_rail_axioms import refine
from propose_rail_patterns_llm import BASE_TYPES, DEFAULT_NAMESPACE, AGENT_NS, run, validate_proposal, validate_refinement_contract, build_request

SOURCE = '''
@prefix : <https://example.org/test/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
:ontology a owl:Ontology ; owl:imports <https://example.org/ref> .
:Airplane a owl:Class .
:person a owl:Class .
:ticket a owl:Class .
:passenger a owl:Class .
<https://www.commoncoreontologies.org/AgentOntology/building> a owl:Class .
<https://www.commoncoreontologies.org/AgentOntology/ticket> a owl:Class .
<https://www.commoncoreontologies.org/AgentOntology/reservation> a owl:Class .
<https://www.commoncoreontologies.org/AgentOntology/passenger> a owl:Class ;
  rdfs:subClassOf <http://purl.obolibrary.org/obo/BFO_0000023> .
:person_01 a owl:NamedIndividual, :person .
[] a owl:Class ; rdfs:subClassOf :passenger .
'''
IMPORT = '''
@prefix : <https://example.org/ref/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
<https://example.org/ref> a owl:Ontology .
''' + '\n'.join(f'<{iri}> a owl:Class .' for iri in (*BASE_TYPES.values(), 'http://purl.obolibrary.org/obo/BFO_0000040')) + '\n'


def fake_proposal(payload, model):
    ns = payload["namespace"]
    names = {"train": "train", "passenger train": "passenger_train",
             "railway station": "railway_station", "rail journey": "rail_journey",
             "rail journey plan": "rail_journey_plan", "rail ticket": "rail_ticket",
             "rail reservation information": "rail_reservation_information",
             "rail passenger role": "rail_passenger_role"}
    parents = {"train": BASE_TYPES["object"], "passenger train": ns + "train",
               "railway station": AGENT_NS + "building", "rail journey": BASE_TYPES["process"],
               "rail journey plan": BASE_TYPES["information_content_entity"],
               "rail ticket": AGENT_NS + "ticket",
               "rail reservation information": AGENT_NS + "reservation",
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
    object_props = []
    for prop, spec in payload["required_manifest_object_property_contract"].items():
        object_props.append({
            "local_name": prop, "label": prop.replace("_", " "),
            "definition": f"A new object property representing {prop} for rail travel.",
            "domain_iri": spec["domain_iri"], "range_iri": spec["range_iri"],
            "reused_analogue_iri": "",
            "rationale": f"Connects a rail journey to an entity using {prop}.",
        })
    data_props = []
    for prop, spec in payload["required_manifest_data_property_contract"].items():
        data_props.append({
            "local_name": prop, "label": prop.replace("_", " "),
            "definition": f"A rail metadata property representing {prop}.",
            "domain_iri": spec["domain_iri"],
            "datatype_iri": spec["datatype_iri"],
            "rationale": f"Identifies an aspect of rail travel using {prop}.",
        })
    return {"rail_classes": proposals, "object_properties": object_props,
            "data_properties": data_props,
            "design_warnings": ["A candidate proposal, not a validated ontology."]}


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

    def test_missing_data_property_fails_refinement_contract(self):
        payload, *rest = build_request(self.root, DEFAULT_NAMESPACE, "test-model")
        candidate = fake_proposal(payload, "test-model")
        candidate["data_properties"].pop()
        with self.assertRaisesRegex(ValueError, "expected 2, got 1"):
            validate_refinement_contract(candidate, payload)

    def test_llm_retry_repairs_missing_data_property(self):
        calls = []
        def flaky(payload, model):
            calls.append(payload)
            proposal = fake_proposal(payload, model)
            if len(calls) == 1:
                proposal["data_properties"].pop()
            else:
                self.assertIn("repair_feedback", payload)
            return proposal
        before = sha256_file(self.root / "src/module.ttl")
        result = run(self.root, model="test-model", namespace=DEFAULT_NAMESPACE,
                     llm_function=flaky, max_design_attempts=2)
        self.assertEqual(result["status"], "PASS_PROPOSAL_ONLY")
        self.assertEqual(result["design_attempts"], 2)
        self.assertEqual(len(calls), 2)
        doc = json.loads((self.root / "generated/rail_pattern_proposal.json").read_text())
        self.assertEqual(len(doc["proposal"]["data_properties"]), 2)
        self.assertEqual(len(doc["provenance"]["rejected_attempts"]), 1)
        self.assertEqual(before, sha256_file(self.root / "src/module.ttl"))

    def test_exhausted_attempts_fails_without_proposal(self):
        def incomplete(payload, model):
            proposal = fake_proposal(payload, model)
            proposal["data_properties"].pop()
            return proposal
        with self.assertRaisesRegex(ValueError, "rejected 2 design attempt"):
            run(self.root, model="test-model", namespace=DEFAULT_NAMESPACE,
                llm_function=incomplete, max_design_attempts=2)
        self.assertFalse((self.root / "generated/rail_pattern_proposal.json").exists())
        audit = json.loads((self.root / "reports/step2_rejected_attempts.json").read_text())
        self.assertEqual(len(audit["attempts"]), 2)

    def test_wrong_core_class_parent_fails_contract(self):
        payload, *rest = build_request(self.root, DEFAULT_NAMESPACE, "test-model")
        candidate = fake_proposal(payload, "test-model")
        train = next(x for x in candidate["rail_classes"] if x["target_concept"] == "train")
        train["parent_iri"] = BASE_TYPES["material_information_bearer"]
        with self.assertRaisesRegex(ValueError, "train: expected parent_iri"):
            validate_refinement_contract(candidate, payload)

    def test_same_count_wrong_property_names_rejected(self):
        payload, *rest = build_request(self.root, DEFAULT_NAMESPACE, "test-model")
        candidate = fake_proposal(payload, "test-model")
        candidate["object_properties"][0]["local_name"] = "has_departure_station"
        with self.assertRaisesRegex(ValueError, "object property local_name mismatch"):
            validate_refinement_contract(candidate, payload)

    def test_property_domain_range_datatype_mismatch_rejected(self):
        payload, *rest = build_request(self.root, DEFAULT_NAMESPACE, "test-model")
        for category, ix, field, bad in [
            ("object_properties", 0, "domain_iri", DEFAULT_NAMESPACE+"train"),
            ("object_properties", 1, "range_iri", DEFAULT_NAMESPACE+"train"),
            ("data_properties", 0, "datatype_iri", "http://www.w3.org/2001/XMLSchema#string"),
        ]:
            candidate = fake_proposal(payload, "test-model")
            candidate[category][ix][field] = bad
            with self.assertRaisesRegex(ValueError, "expected "+field):
                validate_refinement_contract(candidate, payload)

    def test_same_count_wrong_names_are_retried(self):
        calls = []
        def wrong_name_once(payload, model):
            candidate = fake_proposal(payload, model)
            calls.append(candidate)
            if len(calls) == 1:
                candidate["object_properties"][0]["local_name"] = "has_departure_station"
            return candidate
        result = run(self.root, model="test-model", namespace=DEFAULT_NAMESPACE,
                     llm_function=wrong_name_once, max_design_attempts=2)
        self.assertEqual(result["design_attempts"], 2)
        self.assertEqual(len(calls), 2)

    def test_contract_exactly_matches_validated_manifest_v3_predicates(self):
        payload, *rest = build_request(self.root, DEFAULT_NAMESPACE, "test-model")
        candidate = fake_proposal(payload, "test-model")
        validate_refinement_contract(candidate, payload)
        self.assertEqual({x["local_name"] for x in candidate["object_properties"]}, {
            "departs_from_station", "arrives_at_station",
            "occurs_aboard_train", "specifies_rail_journey"})
        self.assertEqual({x["local_name"] for x in candidate["data_properties"]}, {
            "has_planned_departure_time", "has_train_service_code"})
        self.assertEqual(payload["required_manifest_object_property_contract"]
                         ["specifies_rail_journey"]["domain_iri"],
                         DEFAULT_NAMESPACE + "rail_journey_plan")
        self.assertEqual(payload["required_manifest_data_property_contract"]
                         ["has_train_service_code"]["domain_iri"],
                         DEFAULT_NAMESPACE + "passenger_train")

    def test_contract_candidate_runs_through_step3_and_3a(self):
        before = sha256_file(self.root / "src/module.ttl")
        result = run(self.root, model="test-model", namespace=DEFAULT_NAMESPACE,
                     llm_function=fake_proposal)
        self.assertEqual(result["object_properties"], 4)
        self.assertEqual(result["data_properties"], 2)
        step3 = generate(self.root)
        self.assertEqual(step3["candidate_triples"], 62)
        step3a = refine(self.root)
        self.assertEqual(step3a["refined_triples"], 63)
        self.assertEqual(before, sha256_file(self.root / "src/module.ttl"))

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
