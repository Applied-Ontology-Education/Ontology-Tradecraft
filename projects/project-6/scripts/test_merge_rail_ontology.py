"""Step 7 offline regression suite. No Java, network, xAI, or user files needed.

Run in project-6 root:
    python -m unittest discover -s scripts -p test_merge_rail_ontology.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import OWL, RDF, RDFS, SKOS

import merge_rail_ontology as s7


class Step7MergeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name in ("src", "generated", "reports"):
            (self.root / name).mkdir()
        source = Graph()
        ontology = URIRef("https://example.org/ontology")
        source.add((ontology, RDF.type, OWL.Ontology))
        source.add((ontology, OWL.imports, URIRef("https://example.org/import")))
        source.add((s7.EXISTING_TICKET, RDFS.subClassOf, s7.IAO_MATERIAL_BEARER))
        source.add((s7.EXISTING_PASSENGER, RDFS.subClassOf, s7.BFO_ROLE))
        bnode = BNode("restriction1")
        a = URIRef("https://example.org/class/Preexisting")
        source.add((a, RDFS.subClassOf, bnode))
        source.add((bnode, RDF.type, OWL.Restriction))
        source.add((bnode, OWL.onProperty, URIRef("https://example.org/hasProperty")))
        for i in range(421 - len(source)):
            source.add((URIRef(f"https://example.org/old/{i}"), RDFS.label, Literal(f"old {i}")))
        assert len(source) == 421
        source.serialize(self.root / "src/module.ttl", format="turtle")

        cls_names = ("train", "passenger_train", "railway_station", "rail_journey",
                     "rail_journey_plan", "rail_ticket", "rail_reservation_information",
                     "rail_passenger_role")
        child_parent = {
            "train": str(s7.BFO_MATERIAL),
            "passenger_train": s7.NS + "train",
            "railway_station": str(s7.BFO_MATERIAL),
            "rail_journey": "http://purl.obolibrary.org/obo/BFO_0000015",
            "rail_journey_plan": "http://purl.obolibrary.org/obo/IAO_0000030",
            "rail_ticket": str(s7.EXISTING_TICKET),
            "rail_reservation_information": s7.AGENT + "reservation",
            "rail_passenger_role": str(s7.BFO_ROLE),
        }
        candidate = Graph()
        for name in cls_names:
            ent = URIRef(s7.NS + name)
            candidate.add((ent, RDF.type, OWL.Class))
            candidate.add((ent, RDFS.subClassOf, URIRef(child_parent[name])))
            candidate.add((ent, RDFS.label, Literal(name.replace("_", " "), lang="en")))
            definition = ({
                "rail_ticket": "A ticket that is a material information bearer authorizing a rail journey.",
                "railway_station": "A material entity of physical railway facilities including platforms.",
            }).get(name, f"An entity representing {name} in the rail domain.")
            candidate.add((ent, SKOS.definition, Literal(definition, lang="en")))
        candidate.add((s7.ROLE, RDFS.subClassOf, s7.EXISTING_PASSENGER))
        props = ("has_departure_station", "has_arrival_station", "has_conveying_train",
                 "has_intermediate_stop", "has_train_identifier", "has_rail_service_code")
        for i, name in enumerate(props):
            ent = URIRef(s7.NS + name)
            candidate.add((ent, RDF.type, OWL.ObjectProperty if i < 4 else OWL.DatatypeProperty))
            candidate.add((ent, RDFS.label, Literal(name.replace("_", " "), lang="en")))
            candidate.add((ent, SKOS.definition, Literal(f"Rail relation {name}.", lang="en")))
            candidate.add((ent, RDFS.domain, URIRef(s7.NS + ("rail_journey" if i < 4 else "rail_journey_plan"))))
            candidate.add((ent, RDFS.range, URIRef(s7.NS + "railway_station") if i < 4 else URIRef("http://www.w3.org/2001/XMLSchema#string")))
        assert len(candidate) == 63
        candidate.serialize(self.root / "generated/rail_axioms_refined.ttl", format="turtle")

        b = Graph()
        b += source
        for i in range(20):
            b.add((URIRef(f"https://example.org/imported/{i}"), RDFS.label, Literal(f"imported {i}")))
        c = Graph()
        c += b
        c += candidate
        b.serialize(self.root / "generated/step6_baseline_flattened.ttl", format="turtle")
        c.serialize(self.root / "generated/step6_candidate_flattened.ttl", format="turtle")
        source_hash = s7.file_hash(self.root / "src/module.ttl")
        refined_hash = s7.file_hash(self.root / "generated/rail_axioms_refined.ttl")
        self.pairs = [dict(child=str(x), parent=str(y), approved=False)
                      for x,y in candidate.subject_objects(RDFS.subClassOf)]
        self.write("step4_summary.json", {
            "status": "PASS_QC_AND_SHACL_UNAPPROVED", "sparql_passed": True,
            "shacl": {"conforms": True}, "source_sha256": source_hash, "refined_sha256": refined_hash})
        self.write("step5_v2_summary.json", {
            "status": "PASS_GROUNDED_HYBRID_REVIEW_ONLY", "llm_executed": True,
            "semantic_approval": False, "source_sha256": source_hash})
        self.write("step5_v2_grounded_review.json", {
            "no_automated_approval": True, "candidate_relationships": self.pairs,
            "source_sha256": source_hash, "refined_sha256": refined_hash})
        status = "PASS_REASONER_COHERENT_UNAPPROVED"
        self.write("step6_reasoner_summary.json", {
            "status": status, "baseline_reasoner_passed": True,
            "augmented_reasoner_passed": True, "owl_reasoner_executed": True,
            "source_unchanged": True, "source_merged": False,
            "reasoning_scope": "SOURCE_PLUS_STAR_LOCALITY_MODULE_OF_IMPORTS",
            "source_sha256": source_hash, "refined_sha256": refined_hash})
        self.write("step6_reasoner_detail.json", {
            "status": status, "reasoner": "hermit", "merged_into_source": False,
            "baseline_reasoner_run": {"return_code": 0, "timed_out": False},
            "augmented_reasoner_run": {"return_code": 0, "timed_out": False},
            "baseline_flattened_sha256": s7.file_hash(self.root / "generated/step6_baseline_flattened.ttl"),
            "augmented_flattened_sha256": s7.file_hash(self.root / "generated/step6_candidate_flattened.ttl"),
            "new_candidate_triples": 63,
            "baseline": {"source_triples": 421},
            "candidate": {"source_triples": 421, "flattened_triples": len(c)},
            "local_input_hashes": {"source": source_hash, "refined": refined_hash},
        })
        self.mock_shacl = patch.dict(sys.modules, {"pyshacl": types.SimpleNamespace(
            validate=lambda **kwargs: (True, None, "Validation Report\nConforms: True"))})
        self.mock_shacl.start()
        self.addCleanup(self.mock_shacl.stop)

    def write(self, name, data):
        (self.root / "reports" / name).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def rewrite(self, name, fn):
        path = self.root / "reports" / name
        obj = json.loads(path.read_text())
        fn(obj)
        self.write(name, obj)

    def build_fast(self, *, check_only=False):
        fake = {"checks": [{"check": f"QC{i:02}", "passed": True}
                            for i in range(11)],
                "sparql_queries_executed": 9,
                "equivalent_annotation_checks_executed": 2,
                "sparql_passed": True, "shacl_conforms": True}
        with patch.object(s7, "_run_merged_qc", return_value=fake):
            return s7.build(self.root, check_only=check_only)

    def test_merge_passes_and_preserves_original_bytes(self):
        original = (self.root / "src/module.ttl").read_bytes()
        result = s7.build(self.root)
        self.assertEqual(result["status"], "PASS_CONTROLLED_MERGE_VALIDATED_NOT_PROMOTED")
        self.assertEqual(result["merged_triples"], 484)
        self.assertEqual(result["new_named_individuals"], 0)
        self.assertEqual(result["step7_hermit_rerun"], False)
        self.assertEqual((self.root / "src/module.ttl").read_bytes(), original)
        self.assertTrue((self.root / s7.OUTPUT).read_bytes().startswith(original))

    def test_reproducible_bytes(self):
        a = self.build_fast()["merged_sha256"]
        b = self.build_fast()["merged_sha256"]
        self.assertEqual(a,b)

    def test_no_outputs_with_check_only(self):
        result = self.build_fast(check_only=True)
        self.assertEqual(result["status"], "PASS_CONTROLLED_MERGE_CHECK_ONLY")
        self.assertFalse((self.root / s7.OUTPUT).exists())

    def test_original_anonymous_gca_survives(self):
        self.build_fast()
        original = s7.parse_turtle(self.root / "src/module.ttl")
        merged = s7.parse_turtle(self.root / s7.OUTPUT)
        anon = list(original.objects(URIRef("https://example.org/class/Preexisting"), RDFS.subClassOf))
        self.assertEqual(len(anon), 1)
        self.assertTrue(any(isinstance(o, BNode) for o in merged.objects(URIRef("https://example.org/class/Preexisting"),RDFS.subClassOf)))

    def test_import_declaration_survives(self):
        self.build_fast()
        merged = s7.parse_turtle(self.root / s7.OUTPUT)
        self.assertIn((URIRef("https://example.org/ontology"), OWL.imports,
                       URIRef("https://example.org/import")), merged)

    def test_tampered_source_rejected(self):
        with (self.root / "src/module.ttl").open("a", encoding="utf-8") as f:
            f.write("\n# changed source text\n")
        with self.assertRaisesRegex(ValueError, "source hash"):
            s7.build(self.root)
        self.assertFalse((self.root / s7.OUTPUT).exists())

    def test_missing_hermit_approval_rejected(self):
        self.rewrite("step6_reasoner_summary.json", lambda d: d.update(augmented_reasoner_passed=False))
        with self.assertRaisesRegex(ValueError, "Step 6"):
            s7.build(self.root)

    def test_ungrounded_llm_report_rejected(self):
        self.rewrite("step5_v2_summary.json", lambda d: d.update(llm_executed=False))
        with self.assertRaisesRegex(ValueError, "Step 5C"):
            s7.build(self.root)

    def test_station_scope_change_rejected(self):
        c = s7.parse_turtle(self.root / "generated/rail_axioms_refined.ttl")
        c.remove((s7.STATION, RDFS.subClassOf, s7.BFO_MATERIAL))
        c.add((s7.STATION, RDFS.subClassOf, URIRef("http://purl.obolibrary.org/obo/BFO_0000029")))
        c.serialize(self.root / "generated/rail_axioms_refined.ttl", format="turtle")
        with self.assertRaisesRegex(ValueError, "refined hash"):
            s7.build(self.root)

    def test_ticket_scope_change_rejected(self):
        c = s7.parse_turtle(self.root / "generated/rail_axioms_refined.ttl")
        c.remove((s7.TICKET, RDFS.subClassOf, s7.EXISTING_TICKET))
        c.add((s7.TICKET, RDFS.subClassOf, URIRef("http://purl.obolibrary.org/obo/IAO_0000030")))
        c.serialize(self.root / "generated/rail_axioms_refined.ttl", format="turtle")
        with self.assertRaisesRegex(ValueError, "refined hash"):
            s7.build(self.root)

    def test_hermit_input_mismatch_rejected(self):
        p = self.root / "generated/step6_candidate_flattened.ttl"
        with p.open("a", encoding="utf-8") as f:
            f.write("\n<https://example.org/Extra> <https://example.org/p> <https://example.org/o> .\n")
        with self.assertRaisesRegex(ValueError, "reasoner input provenance"):
            s7.build(self.root)

    def test_no_named_individual_added_and_summary_written(self):
        self.build_fast()
        summary = json.loads((self.root / "reports/step7_summary.json").read_text())
        self.assertEqual(summary["new_named_individuals"], 0)
        self.assertFalse(summary["source_overwritten"])
        self.assertFalse(summary["final_promotion_to_source"])
        audit = json.loads((self.root / "reports/step7_merge_audit.json").read_text())
        self.assertEqual(len(audit["step7_checks"]["checks"]), 11)

    def test_shacl_failure_does_not_write_output(self):
        with patch.object(s7, "_run_merged_qc", side_effect=ValueError("SHACL failed")):
            with self.assertRaisesRegex(ValueError, "SHACL"):
                s7.build(self.root)
        self.assertFalse((self.root / s7.OUTPUT).exists())


if __name__ == "__main__":
    unittest.main()
