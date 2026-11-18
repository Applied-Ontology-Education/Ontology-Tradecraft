"""Small deterministic tests for the read-only inventory step."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inventory_ontology import audit_project, sha256_file

BASIC_TTL = '''
@prefix : <https://example.org/test/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
:ontology a owl:Ontology ; owl:imports <https://example.org/imported> .
:Airplane a owl:Class .
:person a owl:Class .
:passenger a owl:Class .
:person_01 a owl:NamedIndividual, :person .
[] a owl:Class ; rdfs:subClassOf :passenger .
'''
RAIL_IMPORT = '''
@prefix : <https://example.org/imported/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
:ontology a owl:Ontology .
:Train a owl:Class ; rdfs:label "Train"@en .
'''


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        (self.root / "imports").mkdir()
        (self.root / "src" / "module.ttl").write_text(BASIC_TTL, encoding="utf-8")

    def test_source_untouched_and_counts(self):
        source = self.root / "src" / "module.ttl"
        before = sha256_file(source)
        summary = audit_project(self.root)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["named_classes"], 3)
        self.assertEqual(summary["named_individuals"], 1)
        self.assertEqual(summary["anonymous_subclass_gcas"], 1)
        self.assertEqual(before, sha256_file(source))

    def test_expected_report_files_are_created(self):
        audit_project(self.root)
        for filename in ("ontology_inventory.json", "rail_gap_analysis.json", "step1_summary.json"):
            self.assertTrue((self.root / "reports" / filename).exists())

    def test_reports_are_deterministic(self):
        audit_project(self.root)
        first = [(self.root / "reports" / p).read_bytes() for p in ("ontology_inventory.json", "rail_gap_analysis.json", "step1_summary.json")]
        audit_project(self.root)
        second = [(self.root / "reports" / p).read_bytes() for p in ("ontology_inventory.json", "rail_gap_analysis.json", "step1_summary.json")]
        self.assertEqual(first, second)

    def test_rail_concept_detected_from_local_import(self):
        (self.root / "imports" / "local.ttl").write_text(RAIL_IMPORT, encoding="utf-8")
        summary = audit_project(self.root)
        self.assertEqual(summary["local_import_files_read"], 1)
        results = json.loads((self.root / "reports" / "rail_gap_analysis.json").read_text(encoding="utf-8"))
        train = next(x for x in results["rail_concepts"] if x["concept"] == "train")
        self.assertEqual(train["status"], "FOUND_IN_LOCAL_IMPORTS")

    def test_missing_input_fails(self):
        (self.root / "src" / "module.ttl").unlink()
        with self.assertRaises(FileNotFoundError):
            audit_project(self.root)

    def test_malformed_import_fails(self):
        (self.root / "imports" / "bad.ttl").write_text("This is not turtle data", encoding="utf-8")
        with self.assertRaises(ValueError):
            audit_project(self.root)


if __name__ == "__main__":
    unittest.main()
