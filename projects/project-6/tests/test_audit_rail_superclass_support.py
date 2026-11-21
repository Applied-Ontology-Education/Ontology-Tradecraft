"""Offline synthetic tests: no Java or actual user ontology files required."""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
from rdflib import URIRef, Graph
from rdflib.namespace import RDFS, RDF, OWL, SKOS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import experiment_rail_mowl_v8 as v8
import audit_rail_superclass_support as a


class AnchorAuditTests(unittest.TestCase):
    def fixture(self, root):
        for name in ('src', 'generated', 'reports', 'imports', 'scripts'):
            (root/name).mkdir(parents=True, exist_ok=True)
        (root/'src/module.ttl').write_text(
            '<urn:source:A> <urn:source:prop> <urn:source:B> .\n', encoding='utf-8')
        base = {(f'urn:source:Class{i}', f'urn:source:Parent{i}') for i in range(10)}
        rail = {pair for fold in v8.FOLDS for pair in fold}
        rail.add((v8.BASE+'rail_passenger_role',
                  'http://purl.obolibrary.org/obo/BFO_0000023'))
        augmented = Graph()
        for child, parent in base | rail:
            augmented.add((URIRef(child), RDFS.subClassOf, URIRef(parent)))
        augmented.serialize(root/'generated/module_augmented.ttl', format='turtle')
        train, valid = Graph(), Graph()
        for child, parent in base:
            train.add((URIRef(child), RDFS.subClassOf, URIRef(parent)))
        for child, parent in rail:
            train.add((URIRef(child), RDF.type, OWL.Class))
            train.add((URIRef(parent), RDF.type, OWL.Class))
        valid.add((URIRef(v8.BASE+'rail_passenger_role'),
                   RDFS.subClassOf, URIRef(v8.AGENT+'passenger')))
        train.add((URIRef(v8.BASE+'rail_passenger_role'),
                   RDFS.subClassOf,
                   URIRef('http://purl.obolibrary.org/obo/BFO_0000023')))
        train.serialize(root/'generated/mowl_rail_train.ttl', format='turtle')
        valid.serialize(root/'generated/mowl_rail_valid.ttl', format='turtle')
        summary = {'status': 'PASS_SPLIT_READY_FOR_MOWL',
                   'source_sha256': v8.digest(root/'src/module.ttl'),
                   'train_sha256': v8.digest(root/'generated/mowl_rail_train.ttl'),
                   'valid_sha256': v8.digest(root/'generated/mowl_rail_valid.ttl')}
        (root/'reports/step5_split_report.json').write_text(json.dumps(summary))
        g = Graph().parse(root/'generated/module_augmented.ttl', format='turtle')
        g.add((URIRef(v8.BASE+'rail_ticket'),SKOS.definition,
               __import__('rdflib').Literal('Physical railway ticket')))
        g.add((URIRef('urn:prop:arrives_at'),RDFS.range, URIRef(v8.BASE+'railway_station')))
        g.serialize(root/'generated/module_augmented.ttl', format='turtle')
        # Add harmless class to enable independently reviewed novel parent.
        train = Graph().parse(root/'generated/mowl_rail_train.ttl', format='turtle')
        train.add((URIRef('urn:parent:Main'),RDF.type,OWL.Class))
        # Use HTTP IRI to satisfy strict anchor review checks.
        train.add((URIRef('https://example.org/external/Independent'),RDF.type,OWL.Class))
        train.serialize(root/'generated/mowl_rail_train.ttl', format='turtle')
        report = json.loads((root/'reports/step5_split_report.json').read_text())
        report['train_sha256'] = v8.digest(root/'generated/mowl_rail_train.ttl')
        (root/'reports/step5_split_report.json').write_text(json.dumps(report))
        return v8.load_split(root)

    def test_audit_counts_and_nontraining_mentions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = self.fixture(root)
            snap = v8.immutability_snapshot(root)
            report = a.evidence_audit(root, inputs, snap)
            self.assertEqual(sum(report['support_counts'].values()),8)
            self.assertEqual(report['support_counts']['RETAINED_PARENT_SUPPORT'],1)
            self.assertEqual(report['support_counts']['COLD_START_NO_TAXONOMY_RELATION'],6)
            station = report['folds'][1]['heldout'][0]
            self.assertEqual(station['child'],v8.BASE+'railway_station')
            self.assertIn('rdfs:range', [ref['kind'] for ref in station['other_owl_or_annotation_evidence_NOT_subclass_axioms']['generated/module_augmented.ttl']['structural_references']])
            self.assertEqual(snap, v8.immutability_snapshot(root))

    def test_reviewed_anchor_increases_support_safely(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = self.fixture(root)
            snap = v8.immutability_snapshot(root)
            report = a.evidence_audit(root, inputs, snap)
            data = a.review_template(report)
            data['approved_anchors'] = [{
                'child': v8.BASE+'rail_ticket',
                'parent': 'https://example.org/external/Independent',
                'rationale':'The independent subclass is supported by source material',
                'independent_source':'Independent expert-reviewed domain statement',
                'reviewer':'University supervisor reviewed these domain assertions',
            }]
            p = root / 'review.json'
            p.write_text(json.dumps(data))
            approved = a.validated_anchors(p, inputs)
            out = a.write_reviewed_folds(root, inputs, report, approved, p, snap)
            self.assertEqual(out['approved_anchor_count'],1)
            self.assertTrue(out['named_taxonomy_holdouts_not_reachable'])
            self.assertFalse(out['full_owl_entailment_checked'])
            fold3 = v8.graph_load(root/out['folds'][2]['training_path'])
            self.assertIn(approved[0],v8.named_pairs(fold3))
            self.assertNotIn(v8.FOLDS[2][1],v8.named_pairs(fold3))
            self.assertEqual(snap, v8.immutability_snapshot(root))

    def test_reject_indirect_reviewed_anchor_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            # Independent is already below Ticket, so approving RailTicket ->
            # Independent would entail withheld RailTicket -> Ticket.
            train_path = root/'generated/mowl_rail_train.ttl'
            training = Graph().parse(train_path, format='turtle')
            training.add((URIRef('https://example.org/external/Independent'),
                          RDFS.subClassOf, URIRef(v8.AGENT+'ticket')))
            training.serialize(train_path, format='turtle')
            split_path = root/'reports/step5_split_report.json'
            split = json.loads(split_path.read_text())
            split['train_sha256'] = v8.digest(train_path)
            split_path.write_text(json.dumps(split))
            inputs = v8.load_split(root)
            report = a.evidence_audit(root, inputs, v8.immutability_snapshot(root))
            review = a.review_template(report)
            review['approved_anchors'] = [{
                'child': v8.BASE+'rail_ticket',
                'parent': 'https://example.org/external/Independent',
                'rationale': 'Expert review without checking logical implications',
                'independent_source': 'Hypothetical additional superclass from ontology',
                'reviewer': 'Course ontology reviewer and supervisor note',
            }]
            path = root/'review.json'
            path.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError, 'create named superclass leakage'):
                a.validated_anchors(path, inputs)

    def test_reject_explicit_or_entailed_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            inputs=self.fixture(root)
            item = dict(rationale='independently reviewed superclass proposal',
                        independent_source='A verified separate TBox design note',
                        reviewer='Course supervisor review of this axiom')
            report=a.evidence_audit(root,inputs,v8.immutability_snapshot(root))
            data=a.review_template(report)
            # Exact withheld target prohibited.
            data['approved_anchors']=[{'child':v8.BASE+'rail_ticket','parent':v8.AGENT+'ticket',**item}]
            p=root/'review.json';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'held-out target'):
                a.validated_anchors(p,inputs)
            # Projected rail ticket -> rail passenger role -> passenger target
            # would leak the fold-1 target if the new axiom is rail_passenger_role -> rail_ticket
            # and rail_ticket -> passenger already independently asserted; here use a
            # known BFO superclass relation when present in training.
            data['approved_anchors']=[{'child':v8.BASE+'rail_passenger_role',
                                       'parent':v8.BASE+'rail_passenger_role',**item}]
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'distinct'):
                a.validated_anchors(p,inputs)

if __name__=='__main__':
    unittest.main(verbosity=2)
