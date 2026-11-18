"""Project 6 Step 3A unit tests; all fixtures local and no xAI API calls."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_rail_axioms import generate, sha256
from refine_rail_axioms import (
    BFO_MATERIAL, BFO_OBJECT, BFO_ROLE, EXISTING_PASSENGER,
    STATION_DEF, TRAIN_DEF, refine,
)

NS = 'https://example.org/ontologies/intel-ontology#'
AGENT = 'https://www.commoncoreontologies.org/AgentOntology/'
BFO = 'http://purl.obolibrary.org/obo/'


class RefineRailAxiomsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for folder in ('src', 'imports', 'generated', 'reports'):
            (self.root / folder).mkdir()

        source = Graph()
        source.add((URIRef(NS[:-1]), RDF.type, OWL.Ontology))
        for part in ('building', 'ticket', 'reservation', 'passenger'):
            source.add((URIRef(AGENT + part), RDF.type, OWL.Class))
        source.add((EXISTING_PASSENGER, RDFS.subClassOf, BFO_ROLE))
        individual = URIRef(AGENT + 'person_01')
        source.add((individual, RDF.type, OWL.NamedIndividual))
        source.add((individual, RDFS.label, Literal('Existing person')))
        gca = BNode()
        source.add((gca, RDFS.subClassOf, URIRef(AGENT + 'passenger')))
        source.serialize(destination=str(self.root / 'src/module.ttl'), format='turtle')
        self.source_hash = sha256(self.root / 'src/module.ttl')

        imported = Graph()
        for type_ in ('BFO_0000015', 'BFO_0000023', 'BFO_0000030', 'BFO_0000040', 'IAO_0000030'):
            imported.add((URIRef(BFO + type_), RDF.type, OWL.Class))
        imported.serialize(destination=str(self.root / 'imports/mock.ttl'), format='turtle')
        parents = [
            ('train', BFO + 'BFO_0000030'),
            ('passenger_train', NS + 'train'),
            ('railway_station', AGENT + 'building'),
            ('rail_journey', BFO + 'BFO_0000015'),
            ('rail_journey_plan', BFO + 'IAO_0000030'),
            ('rail_ticket', AGENT + 'ticket'),
            ('rail_reservation_information', AGENT + 'reservation'),
            ('rail_passenger_role', BFO + 'BFO_0000023'),
        ]
        classes = [
            {
                'target_concept': local.replace('_', ' '),
                'decision': 'CREATE', 'local_name': local,
                'label': local.replace('_', ' ').title(),
                'definition': 'A test definition for ' + local,
                'parent_iri': parent, 'reused_iri': '',
                'review_notes': [], 'rationale': 'Test', 'analogue_iris': [],
            }
            for local, parent in parents
        ]
        obj = [
            {
                'local_name': local, 'label': local.replace('_', ' '),
                'definition': 'A test object relation.',
                'domain_iri': NS + 'rail_journey',
                'range_iri': NS + range_name,
                'rationale': 'Test', 'reused_analogue_iri': '',
            }
            for local, range_name in (
                ('has_departure_station', 'railway_station'),
                ('has_arrival_station', 'railway_station'),
                ('has_conveying_train', 'train'),
                ('has_intermediate_stop', 'railway_station'),
            )
        ]
        dataprops = [
            {
                'local_name': local, 'label': local.replace('_', ' '),
                'definition': 'A test data property.', 'domain_iri': NS + domain,
                'datatype_iri': 'http://www.w3.org/2001/XMLSchema#string',
                'rationale': 'Test',
            }
            for local, domain in (
                ('has_train_identifier', 'train'),
                ('has_rail_service_code', 'rail_journey_plan'),
            )
        ]
        proposal = {
            'namespace': NS, 'status': 'PROPOSED_NOT_VALIDATED_OWL',
            'provenance': {
                'source_sha256': self.source_hash,
                'local_import_hashes': {
                    'imports/mock.ttl': sha256(self.root / 'imports/mock.ttl')
                },
            },
            'proposal': {
                'rail_classes': classes, 'object_properties': obj,
                'data_properties': dataprops, 'design_warnings': [],
            },
        }
        (self.root / 'generated/rail_pattern_proposal.json').write_text(
            json.dumps(proposal), encoding='utf-8'
        )
        generate(self.root)
        self.candidate_hash = sha256(self.root / 'generated/rail_axioms_candidates.ttl')
        self.proposal_hash = sha256(self.root / 'generated/rail_pattern_proposal.json')

    def test_refines_expected_axioms_and_no_source_changes(self):
        result = refine(self.root)
        self.assertEqual(result['status'], 'PASS_REFINED_UNAPPROVED')
        self.assertEqual((result['baseline_triples'], result['refined_triples']), (62, 63))
        self.assertEqual(result['refinement_operations'], 5)
        g = Graph().parse(self.root / 'generated/rail_axioms_refined.ttl', format='turtle')
        train, station, role = (URIRef(NS + x) for x in ('train', 'railway_station', 'rail_passenger_role'))
        self.assertEqual(set(g.objects(train, RDFS.subClassOf)), {BFO_MATERIAL})
        self.assertEqual(set(g.objects(station, RDFS.subClassOf)), {BFO_MATERIAL})
        self.assertEqual(set(g.objects(role, RDFS.subClassOf)), {BFO_ROLE, EXISTING_PASSENGER})
        self.assertEqual(list(g.objects(train, SKOS.definition)), [Literal(TRAIN_DEF, lang='en')])
        self.assertEqual(list(g.objects(station, SKOS.definition)), [Literal(STATION_DEF, lang='en')])
        self.assertEqual(len(list(g.subjects(RDF.type, OWL.NamedIndividual))), 0)
        self.assertEqual(8, len(set(g.subjects(RDF.type, OWL.Class))))
        self.assertEqual(4, len(set(g.subjects(RDF.type, OWL.ObjectProperty))))
        self.assertEqual(2, len(set(g.subjects(RDF.type, OWL.DatatypeProperty))))
        self.assertEqual(self.source_hash, sha256(self.root / 'src/module.ttl'))
        self.assertEqual(self.candidate_hash, sha256(self.root / 'generated/rail_axioms_candidates.ttl'))
        self.assertEqual(self.proposal_hash, sha256(self.root / 'generated/rail_pattern_proposal.json'))

    def test_output_is_deterministic(self):
        refine(self.root)
        outputs = [
            self.root / 'generated/rail_axioms_refined.ttl',
            self.root / 'reports/step3a_summary.json',
            self.root / 'reports/step3a_refinement_audit.json',
        ]
        saved = [p.read_bytes() for p in outputs]
        refine(self.root)
        self.assertEqual(saved, [p.read_bytes() for p in outputs])

    def test_audit_contains_all_original_and_open_decisions(self):
        refine(self.root)
        audit = json.loads((self.root / 'reports/step3a_refinement_audit.json').read_text())
        self.assertEqual(5, len(audit['edits']))
        self.assertEqual(2, len(audit['open_design_decisions']))
        self.assertTrue(audit['no_merge'])
        self.assertIn('NOT OWL consistency', audit['validation_scope'])
        self.assertTrue(any(x['action'] == 'ADD_TRIPLE' for x in audit['edits']))

    def test_stale_source_fails(self):
        with (self.root / 'src/module.ttl').open('a') as fd:
            fd.write('\n# changed')
        with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
            refine(self.root)

    def test_modified_import_fails(self):
        with (self.root / 'imports/mock.ttl').open('a') as fd:
            fd.write('\n# changed')
        with self.assertRaisesRegex(ValueError, 'Import hash mismatch'):
            refine(self.root)

    def test_added_import_fails(self):
        (self.root / 'imports/extra.ttl').write_text('')
        with self.assertRaisesRegex(ValueError, 'Import file set changed'):
            refine(self.root)

    def test_changed_proposal_fails(self):
        with (self.root / 'generated/rail_pattern_proposal.json').open('a') as fd:
            fd.write('\n')
        with self.assertRaisesRegex(ValueError, 'Step 3 review references a different'):
            refine(self.root)

    def test_candidate_modified_count_fails(self):
        with (self.root / 'generated/rail_axioms_candidates.ttl').open('a') as fd:
            fd.write('\n<https://test.example/a> <https://test.example/b> <https://test.example/c> .\n')
        with self.assertRaisesRegex(ValueError, 'candidate content differs'):
            refine(self.root)

    def test_same_count_modified_candidate_fails(self):
        path = self.root / 'generated/rail_axioms_candidates.ttl'
        g = Graph().parse(path, format='turtle')
        item = URIRef(NS + 'rail_ticket')
        previous = next(g.objects(item, RDFS.label))
        g.remove((item, RDFS.label, previous))
        g.add((item, RDFS.label, Literal('Changed ticket label', lang='en')))
        g.serialize(destination=str(path), format='turtle')
        self.assertEqual(len(g), 62)
        with self.assertRaisesRegex(ValueError, 'candidate content differs'):
            refine(self.root)

    def test_unexpected_train_parent_fails(self):
        path = self.root / 'generated/rail_axioms_candidates.ttl'
        g = Graph().parse(path, format='turtle')
        train = URIRef(NS + 'train')
        g.remove((train, RDFS.subClassOf, BFO_OBJECT))
        g.add((train, RDFS.subClassOf, BFO_ROLE))
        g.serialize(destination=str(path), format='turtle')
        with self.assertRaisesRegex(ValueError, 'candidate content differs'):
            refine(self.root)

    def test_missing_existing_passenger_role_fails(self):
        path = self.root / 'src/module.ttl'
        source = Graph().parse(path, format='turtle')
        source.remove((EXISTING_PASSENGER, RDFS.subClassOf, BFO_ROLE))
        source.serialize(destination=str(path), format='turtle')
        # Change every upstream hash check so we reach the semantic assertion.
        new_hash = sha256(path)
        proposal_path = self.root / 'generated/rail_pattern_proposal.json'
        obj = json.loads(proposal_path.read_text())
        obj['provenance']['source_sha256'] = new_hash
        proposal_path.write_text(json.dumps(obj), encoding='utf-8')
        review_path = self.root / 'reports/step3_axiom_review.json'
        review = json.loads(review_path.read_text())
        review['source_sha256'] = new_hash
        review['proposal_sha256'] = sha256(proposal_path)
        review_path.write_text(json.dumps(review), encoding='utf-8')
        summary_path = self.root / 'reports/step3_summary.json'
        summary = json.loads(summary_path.read_text())
        summary['source_sha256'] = new_hash
        summary_path.write_text(json.dumps(summary), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Existing passenger class must be'):
            refine(self.root)


if __name__ == '__main__':
    unittest.main()
