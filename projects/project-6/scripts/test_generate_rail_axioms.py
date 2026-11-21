"""Offline Project 6 Step 3 tests. No xAI key or project data required."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_rail_axioms import generate, sha256

NS = 'https://example.org/ontologies/intel-ontology#'
AGENT = 'https://www.commoncoreontologies.org/AgentOntology/'
BFO = 'http://purl.obolibrary.org/obo/'


class AxiomGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for folder in ('src', 'imports', 'generated', 'reports'):
            (self.root / folder).mkdir()
        source = Graph()
        ontology = URIRef('https://example.org/ontologies/intel-ontology')
        source.add((ontology, RDF.type, OWL.Ontology))
        for term in ('building', 'ticket', 'reservation', 'passenger'):
            source.add((URIRef(AGENT + term), RDF.type, OWL.Class))
        individual = URIRef(AGENT + 'person_01')
        source.add((individual, RDF.type, OWL.NamedIndividual))
        source.add((individual, RDFS.label, Literal('Test person')))
        # Preserve an anonymous GCA from the source as a regression fixture.
        gca = BNode()
        source.add((gca, RDFS.subClassOf, URIRef(AGENT + 'passenger')))
        source.serialize(destination=str(self.root / 'src/module.ttl'), format='turtle')
        imports = Graph()
        for name in ('BFO_0000015', 'BFO_0000023', 'BFO_0000030', 'IAO_0000030'):
            imports.add((URIRef(BFO + name), RDF.type, OWL.Class))
        imports.serialize(destination=str(self.root / 'imports/test-ontology.ttl'), format='turtle')
        self.source_hash = sha256(self.root / 'src/module.ttl')
        self.proposal = self.make_proposal()
        self.write_proposal()

    def make_proposal(self):
        classes = []
        data = [
            ('train', 'BFO_0000030'),
            ('passenger_train', NS + 'train'),
            ('railway_station', AGENT + 'building'),
            ('rail_journey', 'BFO_0000015'),
            ('rail_journey_plan', 'IAO_0000030'),
            ('rail_ticket', AGENT + 'ticket'),
            ('rail_reservation_information', AGENT + 'reservation'),
            ('rail_passenger_role', 'BFO_0000023'),
        ]
        for local, parent in data:
            parent = BFO + parent if parent.startswith(('BFO_', 'IAO_')) else parent
            classes.append({
                'target_concept': local.replace('_', ' '),
                'decision': 'CREATE', 'local_name': local,
                'label': local.replace('_', ' ').title(),
                'definition': f'An example definition of {local}.',
                'parent_iri': parent, 'reused_iri': '',
                'review_notes': [], 'rationale': 'Test only', 'analogue_iris': [],
            })
        properties = [
            ('has_departure_station', 'railway_station'),
            ('has_arrival_station', 'railway_station'),
            ('has_conveying_train', 'train'),
            ('has_intermediate_stop', 'railway_station'),
        ]
        obj = [{
            'local_name': local, 'label': local.replace('_', ' '),
            'definition': f'An example relation {local}.',
            'domain_iri': NS + 'rail_journey',
            'range_iri': NS + rangename,
            'rationale': 'Test', 'reused_analogue_iri': '',
        } for local, rangename in properties]
        datatype = [{
            'local_name': local, 'label': local.replace('_', ' '),
            'definition': f'Example datatype relation {local}.',
            'domain_iri': NS + domain,
            'datatype_iri': 'http://www.w3.org/2001/XMLSchema#string',
            'rationale': 'Test',
        } for local, domain in (('has_train_identifier', 'train'),
                                ('has_rail_service_code', 'rail_journey_plan'))]
        return {
            'status': 'PROPOSED_NOT_VALIDATED_OWL',
            'namespace': NS,
            'script_version': 'test',
            'provenance': {
                'source_sha256': self.source_hash,
                'local_import_hashes': {'imports/test-ontology.ttl': sha256(self.root / 'imports/test-ontology.ttl')},
            },
            'proposal': {'rail_classes': classes, 'object_properties': obj,
                         'data_properties': datatype, 'design_warnings': []},
        }

    def write_proposal(self):
        path = self.root / 'generated/rail_pattern_proposal.json'
        path.write_text(json.dumps(self.proposal), encoding='utf-8')

    def test_expected_candidates_and_source_unchanged(self):
        result = generate(self.root)
        self.assertEqual(result['status'], 'PASS_CANDIDATES_REVIEW_REQUIRED')
        self.assertEqual(result['candidate_classes'], 8)
        self.assertEqual(result['candidate_object_properties'], 4)
        self.assertEqual(result['candidate_data_properties'], 2)
        self.assertEqual(result['candidate_triples'], 62)
        self.assertEqual(self.source_hash, sha256(self.root / 'src/module.ttl'))
        graph = Graph().parse(self.root / 'generated/rail_axioms_candidates.ttl', format='turtle')
        self.assertEqual(len(graph), 62)
        self.assertEqual(len(list(graph.subjects(RDF.type, OWL.NamedIndividual))), 0)
        self.assertEqual(set(graph.objects(URIRef(NS + 'rail_journey'), RDFS.subClassOf)),
                         {URIRef(BFO + 'BFO_0000015')})
        self.assertFalse(result['merged_into_source'])

    def test_review_station_role_and_station_properties(self):
        generate(self.root)
        review = json.loads((self.root / 'reports/step3_axiom_review.json').read_text())
        flagged = [x for x in review['rows'] if x['status'] == 'REVIEW_REQUIRED']
        self.assertEqual(len(flagged), 7)  # four classes + three station properties
        flagged_local = {x.get('target') or x.get('local_name') for x in flagged}
        self.assertIn('railway station', flagged_local)
        self.assertIn('rail passenger role', flagged_local)
        self.assertIn('has_intermediate_stop', flagged_local)

    def test_determinism(self):
        generate(self.root)
        files = [self.root / 'generated/rail_axioms_candidates.ttl',
                 self.root / 'reports/step3_summary.json',
                 self.root / 'reports/step3_axiom_review.json']
        expected = [p.read_bytes() for p in files]
        generate(self.root)
        self.assertEqual(expected, [p.read_bytes() for p in files])

    def test_stale_source_hash_fails(self):
        self.proposal['provenance']['source_sha256'] = '0' * 64
        self.write_proposal()
        with self.assertRaisesRegex(ValueError, 'source SHA-256 differs'):
            generate(self.root)

    def test_modified_import_fails(self):
        with (self.root / 'imports/test-ontology.ttl').open('a') as out:
            out.write('\n# changed import\n')
        with self.assertRaisesRegex(ValueError, 'Import changed'):
            generate(self.root)

    def test_new_import_fails(self):
        (self.root / 'imports/new.ttl').write_text('')
        with self.assertRaisesRegex(ValueError, 'Local import file set changed'):
            generate(self.root)

    def test_unknown_parent_fails(self):
        self.proposal['proposal']['rail_classes'][0]['parent_iri'] = NS + 'no_such_class'
        self.write_proposal()
        with self.assertRaisesRegex(ValueError, 'Superclass not declared'):
            generate(self.root)

    def test_wrong_role_parent_fails(self):
        self.proposal['proposal']['rail_classes'][-1]['parent_iri'] = NS + 'train'
        self.write_proposal()
        with self.assertRaisesRegex(ValueError, 'rail passenger role must have direct superclass'):
            generate(self.root)

    def test_cycle_fails(self):
        self.proposal['proposal']['rail_classes'][0]['parent_iri'] = NS + 'passenger_train'
        self.write_proposal()
        with self.assertRaisesRegex(ValueError, 'Cycle among candidate'):
            generate(self.root)

    def test_unknown_range_fails(self):
        self.proposal['proposal']['object_properties'][0]['range_iri'] = NS + 'nonexistent'
        self.write_proposal()
        with self.assertRaisesRegex(ValueError, 'Unknown range class'):
            generate(self.root)

    def test_collision_with_source_fails(self):
        self.proposal['proposal']['rail_classes'][0]['local_name'] = 'train'
        source = Graph().parse(self.root / 'src/module.ttl', format='turtle')
        source.add((URIRef(NS + 'train'), RDF.type, OWL.Class))
        source.serialize(destination=str(self.root / 'src/module.ttl'), format='turtle')
        self.proposal['provenance']['source_sha256'] = sha256(self.root / 'src/module.ttl')
        self.write_proposal()
        with self.assertRaisesRegex(ValueError, 'already used'):
            generate(self.root)


if __name__ == '__main__':
    unittest.main()
