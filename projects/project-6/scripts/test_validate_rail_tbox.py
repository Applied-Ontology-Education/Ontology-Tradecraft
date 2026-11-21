"""Offline tests for Project 6 Step 4 generation and SPARQL QC.

Actual pySHACL conformance is run by the live script with pyshacl installed.
One test uses a fake pyshacl module only to test control flow, NOT SHACL semantics.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SH, SKOS

from validate_rail_tbox import (QC_NS, RAIL_NS, create_queries, create_shapes,
                                file_hash, load_json, parse_turtle, validate)

BFO_OBJECT = URIRef('http://purl.obolibrary.org/obo/BFO_0000030')
BFO_MATERIAL = URIRef('http://purl.obolibrary.org/obo/BFO_0000040')
BFO_ROLE = URIRef('http://purl.obolibrary.org/obo/BFO_0000023')
BFO_PROCESS = URIRef('http://purl.obolibrary.org/obo/BFO_0000015')
PERSON_ROLE = URIRef('https://www.commoncoreontologies.org/AgentOntology/passenger')
BUILDING = URIRef('https://www.commoncoreontologies.org/AgentOntology/building')
LOCAL = lambda n: URIRef(RAIL_NS + n)


def serialize(graph: Graph) -> str:
    return '\n'.join(sorted(f'{s.n3()} {p.n3()} {o.n3()} .' for s,p,o in graph)) + '\n'


def fixture(root: Path) -> None:
    for folder in ('src', 'generated', 'reports', 'imports'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root/'src/module.ttl').write_text(f'{PERSON_ROLE.n3()} a {OWL.Class.n3()} .\n', encoding='utf-8')
    (root/'imports/mock.ttl').write_text(f'{BFO_MATERIAL.n3()} a {OWL.Class.n3()} .\n', encoding='utf-8')
    parents = {
        'train': BFO_OBJECT, 'passenger_train': LOCAL('train'),
        'railway_station': BUILDING, 'rail_journey': BFO_PROCESS,
        'rail_journey_plan': URIRef('http://purl.obolibrary.org/obo/IAO_0000030'),
        'rail_ticket': URIRef('https://www.commoncoreontologies.org/AgentOntology/ticket'),
        'rail_reservation_information': URIRef('https://www.commoncoreontologies.org/AgentOntology/reservation'),
        'rail_passenger_role': BFO_ROLE,
    }
    properties = {
        'has_departure_station': (OWL.ObjectProperty, 'rail_journey', 'railway_station'),
        'has_arrival_station': (OWL.ObjectProperty, 'rail_journey', 'railway_station'),
        'has_conveying_train': (OWL.ObjectProperty, 'rail_journey', 'train'),
        'has_intermediate_stop': (OWL.ObjectProperty, 'rail_journey', 'railway_station'),
        'has_train_identifier': (OWL.DatatypeProperty, 'train', None),
        'has_rail_service_code': (OWL.DatatypeProperty, 'rail_journey_plan', None),
    }
    old_definitions = {
        'train': 'A material object that is a rail vehicle designed to travel on tracks and that can convey passengers or freight.',
        'railway_station': 'A building that is a facility at which rail journeys begin, end, or pause.',
    }
    revised_definitions = {
        'train': 'A material entity composed of one or more rail vehicles operated together on a railway to convey passengers or freight.',
        'railway_station': 'A material entity comprising physical railway facilities at which rail journeys begin, end, or pause, potentially including platforms, track infrastructure, structures, and buildings.',
    }
    g = Graph()
    for n, parent in parents.items():
        t = LOCAL(n)
        g.add((t, RDF.type, OWL.Class))
        g.add((t, RDFS.subClassOf, parent))
        g.add((t, RDFS.label, Literal(n.replace('_', ' '), lang='en')))
        g.add((t, SKOS.definition, Literal(old_definitions.get(n, f'Definition of {n}.'), lang='en')))
    for n,(typ,domain,ran) in properties.items():
        t = LOCAL(n)
        g.add((t, RDF.type, typ))
        g.add((t, RDFS.domain, LOCAL(domain)))
        g.add((t, RDFS.range, LOCAL(ran) if ran else URIRef('http://www.w3.org/2001/XMLSchema#string')))
        g.add((t, RDFS.label, Literal(n.replace('_',' '), lang='en')))
        g.add((t, SKOS.definition, Literal(f'Definition of {n}.', lang='en')))
    assert len(g) == 62
    (root/'generated/rail_axioms_candidates.ttl').write_text(serialize(g), encoding='utf-8')
    edited = Graph()
    for t in g: edited.add(t)
    edits = []
    def replace(n, pred, before, after):
        t = LOCAL(n)
        b = Literal(before,lang='en') if pred == SKOS.definition else before
        a = Literal(after,lang='en') if pred == SKOS.definition else after
        edited.remove((t, pred, b)); edited.add((t,pred,a))
        edits.append({'action':'REPLACE_TRIPLE_OBJECT','subject':str(t),'predicate':str(pred),
                      'before':str(before),'after':str(after)})
    replace('train', RDFS.subClassOf, BFO_OBJECT, BFO_MATERIAL)
    replace('railway_station', RDFS.subClassOf, BUILDING, BFO_MATERIAL)
    for n in ('train','railway_station'):
        replace(n, SKOS.definition, old_definitions[n], revised_definitions[n])
    edited.add((LOCAL('rail_passenger_role'), RDFS.subClassOf, PERSON_ROLE))
    edits.append({'action':'ADD_TRIPLE','subject':str(LOCAL('rail_passenger_role')),
                  'predicate':str(RDFS.subClassOf),'before':None,'after':str(PERSON_ROLE)})
    assert len(edited) == 63
    (root/'generated/rail_axioms_refined.ttl').write_text(serialize(edited), encoding='utf-8')
    source_sha = file_hash(root/'src/module.ttl')
    proposal = {'provenance':{'source_sha256': source_sha,
                             'local_import_hashes':{'imports/mock.ttl': file_hash(root/'imports/mock.ttl')}}}
    proposal_path = root/'generated/rail_pattern_proposal.json'
    proposal_path.write_text(json.dumps(proposal), encoding='utf-8')
    (root/'reports/step3a_refinement_audit.json').write_text(json.dumps({
        'status':'REFINED_UNAPPROVED','no_merge':True,
        'source_sha256':source_sha,'proposal_sha256':file_hash(proposal_path),
        'step3_candidate_sha256':file_hash(root/'generated/rail_axioms_candidates.ttl'),
        'refined_triples':63,'baseline_triples':62,'edits':edits,
    }),encoding='utf-8')
    (root/'reports/step3a_summary.json').write_text(json.dumps({
        'status':'PASS_REFINED_UNAPPROVED','merged_into_source':False,
        'source_sha256':source_sha,'refined_triples':63,'refinement_operations':5,
    }),encoding='utf-8')


class RailTboxStep4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fixture(self.root)

    def test_generate_only_reports_incomplete_shacl(self):
        result = validate(self.root, generate_only=True)
        self.assertEqual(result['status'], 'GENERATED_ONLY_NOT_SHACL_VALIDATED')
        self.assertIsNone(result['shacl']['conforms'])
        self.assertTrue(result['sparql_passed'])
        self.assertEqual(result['generated_query_count'], 11)
        self.assertEqual(result['generated_shacl_node_shapes'], 14)

    def test_source_unmodified(self):
        before = file_hash(self.root/'src/module.ttl')
        validate(self.root, generate_only=True)
        self.assertEqual(before, file_hash(self.root/'src/module.ttl'))

    def test_repeatable_byte_identical_generated_artifacts(self):
        validate(self.root, generate_only=True)
        paths = list((self.root/'generated/sparql_qc').glob('*.rq')) + [
            self.root/'generated/rail_tbox_shapes.ttl',
            self.root/'reports/step4_summary.json',
            self.root/'reports/step4_qc_results.json',
        ]
        before = {str(p):file_hash(p) for p in paths}
        validate(self.root, generate_only=True)
        self.assertEqual(before, {str(p):file_hash(p) for p in paths})

    def test_missing_prerequisite_fails(self):
        (self.root/'generated/rail_axioms_refined.ttl').unlink()
        with self.assertRaises(FileNotFoundError): validate(self.root, generate_only=True)

    def test_tampered_refined_graph_fails_even_if_same_count(self):
        p = self.root/'generated/rail_axioms_refined.ttl'
        g = parse_turtle(p)
        g.remove((LOCAL('railway_station'),RDFS.subClassOf,BFO_MATERIAL))
        g.add((LOCAL('railway_station'),RDFS.subClassOf,BUILDING))
        self.assertEqual(len(g),63)
        p.write_text(serialize(g),encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'does not equal audited'):validate(self.root,generate_only=True)

    def test_changed_source_hash_fails(self):
        with (self.root/'src/module.ttl').open('a',encoding='utf-8') as f: f.write('\n# changed\n')
        with self.assertRaisesRegex(ValueError,'Source hash'):validate(self.root,generate_only=True)

    def test_changed_import_hash_fails(self):
        with (self.root/'imports/mock.ttl').open('a',encoding='utf-8') as f: f.write('# changed\n')
        with self.assertRaisesRegex(ValueError, 'Local imports changed'):validate(self.root,generate_only=True)

    def test_sparql_failure_is_detectable(self):
        g = parse_turtle(self.root/'generated/rail_axioms_refined.ttl')
        queries = create_queries(g)
        g.remove((LOCAL('train'),RDFS.label,Literal('train',lang='en')))
        missing = list(g.query(queries['04_missing_english_label']))
        self.assertTrue(missing)

    def test_generated_shapes_target_all_vocab(self):
        g = parse_turtle(self.root/'generated/rail_axioms_refined.ttl')
        shapes = create_shapes(g)
        self.assertEqual(len(set(shapes.subjects(RDF.type,SH.NodeShape))),14)
        targets = set(shapes.objects(None, SH.targetNode))
        self.assertIn(LOCAL('train'), targets)
        self.assertIn(LOCAL('rail_passenger_role'), targets)
        self.assertIn(LOCAL('has_departure_station'),targets)
        role_shape = URIRef(QC_NS + 'rail_passenger_role_Shape')
        self.assertTrue(any((ps, SH.hasValue, PERSON_ROLE) in shapes
                            for ps in shapes.objects(role_shape, SH.property)))

    def test_missing_pyshacl_is_not_reported_as_success(self):
        with patch.dict(sys.modules, {'pyshacl': None}):
            with self.assertRaisesRegex(RuntimeError, 'pyshacl is not installed'):
                validate(self.root, generate_only=False)

    def test_shacl_integration_branch_mock_only(self):
        called = []
        def fake_validate(**kw):
            called.append(kw)
            return True, Graph(), 'Validation Report\nConforms: True'
        with patch.dict(sys.modules, {'pyshacl': types.SimpleNamespace(validate=fake_validate)}):
            result = validate(self.root, generate_only=False)
        self.assertEqual(result['status'], 'PASS_QC_AND_SHACL_UNAPPROVED')
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0]['inference'], 'none')

    def test_sparql_rules_parse_and_execute(self):
        validate(self.root, generate_only=True)
        g = parse_turtle(self.root/'generated/rail_axioms_refined.ttl')
        for p in (self.root/'generated/sparql_qc').glob('*.rq'):
            rows = list(g.query(p.read_text(encoding='utf-8')))
            self.assertEqual(rows, [], p.name)


if __name__ == '__main__':
    unittest.main()
