"""Offline regression checks: real RDFlib graphs, mocked HermiT only."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from rdflib import URIRef, Graph
from rdflib.namespace import OWL, RDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_v9_owl_entailment as a
import audit_v9_owl_entailment_v2 as v2

fixture_spec=importlib.util.spec_from_file_location('original_test',ROOT/'tests/test_audit_v9_owl_entailment.py')
original_test=importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(original_test)


class TestV2(unittest.TestCase):
    def setUp(self):
        self.fixture=original_test.AuditedFixture('test_missing_import_mapping_blocks_audit')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        self.import_dir=self.root/'imports/audit_dependencies'

    def add_import(self, iri, document='imports/core.ttl'):
        path=self.root/document
        g=a.load(path)
        ontology=next(g.subjects(RDF.type,OWL.Ontology))
        g.add((ontology, OWL.imports,URIRef(iri)))
        g.serialize(str(path),format='turtle')
        prep_path=self.root/a.V9PREP
        summary_path=self.root/a.V9SUMMARY
        prep=json.loads(prep_path.read_text())
        summ=json.loads(summary_path.read_text())
        digest=a.digest(path)
        prep['inputs_sha256'][document]=digest
        summ['input_sha256'][document]=digest
        prep_path.write_text(json.dumps(prep))
        summary_path.write_text(json.dumps(summ))

    def write_dependency(self, rel, iri, imports=()):
        path=self.root/rel
        path.parent.mkdir(parents=True,exist_ok=True)
        g=Graph()
        ontology=URIRef(iri)
        g.add((ontology,RDF.type,OWL.Ontology))
        for sub in imports:g.add((ontology,OWL.imports,URIRef(sub)))
        g.serialize(str(path),format='xml' if path.suffix in ('.rdf','.xml','.owl') else 'turtle')

    def test_named_only_preparation_does_not_require_imports(self):
        self.add_import('urn:fixture:unavailable')
        prep=v2.prepare(self.root,'named-only',self.import_dir)
        self.assertEqual(prep['status'],'PREPARED_NOT_REASONED')
        self.assertEqual(len(prep['folds']),4)
        self.assertFalse(prep['full_owl_semantic_entailment_checked'])
        for f in prep['folds']:
            self.assertEqual(a.digest(self.root/f['dataset']['path']),f['dataset']['sha256'])

    def test_named_only_reasoned_pass_does_not_claim_full(self):
        prep=v2.prepare(self.root,'named-only',self.import_dir)
        def mocked(path, holdout, jvm_gb):
            return {'consistent':True,'results':[{'child':c,'parent':p,'status':'NOT_ENTAILED'} for c,p in holdout]}
        with patch.object(a,'reason_with_hermit',side_effect=mocked):
            report=v2.reason(self.root,prep,8)
        self.assertEqual(report['status'],'PASS_HERMIT_NAMED_CONTEXT_ONLY_FULL_CLOSURE_PENDING')
        self.assertFalse(report['full_owl_semantic_entailment_checked'])
        self.assertEqual(len(report['folds']),4)

    def test_full_closure_blocks_missing_and_writes_report(self):
        self.add_import('urn:fixture:unavailable')
        prep=v2.prepare(self.root,'full-closure',self.import_dir)
        self.assertEqual(prep['status'],'BLOCKED_MISSING_LOCAL_OWL_IMPORTS')
        deps=json.loads((self.root/v2.OUT/'import_dependency_report.json').read_text())
        self.assertEqual(deps['unresolved_distinct_iris'],1)
        self.assertEqual(deps['unresolved_iris'][0]['iri'],'urn:fixture:unavailable')
        self.assertFalse((self.root/v2.OUT/'full_closure_entailment_report.json').exists())

    def test_matched_local_dependency_allows_full_preparation(self):
        self.add_import('urn:fixture:unavailable')
        self.write_dependency('imports/audit_dependencies/one.owl','urn:fixture:unavailable')
        prep=v2.prepare(self.root,'full-closure',self.import_dir)
        self.assertEqual(prep['status'],'PREPARED_NOT_REASONED')
        self.assertEqual(prep['import_resolution_status'],'COMPLETE_EXPLICIT_LOCAL_CLOSURE')
        self.assertEqual(len(prep['folds']),4)
        self.assertTrue(prep['source_augmented_crosscheck']['isomorphic_to_augmented'])
        for fold in prep['folds']:
            self.assertTrue(a.ANCHORS.issubset(a.named(a.load(self.root/fold['dataset']['path']))))

    def test_recursive_dependency_must_also_resolve(self):
        self.add_import('urn:fixture:unavailable')
        self.write_dependency('imports/audit_dependencies/one.ttl','urn:fixture:unavailable',['urn:fixture:transitive'])
        prep=v2.prepare(self.root,'full-closure',self.import_dir)
        self.assertEqual(prep['status'],'BLOCKED_MISSING_LOCAL_OWL_IMPORTS')
        deps=json.loads((self.root/v2.OUT/'import_dependency_report.json').read_text())
        self.assertEqual(deps['unresolved_iris'][0]['iri'],'urn:fixture:transitive')

    def test_ambiguous_local_identity_fails_closed(self):
        self.add_import('urn:fixture:unavailable')
        self.write_dependency('imports/audit_dependencies/a.owl','urn:fixture:unavailable')
        self.write_dependency('imports/audit_dependencies/b.owl','urn:fixture:unavailable')
        prep=v2.prepare(self.root,'full-closure',self.import_dir)
        self.assertEqual(prep['status'],'BLOCKED_MISSING_LOCAL_OWL_IMPORTS')
        deps=json.loads((self.root/v2.OUT/'import_dependency_report.json').read_text())
        self.assertEqual(len(deps['ambiguous_iris']),1)

    def test_entailment_is_a_failure_even_in_named_only(self):
        prep=v2.prepare(self.root,'named-only',self.import_dir)
        def mocked(path, holdout, jvm_gb):
            return {'consistent':True,'results':[{'child':c,'parent':p,'status':'LEAKED_ENTAILED' if i==0 else 'NOT_ENTAILED'} for i,(c,p) in enumerate(holdout)]}
        with patch.object(a,'reason_with_hermit',side_effect=mocked):
            report=v2.reason(self.root,prep,8)
        self.assertEqual(report['status'],'FAIL_ENTAILED_OR_UNSATISFIABLE')
        self.assertTrue(report['violations'])

    def test_reasoner_absence_is_never_a_pass(self):
        prep=v2.prepare(self.root,'named-only',self.import_dir)
        with patch.object(a,'reason_with_hermit',side_effect=RuntimeError('HermiT unavailable')):
            with self.assertRaisesRegex(RuntimeError,'HermiT unavailable'):
                v2.reason(self.root,prep,8)
        report=json.loads((self.root/v2.OUT/'named_only_entailment_report.json').read_text())
        self.assertEqual(report['status'],'ERROR_INCOMPLETE_DO_NOT_CLAIM_VALIDATED')


if __name__=='__main__':unittest.main()
