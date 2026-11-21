import importlib.util
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rdflib import Graph, URIRef, BNode, Literal
from rdflib.namespace import RDF,RDFS,OWL

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/audit_v9_owl_entailment.py'
spec=importlib.util.spec_from_file_location('audit',SCRIPT)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class AuditedFixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        def write(rel,g):
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True)
            g.serialize(str(p),format='xml' if p.suffix=='.owl' else 'turtle')
        src=Graph();ref=Graph()
        src_ont=URIRef('urn:fixture:src')
        src.add((src_ont,RDF.type,OWL.Ontology))
        src.add((src_ont,OWL.imports,URIRef('urn:fixture:import:0')))
        # Complex anonymous OWL restriction must survive the union
        x=BNode();src.add((URIRef(a.BASE+'rail_journey'),RDFS.subClassOf,x))
        src.add((x,RDF.type,OWL.Restriction))
        src.add((x,OWL.onProperty,URIRef(a.BASE+'has_train_participant')))
        src.add((x,OWL.someValuesFrom,URIRef(a.BASE+'train')))
        for fold in a.FOLDS:
            for child,parent in fold:
                ref.add((URIRef(child),RDF.type,OWL.Class))
                ref.add((URIRef(parent),RDF.type,OWL.Class))
                ref.add((URIRef(child),RDFS.subClassOf,URIRef(parent)))
        for child,parent in a.ANCHORS:
            ref.add((URIRef(child),RDF.type,OWL.Class))
            ref.add((URIRef(parent),RDF.type,OWL.Class))
        write('src/module.ttl',src)
        write('generated/rail_axioms_refined.ttl',ref)
        aug=Graph()
        for t in src:aug.add(t)
        for t in ref:aug.add(t)
        write('generated/module_augmented.ttl',aug)
        for i,rel in enumerate(a.IMPORTS):
            imp=Graph();imp.add((URIRef(f'urn:fixture:import:{i}'),RDF.type,OWL.Ontology))
            write(rel,imp)
        review={'human_review_status':'SEVEN_APPROVED_FOR_EXPERIMENT_2026_11_19',
                'target_holdouts_must_remain_excluded':True,
                'approved_anchors':[{'child':c,'parent':p} for c,p in sorted(a.ANCHORS)]}
        rp=self.root/a.REVIEW;rp.parent.mkdir(parents=True,exist_ok=True)
        rp.write_text(json.dumps(review))
        files=['src/module.ttl','generated/module_augmented.ttl']+list(a.IMPORTS)
        for file in files:
            p=self.root/file
            # this fixture creates them above
        hashes={rel:a.digest(self.root/rel) for rel in files}
        folds=[]
        for i,held in enumerate(a.FOLDS,1):
            edges=set().union(*[set(f) for f in a.FOLDS])-set(held)
            edges|=a.ANCHORS
            outer=Graph()
            for child,parent in edges:outer.add((URIRef(child),RDFS.subClassOf,URIRef(parent)))
            rel=f'generated/experiments/rail_mowl_v9_reviewed/fold_{i}_outer_train.ttl'
            write(rel,outer)
            innerrel=f'generated/experiments/rail_mowl_v9_reviewed/fold_{i}_inner_train.ttl'
            write(innerrel,outer)
            folds.append({'fold':i,'outer_training_path':rel,'outer_sha256':a.digest(self.root/rel),'outer_training_pairs':len(edges),
                          'inner_training_path':innerrel,'inner_sha256':a.digest(self.root/innerrel),
                          'heldout':[{'child':c,'parent':p} for c,p in held]})
        prep={'mode':'SEVEN_APPROVED_ANCHORS_NESTED_DEVELOPMENT','folds':folds,'review_file_sha256':a.digest(rp),'inputs_sha256':hashes}
        summary={'experiment_mode':'NESTED_DEV_SEVEN_APPROVED_ANCHORS','configs_per_fold':10,
                 'folds_completed':4,'outer_test_sample_size':8,'projection_fallback_to_native_folds':4,
                 'native_8_class_mean':0.6037136092782021,'native_auc':.953125,
                 'native_gap':.4515440862160176,'review_file_sha256':a.digest(rp),'input_sha256':hashes}
        for rel,value in ((a.V9PREP,prep),(a.V9SUMMARY,summary)):
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))

    def test_4_fold_prepare_preserves_axioms_and_excludes_targets(self):
        info=a.prepare(self.root)
        self.assertEqual(len(info['folds']),4)
        self.assertTrue(info['source_augmented_crosscheck']['isomorphic_to_augmented'])
        for f in info['folds']:
            g=Graph().parse(str(self.root/f['full_context']['file']),format='xml')
            self.assertEqual(len(list(g.subjects(OWL.onProperty,URIRef(a.BASE+'has_train_participant')))),1)
            self.assertFalse(set(a.FOLDS[f['fold']-1])&(a.named(g)|a.closure(a.named(g))))
            self.assertTrue(a.ANCHORS.issubset(a.named(g)))

    def test_missing_import_mapping_blocks_audit(self):
        g=a.load(self.root/'src/module.ttl')
        g.add((URIRef('urn:fixture:src'),OWL.imports,URIRef('urn:fixture:unavailable')))
        g.serialize(self.root/'src/module.ttl',format='turtle')
        parts={'src/module.ttl':g}
        for rel in a.IMPORTS:parts[rel]=a.load(self.root/rel)
        missing=a.import_map_validation(parts)['unresolved_imports']
        self.assertEqual(missing,[{'document':'src/module.ttl','iri':'urn:fixture:unavailable'}])

    def test_report_catches_entailment_and_unsat(self):
        info=a.prepare(self.root)
        def fake_reason(p,targets,mem):
            leak='fold_2_full_context.owl' in str(p)
            return {'consistent':True,'owlapi_logical_axiom_count':9,'reasoner':'HermiT',
                    'results':[{'child':c,'parent':pp,'status':'LEAKED_ENTAILED' if leak else 'NOT_ENTAILED',
                                'entailed':leak,'child_satisfiable':True} for c,pp in targets]}
        with patch.object(a,'reason_with_hermit',side_effect=fake_reason):
            r=a.run(self.root,info,8)
        self.assertEqual(r['status'],'FAIL_ENTAILMENT_OR_UNSATISFIABLE')
        self.assertTrue(r['violations'])

    def test_incomplete_reasoner_is_never_a_pass(self):
        info=a.prepare(self.root)
        with patch.object(a,'reason_with_hermit',side_effect=RuntimeError('HermiT unavailable')):
            with self.assertRaisesRegex(RuntimeError,'HermiT unavailable'):
                a.run(self.root,info,8)
        r=json.loads((self.root/a.OUT/'entailment_report.json').read_text())
        self.assertEqual(r['status'],'ERROR_INCOMPLETE_DO_NOT_CLAIM_VALIDATED')

if __name__=='__main__':unittest.main()
