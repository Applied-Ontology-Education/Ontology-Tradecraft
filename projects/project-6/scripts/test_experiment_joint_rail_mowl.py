import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from rdflib import Graph, URIRef
from rdflib.namespace import OWL,RDF,RDFS

SCRIPT=Path(__file__).with_name('experiment_joint_rail_mowl.py')
spec=importlib.util.spec_from_file_location('project6_joint_transfer',SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestJointTransfer(unittest.TestCase):
    def test_closure(self):
        edges={('a','b'),('b','c'),('c','d')}
        self.assertIn(('a','d'),mod.closure(edges))
        self.assertNotIn(('d','a'),mod.closure(edges))

    def test_reachable(self):
        edges={('a','b'),('b','c')}
        self.assertTrue(mod.reachable(edges,'a','c'))
        self.assertFalse(mod.reachable(edges,'c','a'))

    def test_no_leakage_dev_split(self):
        edges={(f'c{i}', f'p{j}') for i in range(20) for j in range(3)}
        train,held=mod.internal_dev_split(edges,42,.16)
        self.assertTrue(train)
        self.assertEqual(len(held),round(len(edges)*.16))
        self.assertFalse(held & train)
        self.assertFalse(held & mod.closure(train))
        self.assertEqual(train|held,edges)

    def test_dev_split_repeatable(self):
        edges={(f'c{i}',f'p{j}') for i in range(20) for j in range(3)}
        self.assertEqual(mod.internal_dev_split(edges,42,.16),mod.internal_dev_split(edges,42,.16))

    def test_dev_split_tiny_fails(self):
        with self.assertRaises(ValueError):
            mod.internal_dev_split({('a','b')},42,.2)

    def test_dev_fraction_invalid(self):
        with self.assertRaises(ValueError):
            mod.internal_dev_split({('a','b')},42,.9)

    def test_write_training_graph_keeps_only_declarations(self):
        source=Graph()
        for e in 'abc': source.add((URIRef('urn:'+e),RDF.type,OWL.Class))
        source.add((URIRef('urn:a'),RDFS.subClassOf,URIRef('urn:b')))
        source.add((URIRef('urn:b'),RDFS.subClassOf,URIRef('urn:c')))
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'temp.ttl'
            mod.write_training_graph(source,{('urn:a','urn:b')},output)
            result=mod.rdf(output)
            self.assertEqual(mod.named_pairs(result),{('urn:a','urn:b')})
            self.assertEqual(sum(1 for _ in result.triples((None,RDF.type,OWL.Class))),3)

    def test_hash_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.txt';p.write_text('a')
            before=mod.digest(p)
            self.assertEqual(before,mod.digest(p))
            p.write_text('b')
            self.assertNotEqual(before,mod.digest(p))

    def test_score_stats(self):
        r=[{'cosine':.70},{'cosine':.90},{'cosine':.20}]
        self.assertEqual(mod.stats(r)['count'],3)
        self.assertEqual(mod.stats(r)['mean'],.6)
        self.assertEqual(mod.stats(r)['at_or_above_0_70'],2)

    def test_score_empty(self):
        self.assertIsNone(mod.stats([])['mean'])

    def test_auc(self):
        pos=[{'cosine':.9},{'cosine':.7}]
        ctr=[{'cosine':.1},{'cosine':.2}]
        self.assertEqual(mod.auc_pos_control(pos,ctr),1.)

    def test_auc_tie(self):
        self.assertEqual(mod.auc_pos_control([{'cosine':.5}],[{'cosine':.5}]),.5)

    def test_contols_unasserted_and_repeatable(self):
        classes=['x','y','z','w']
        ex={('x','y'),('x','z')}
        a=mod.unasserted_controls(classes,ex,6,99)
        self.assertEqual(a,mod.unasserted_controls(classes,ex,6,99))
        self.assertFalse(set(a)&ex)
        self.assertTrue(all(x!=y for x,y in a))

    def test_control_selection_penalty(self):
        self.assertLess(mod.select_score(.8,-.1),mod.select_score(.8,.2))
        self.assertEqual(mod.select_score(.8,.2),.8)

    def test_semantic_features_repeatable_and_informative(self):
        index={'http://example.org/rail_train':0,'http://example.org/railway_station':1,
               'http://example.org/aircraft':2}
        desc={'http://example.org/rail_train':'Rail vehicle engine track',
              'http://example.org/railway_station':'Train stopping station platform'}
        a=mod.hashed_text_features(index,desc,256)
        b=mod.hashed_text_features(index,desc,256)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(tuple(a.shape),(3,256))
        self.assertGreater(torch.dot(a[0],a[1]),torch.dot(a[0],a[2]))

    def test_similarity_rows(self):
        vectors=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
        index={'a':0,'b':1,'c':2}
        rows=mod.similarity_rows(vectors,{('a','b'),('a','c')},index,'TEST')
        self.assertEqual([r['cosine'] for r in rows],[1.,0.])

    def test_missing_vector_is_error(self):
        with self.assertRaises(ValueError):
            mod.pair_index([('a','b')],{'a':0})

    def test_input_missing_fails(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                mod.check_inputs(Path(td))

    def test_provenance_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)
            for rel in ('reports/step5_split_report.json','reports/step5_cosine_scores.json'):
                p=r/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({'train_sha256':'not-a-hash'}))
            for rel in ('generated/mowl_rail_train.ttl','generated/mowl_rail_valid.ttl','src/module.ttl','generated/rail_axioms_refined.ttl'):
                p=r/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('')
            with self.assertRaises(ValueError):
                mod.check_inputs(r)

    def test_projection_does_not_write_input_ontology(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            p=root/'src/module.ttl';p.parent.mkdir(parents=True);p.write_text('original')
            g=Graph()
            for c in 'abc':g.add((URIRef('urn:'+c),RDF.type,OWL.Class))
            mod.write_training_graph(g,{('urn:a','urn:b')},root/'generated/dev.ttl')
            self.assertEqual(p.read_text(),'original')

    def test_output_ascii_json_and_replacement(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.json';mod.write_json(p,{'t':'→','x':1})
            self.assertIn('\\u2192',p.read_text())
            self.assertEqual(mod.read_json(p)['x'],1)

if __name__=='__main__':unittest.main()

class TestMatchedControls(unittest.TestCase):
    def test_matches_child_and_excludes_valid_pair(self):
        idx={k:i for i,k in enumerate(['train','material','aircraft','station','role','process','site','bearer'])}
        edges={('train','material'),('role','process')}
        controls=mod.matched_child_controls(idx,edges,set(edges),3,42)
        self.assertEqual(len(controls),6)
        self.assertEqual({x for x,_ in controls},{'train','role'})
        self.assertFalse(set(controls)&edges)

    def test_impossible_controls_fails(self):
        with self.assertRaises(ValueError):
            mod.matched_child_controls({'a':0,'b':1}, {('a','b')}, {('a','b')}, 1,42)

class TestRealisticStep5Split(unittest.TestCase):
    def example_project(self, root):
        # Synthetic 36/4 named edges, with 2 of the 4 validation edges rail.
        source=Graph();refined=Graph(); train=Graph();valid=Graph()
        for i in range(20):
            for gr in (source,refined,train):
                gr.add((URIRef(f'urn:c{i}'),RDF.type,OWL.Class))
        for p in ('urn:p0','urn:p1'):
            for gr in (source,refined,train):
                gr.add((URIRef(p),RDF.type,OWL.Class))
        held=set()
        edges=set()
        for i in range(20):
            for j in (0,1):
                pair=(f'urn:c{i}',f'urn:p{j}')
                edges.add(pair)
                if i<4 and j==1:
                    held.add(pair)
                    valid.add((URIRef(pair[0]),RDFS.subClassOf,URIRef(pair[1])))
                else:
                    train.add((URIRef(pair[0]),RDFS.subClassOf,URIRef(pair[1])))
        for rel,graph in [('src/module.ttl',source),('generated/rail_axioms_refined.ttl',refined),
                          ('generated/mowl_rail_train.ttl',train),('generated/mowl_rail_valid.ttl',valid)]:
            path=root/rel;path.parent.mkdir(exist_ok=True,parents=True);graph.serialize(str(path),format='turtle')
        split={'source_sha256':mod.digest(root/'src/module.ttl'),
               'refined_sha256':mod.digest(root/'generated/rail_axioms_refined.ttl'),
               'train_sha256':mod.digest(root/'generated/mowl_rail_train.ttl'),
               'valid_sha256':mod.digest(root/'generated/mowl_rail_valid.ttl'),
               'train_axiom_pairs':36,'valid_axiom_pairs':4,
               'rail_candidate_holdout_pairs':2,
               'source_import_holdout_pairs':2,
               'validation_fraction_requested':0.2,
               'holdout_pairs':[{'child':a,'parent':b} for a,b in sorted(held)],
               'candidate_pairs':[{'child':a,'parent':b} for a,b in sorted(held)[:2]],
               'import_hashes':{}}
        (root/'reports').mkdir(exist_ok=True)
        mod.write_json(root/'reports/step5_split_report.json',split)
        baseline={'train_sha256':split['train_sha256'],'valid_sha256':split['valid_sha256'],
                  'heldout_rail_pairs':2,'mean_heldout_rail_cosine':.126342,
                  'mean_heldout_source_import_cosine':-.009948,'version':'mowl-v2'}
        mod.write_json(root/'reports/step5_cosine_scores.json',baseline)

    def test_provenance_with_80_20_report(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.example_project(root)
            inp=mod.check_inputs(root)
            self.assertEqual(len(inp['train']),36)
            self.assertEqual(len(inp['valid']),4)
            self.assertEqual(len(inp['rail_hold']),2)

    def test_cannot_reuse_outer_holdout_as_training(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.example_project(root)
            p=root/'generated/mowl_rail_train.ttl'
            graph=mod.rdf(p)
            graph.add((URIRef('urn:c0'),RDFS.subClassOf,URIRef('urn:p1')))
            graph.serialize(destination=str(p),format='turtle')
            # Must refuse a modified split even before model fitting.
            with self.assertRaises(ValueError):
                mod.check_inputs(root)

    def test_preflight_without_mowl(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.example_project(root)
            p=subprocess.run([sys.executable,str(SCRIPT),'--project-root',str(root),'--preflight-only'],
                             capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            data=json.loads(p.stdout)
            self.assertEqual(data['outer_rail_holdouts'],2)
            self.assertTrue(data['development_holdout_pairs']>=3)
