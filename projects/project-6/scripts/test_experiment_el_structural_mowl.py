"""Offline provenance and parser tests. No Java, mOWL, Grok or model training needed."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rdflib import BNode, Graph, URIRef, Literal
from rdflib.collection import Collection
from rdflib.compare import isomorphic
from rdflib.namespace import RDF,RDFS,OWL,XSD

import experiment_el_structural_mowl as el
import experiment_joint_rail_mowl as p5

A=URIRef('urn:test:A');B=URIRef('urn:test:B');C=URIRef('urn:test:C')
D=URIRef('urn:test:D');R=URIRef('urn:test:R')


def some(g, filler=B):
    n=BNode();g.add((n,RDF.type,OWL.Restriction));g.add((n,OWL.onProperty,R));g.add((n,OWL.someValuesFrom,filler));return n


def both(g,*items):
    n=BNode();list_node=BNode();Collection(g,list_node,items)
    g.add((n,OWL.intersectionOf,list_node));g.add((n,RDF.type,OWL.Class));return n


class ParserTests(unittest.TestCase):
    def test_named(self):
        g=Graph(); res=el.expression(g,A)
        self.assertEqual(res[0],'named');self.assertEqual(res[1],{A});self.assertEqual(res[2],set())

    def test_non_el_has_value_excluded(self):
        g=Graph(); n=BNode();g.add((n,OWL.onProperty,R));g.add((n,OWL.hasValue,B))
        self.assertIsNone(el.expression(g,n))

    def test_cardinality_excluded(self):
        g=Graph();n=BNode();g.add((n,OWL.onProperty,R))
        g.add((n,OWL.someValuesFrom,B));g.add((n,OWL.minCardinality,Literal(1)))
        self.assertIsNone(el.expression(g,n))

    def test_data_some_excluded_by_owl_class_datatype(self):
        g=Graph();n=some(g, XSD.string)
        self.assertIsNone(el.expression(g,n))

    def test_nested_some(self):
        g=Graph();nested=some(g,C); outer=some(g,nested)
        parsed=el.expression(g,outer)
        self.assertEqual(parsed[0],'some');self.assertEqual(parsed[1],{C})
        self.assertIn((nested,OWL.someValuesFrom,C),parsed[2])

    def test_bad_cycle(self):
        g=Graph();node=BNode();g.add((node,OWL.onProperty,R));g.add((node,OWL.someValuesFrom,node))
        self.assertIsNone(el.expression(g,node))

    def test_conjunction(self):
        g=Graph();node=both(g,A,B,C)
        parsed=el.expression(g,node)
        self.assertEqual(parsed[0],'intersection')
        self.assertEqual(parsed[1],{A,B,C})
        self.assertEqual(len([t for t in parsed[2] if t[1]==RDF.first]),3)

    def test_invalid_intersection_length(self):
        g=Graph();node=both(g,A)
        self.assertIsNone(el.expression(g,node))

    def test_named_subclass_not_injected(self):
        g=Graph();g.add((A,RDFS.subClassOf,B));rows,c=el.extract_el_axioms(g)
        self.assertEqual(rows,[]);self.assertEqual(c['named_pairs_left_to_split'],1)

    def test_gci1_gci2_gci3_extracted(self):
        g=Graph();g.add((A,RDFS.subClassOf,some(g,B)))
        g.add((some(g,B),RDFS.subClassOf,C))
        g.add((both(g,A,B),RDFS.subClassOf,D))
        rows,_=el.extract_el_axioms(g)
        self.assertEqual(set(r['kind'] for r in rows), {'named_to_some','some_to_named','intersection_to_named'})
        self.assertEqual(len(rows),3)

    def test_equivalentclass_bidirectional(self):
        g=Graph();n=some(g,C);g.add((A,OWL.equivalentClass,n))
        rows,_=el.extract_el_axioms(g)
        self.assertEqual(len(rows),2)
        self.assertIn('named_to_some',{r['kind'] for r in rows})
        self.assertIn('some_to_named',{r['kind'] for r in rows})

    def test_unsupported_expression_skipped_without_partial_axiom(self):
        g=Graph();n=BNode()
        g.add((n,OWL.onProperty,R));g.add((n,OWL.hasValue,C))
        g.add((A,RDFS.subClassOf,n))
        rows,counts=el.extract_el_axioms(g)
        self.assertEqual(rows,[])
        self.assertEqual(counts['unsupported_or_non_el_expression'],1)

    def test_holdout_child_filter(self):
        g=Graph();g.add((A,RDFS.subClassOf,some(g,B)))
        g.add((C,RDFS.subClassOf,some(g,D)))
        rows,_=el.extract_el_axioms(g)
        kept,excluded=el.select_axioms(rows,{(A,B)})
        self.assertEqual(len(kept),1)
        self.assertEqual(excluded['mentions_validation_child'],1)
        self.assertEqual(kept[0]['classes'],{C,D})

    def test_filter_references_nested_holdout_child(self):
        g=Graph();g.add((C,RDFS.subClassOf,some(g,some(g,A))))
        rows,_=el.extract_el_axioms(g)
        kept,excluded=el.select_axioms(rows,{(A,B)})
        self.assertEqual(kept,[])
        self.assertEqual(excluded['mentions_validation_child'],1)

    def test_graph_maintains_named_train_and_restrictions(self):
        g=Graph();g.add((A,RDF.type,OWL.Class));g.add((B,RDF.type,OWL.Class))
        g.add((C,RDF.type,OWL.Class));g.add((A,RDFS.subClassOf,B))
        x=Graph();x.add((C,RDFS.subClassOf,some(x,D)))
        rows,_=el.extract_el_axioms(x)
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'a.ttl'
            summary=el.build_training_graph(g,{(str(A),str(B))},rows,dest)
            generated=p5.rdf(dest)
            self.assertEqual(p5.named_pairs(generated),{(str(A),str(B))})
            self.assertEqual(summary['additional_owl_axioms'],1)
            self.assertIn((C,RDFS.subClassOf,None),[(s,p,None) for s,p,o in generated if p==RDFS.subClassOf and s==C])

    def test_deterministic_file_bytes_for_distinct_blank_node_id(self):
        g=Graph();g.add((A,RDF.type,OWL.Class));g.add((B,RDF.type,OWL.Class))
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'test.ttl'
            for i in range(2):
                x=Graph();x.add((A,RDFS.subClassOf,some(x,B)))
                rows,_=el.extract_el_axioms(x)
                info=el.build_training_graph(g,set(),rows,dest)
                if i==0:first=info['sha256']
                else:self.assertEqual(first,info['sha256'])

    def test_no_holdout_pair_allowed_in_build(self):
        g=Graph();g.add((A,RDF.type,OWL.Class));g.add((B,RDF.type,OWL.Class))
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'train.ttl'
            info=el.build_training_graph(g,{(str(A),str(B))},[],file)
            self.assertEqual(info['named_training_pairs'],1)
            self.assertNotIn((str(B),str(A)),p5.named_pairs(p5.rdf(file)))

    def test_build_does_not_mutate_baseline_graph(self):
        base=Graph();base.add((A,RDFS.subClassOf,B));snapshot=set(base)
        with tempfile.TemporaryDirectory() as tmp:
            el.build_training_graph(base,{(str(A),str(B))},[],Path(tmp)/'test.ttl')
        self.assertEqual(snapshot,set(base))

    def test_real_source_non_el_gc_as_are_counted(self):
        f=Path('/mnt/data/module(2).ttl')
        if not f.is_file():self.skipTest('Source fixture unavailable')
        g=p5.rdf(f)
        rows,why=el.extract_el_axioms(g)
        self.assertEqual(why['unsupported_or_non_el_expression'],1)
        self.assertEqual(len(rows),3)
        self.assertTrue(all(r['kind']=='nested_el' for r in rows))

    def test_report_always_rejects_bad_import_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'imports').mkdir()
            (root/'imports'/'a.ttl').write_text('@prefix : <urn:test:> . :A a <http://www.w3.org/2002/07/owl#Class> .')
            with self.assertRaisesRegex(ValueError,'provenance'):
                el.collect_approved_owl(root,{'source':root/'imports'/'a.ttl',
                    'refined':root/'imports'/'a.ttl',
                    'split':{'import_hashes':{'imports/a.ttl':'wrong'}}})

    def test_base_checks_enforces_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=root/'source.ttl';path.write_text('hello')
            inputs={'source':path,'refined':path,'path_train':path,'path_valid':path,
                    'split':{k:'incorrect' for k in ('source_sha256','refined_sha256','train_sha256','valid_sha256')}}
            with self.assertRaisesRegex(RuntimeError,'changed'):
                el.base_checks(root,inputs)


if __name__=='__main__':unittest.main()

class PrepareIntegrationTests(unittest.TestCase):
    def test_prepare_isolated_output_and_provenance(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for sub in ('src','generated','reports','imports'):(root/sub).mkdir()
            source=root/'src/module.ttl'
            refine=root/'generated/rail_axioms_refined.ttl'
            train=root/'generated/mowl_rail_train.ttl'
            valid=root/'generated/mowl_rail_valid.ttl'
            imp=root/'imports/test.ttl'
            g_source=Graph();g_source.add((A,RDF.type,OWL.Class));g_source.add((B,RDF.type,OWL.Class));g_source.add((C,RDF.type,OWL.Class));g_source.add((D,RDF.type,OWL.Class))
            g_source.add((A,RDFS.subClassOf,C));g_source.serialize(destination=source,format='turtle')
            g_refined=Graph();g_refined.add((D,RDF.type,OWL.Class));g_refined.serialize(destination=refine,format='turtle')
            g_train=Graph()
            for node in [A,B,C,D]:g_train.add((node,RDF.type,OWL.Class))
            g_train.add((A,RDFS.subClassOf,C));g_train.add((C,RDFS.subClassOf,D))
            g_train.serialize(destination=train,format='turtle')
            g_valid=Graph();g_valid.add((B,RDFS.subClassOf,A));g_valid.serialize(destination=valid,format='turtle')
            g_import=Graph();g_import.add((C,RDFS.subClassOf,some(g_import,D)))
            g_import.serialize(destination=imp,format='turtle')
            split={'source_sha256':p5.digest(source),'refined_sha256':p5.digest(refine),
                   'train_sha256':p5.digest(train),'valid_sha256':p5.digest(valid),
                   'import_hashes':{'imports/test.ttl':p5.digest(imp)}}
            inp={'source':source,'refined':refine,'path_train':train,'path_valid':valid,'split':split,
                 'train':{(str(A),str(C)),(str(C),str(D))},
                 'valid':{(str(B),str(A))},'rail_hold':{(str(B),str(A))},'base_hold':set()}
            dev_train={(str(A),str(C))};dev_hold={(str(C),str(D))}
            with patch.object(el.p5,'check_inputs',return_value=inp),\
                 patch.object(el.p5,'internal_dev_split',return_value=(dev_train,dev_hold)):
                actual,dt,dh,diag=el.prepare(root,42,.16)
            self.assertEqual(diag['outer']['additional_owl_axioms'],1)
            self.assertEqual(diag['development']['additional_owl_axioms'],0) # C is dev heldout child
            self.assertEqual(diag['split']['outer_rail_holdout'],1)
            self.assertTrue((root/'reports/step5_el_structure_diagnostic.json').is_file())
            self.assertTrue((root/'generated/mowl_el_structural_outer_train.ttl').is_file())
            self.assertEqual(p5.digest(source),split['source_sha256'])
            self.assertEqual(p5.digest(train),split['train_sha256'])
            self.assertEqual(p5.named_pairs(p5.rdf(root/'generated/mowl_el_structural_outer_train.ttl')),inp['train'])

    def test_missing_project_dep_reports_clear_error(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):
                p5.check_inputs(Path(t))

class FreezeRegressionTests(unittest.TestCase):
    def test_diagnose_never_canonicalizes_or_writes_training_ontologies(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for sub in ('src','generated','reports','imports'):(root/sub).mkdir()
            source=root/'src/module.ttl'; refined=root/'generated/rail_axioms_refined.ttl'
            train=root/'generated/mowl_rail_train.ttl'; valid=root/'generated/mowl_rail_valid.ttl'
            imported=root/'imports/test.ttl'
            g=Graph();g.add((A,RDF.type,OWL.Class));g.add((C,RDFS.subClassOf,some(g,B)))
            g.serialize(destination=source,format='turtle')
            Graph().serialize(destination=refined,format='turtle')
            gt=Graph()
            for node in (A,B,C,D):gt.add((node,RDF.type,OWL.Class))
            gt.add((A,RDFS.subClassOf,B));gt.add((B,RDFS.subClassOf,C))
            gt.serialize(destination=train,format='turtle')
            gv=Graph();gv.add((D,RDFS.subClassOf,A));gv.serialize(destination=valid,format='turtle')
            Graph().serialize(destination=imported,format='turtle')
            split={'source_sha256':p5.digest(source),'refined_sha256':p5.digest(refined),
                   'train_sha256':p5.digest(train),'valid_sha256':p5.digest(valid),
                   'import_hashes':{'imports/test.ttl':p5.digest(imported)}}
            inputs={'source':source,'refined':refined,'path_train':train,'path_valid':valid,
                    'split':split,'train':{(str(A),str(B)),(str(B),str(C))},
                    'valid':{(str(D),str(A))},'rail_hold':{(str(D),str(A))},'base_hold':set()}
            with patch.object(el.p5,'check_inputs',return_value=inputs),\
                 patch.object(el.p5,'internal_dev_split',return_value=({(str(A),str(B))},{(str(B),str(C))})),\
                 patch.object(el,'build_training_graph',side_effect=AssertionError('must not build')):
                report=el.diagnose(root,42,.16)
            self.assertEqual(report['status'],'PASS_EL_DIAGNOSE_ONLY')
            self.assertFalse(report['training_graphs_written'])
            self.assertFalse((root/'generated/mowl_el_structural_outer_train.ttl').exists())
            self.assertFalse((root/'generated/mowl_el_structural_dev_train.ttl').exists())
            self.assertEqual(report['outer']['additional_owl_axioms'],1)

    def test_semantically_identical_intersection_order_has_same_hash(self):
        base=Graph()
        with tempfile.TemporaryDirectory() as t:
            dest=Path(t)/'out.ttl'
            output=[]
            for items in [(A,B,C),(C,B,A)]:
                src=Graph()
                src.add((D,RDFS.subClassOf,both(src,*items)))
                rows,_=el.extract_el_axioms(src)
                result=el.build_training_graph(base,set(),rows,dest)
                output.append(result['sha256'])
            self.assertEqual(output[0],output[1])

    def test_canonicalization_is_not_used_anywhere_in_fast_writer(self):
        self.assertFalse(hasattr(el,'to_canonical_graph'))
        data=Graph();data.add((A,RDFS.subClassOf,both(data,B,C)))
        rows,_=el.extract_el_axioms(data)
        with tempfile.TemporaryDirectory() as t:
            el.build_training_graph(Graph(),set(),rows,Path(t)/'out.ttl')

    def test_diagnostic_timeout_stops_child(self):
        import subprocess
        fake=unittest.mock.Mock()
        fake.wait.side_effect=[subprocess.TimeoutExpired(['cmd'],1),0]
        with patch.object(el.subprocess,'Popen',return_value=fake),\
             patch.object(el.sys,'argv',['script','--diagnose-only','--diagnose-timeout-seconds','1']):
            code=el.main()
        self.assertEqual(code,124)
        fake.kill.assert_called_once()

class DevelopmentOnlyRegressionTests(unittest.TestCase):
    """No JVM/mOWL required. Guard outer-holdout isolation and legacy behavior."""

    @staticmethod
    def fixture(root):
        for sub in ('src','generated','reports','imports'):(root/sub).mkdir(exist_ok=True)
        paths={k:root/rel for k,rel in {
            'source':'src/module.ttl','refined':'generated/rail_axioms_refined.ttl',
            'path_train':'generated/mowl_rail_train.ttl',
            'path_valid':'generated/mowl_rail_valid.ttl'}.items()}
        g=Graph()
        for n in (A,B,C,D):g.add((n,RDF.type,OWL.Class))
        g.add((A,RDFS.subClassOf,B));g.add((B,RDFS.subClassOf,C))
        g.serialize(destination=paths['path_train'],format='turtle')
        Graph().serialize(destination=paths['path_valid'],format='turtle')
        Graph().serialize(destination=paths['source'],format='turtle')
        Graph().serialize(destination=paths['refined'],format='turtle')
        sha={f'{k}_sha256':p5.digest(paths[k]) for k in ('source','refined')}
        sha['train_sha256']=p5.digest(paths['path_train'])
        sha['valid_sha256']=p5.digest(paths['path_valid'])
        sha['import_hashes']={}
        return dict(paths,split=sha,
                    train={(str(A),str(B)),(str(B),str(C))},
                    valid={(str(D),str(A))},
                    rail_hold={(str(D),str(A))},base_hold=set())

    def test_prepare_development_only_does_not_write_outer_training_graph(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            inputs=self.fixture(root)
            struct=Graph()
            struct.add((C,RDFS.subClassOf,some(struct,A)))
            entries,_=el.extract_el_axioms(struct)
            self.assertEqual(len(entries),1)
            # Existing outer file and report are sentinel user artifacts and
            # must remain unchanged during a development-only run.
            outer=root/'generated/mowl_el_structural_outer_train.ttl'
            outer.write_bytes(b'KEEP THIS OLD OUTER GRAPH')
            with patch.object(el.p5,'check_inputs',return_value=inputs),\
                 patch.object(el.p5,'internal_dev_split',return_value=({(str(A),str(B))},{(str(B),str(C))})),\
                 patch.object(el,'collect_approved_owl',return_value=(entries,{})):
                actual,devtrain,devhold,diag=el.prepare(root,42,.16,development_only=True)
            self.assertEqual(outer.read_bytes(),b'KEEP THIS OLD OUTER GRAPH')
            self.assertFalse(diag['outer']['training_graph_written'])
            self.assertTrue(diag['development']['training_graph_written'])
            self.assertEqual(diag['split']['outer_holdout'],1)
            self.assertTrue((root/'generated/mowl_el_structural_dev_train.ttl').is_file())
            self.assertEqual(p5.named_pairs(p5.rdf(root/'generated/mowl_el_structural_dev_train.ttl')),devtrain)
            self.assertEqual(diag['development']['additional_owl_axioms'],1)

    def test_run_development_only_skips_outer_fit_normalization_and_outputs(self):
        from types import SimpleNamespace
        import sys
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            inputs=self.fixture(root)
            old_report=root/'reports/step5_el_structural_comparison.json'
            old_vectors=root/'generated/mowl_el_structural_vectors.npz'
            old_report.write_bytes(b'KEEP OLD REPORT')
            old_vectors.write_bytes(b'KEEP OLD VECTORS')
            dev_train={(str(A),str(B))}; dev_hold={(str(B),str(C))}
            diag={'outer':{'additional_owl_axioms':3},
                  'development':{'additional_owl_axioms':2,'sha256':'dev_hash'},
                  'input_sha256':{'source_sha256':'abc'}}
            result={'config':{'dimension':50},'mean_cosine':0.20,
                    'positive_control_gap':0.09,'positive_control_auc':0.68,
                    'train_positive_mean_cosine':0.35}
            fake_mod=SimpleNamespace(init_jvm=lambda size:None)
            def training(*args,**kwargs):
                self.assertEqual(args[3],dev_hold)
                self.assertEqual(args[-1],'el_dev_1')
                self.assertNotIn('el_outer',args[-1])
                return result.copy(),[],[],None,{}
            with patch.dict(sys.modules,{'mowl':fake_mod}),\
                 patch.object(el,'prepare',return_value=(inputs,dev_train,dev_hold,diag)) as prep,\
                 patch.object(el,'real_gci_counts',return_value={'gci1':1,'gci2':2,'gci3':3}) as counts,\
                 patch.object(el.p5,'collect_text',return_value={}),\
                 patch.object(el.p5,'train_once',side_effect=training) as fit,\
                 patch.object(el,'base_checks'):
                report=el.run(root,configs=1,tune_epochs=50,development_only=True)
            self.assertTrue(prep.call_args.kwargs['development_only'])
            self.assertEqual(fit.call_count,1)
            self.assertEqual(counts.call_count,1) # Never normalize outer graph
            self.assertEqual(report['status'],'DEVELOPMENT_ONLY_COMPLETE')
            self.assertIs(report['outer_holdout_evaluated'],False)
            self.assertIs(report['outer_final_fit_run'],False)
            self.assertIs(report['outer_training_graph_written'],False)
            self.assertEqual(old_report.read_bytes(),b'KEEP OLD REPORT')
            self.assertEqual(old_vectors.read_bytes(),b'KEEP OLD VECTORS')
            dev_report=root/'reports/step5_el_structural_development.json'
            self.assertTrue(dev_report.is_file())
            read=p5.read_json(dev_report)
            self.assertEqual(read['tune_epochs'],50)
            self.assertEqual(read['chosen_dev_result']['mean_cosine'],0.2)
            self.assertNotIn('rail_holdout',read)
            self.assertNotIn('all_holdout',read)

    def test_development_mode_requires_nonzero_extra_el_and_doesnt_load_jvm(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            diag={'outer':{'additional_owl_axioms':4},
                  'development':{'additional_owl_axioms':0,'sha256':'unused'},
                  'input_sha256':{}}
            with patch.object(el,'prepare',return_value=({},set(),set(),diag)),\
                 self.assertRaisesRegex(ValueError,'development leakage safeguards'):
                el.run(root,configs=1,development_only=True)

    def test_cli_development_mode_uses_separate_report_without_outer_data(self):
        fake_report={'status':'DEVELOPMENT_ONLY_COMPLETE',
                     'chosen_dev_result':{'mean_cosine':0.21},
                     'normal_forms':{'development':{'gci2':1}}}
        with patch.object(el,'run',return_value=fake_report) as run,\
             patch.object(el.sys,'argv',['script','--development-only','--configs','1','--tune-epochs','50']):
            self.assertEqual(el.main(),0)
        self.assertTrue(run.call_args.kwargs['development_only'])
        self.assertEqual(run.call_args.kwargs['tune_epochs'],50)

    def test_cli_disallows_strict_in_development_only_mode(self):
        with patch.object(el.sys,'argv',['script','--development-only','--strict']):
            with self.assertRaises(SystemExit) as caught:
                el.main()
        self.assertEqual(caught.exception.code,2)
