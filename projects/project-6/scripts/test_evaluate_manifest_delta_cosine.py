"""No-network checks for strict ABox-only evaluation and score accounting."""
import hashlib
import json
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_manifest_delta_cosine as e
X=Namespace('https://example.org/test#')


def fixture():
    a=Graph()
    a.add((X.Person,RDF.type,OWL.Class))
    a.add((X.Passenger,RDF.type,OWL.Class))
    a.add((X.Passenger,RDFS.subClassOf,X.Person))
    a.add((X.Person,RDFS.label,Literal('person')))
    a.add((X.Passenger,RDFS.label,Literal('passenger')))
    a.add((X.labeled,RDF.type,OWL.DatatypeProperty))
    a.add((X.labeled,RDFS.label,Literal('has identifier')))
    a.add((X.rel,RDF.type,OWL.ObjectProperty))
    a.add((X.rel,RDFS.label,Literal('possesses')))
    a.add((X.existing_person,RDF.type,OWL.NamedIndividual))
    a.add((X.existing_person,RDF.type,X.Person))
    b=Graph()
    b+=a
    b.add((X.new_class,RDF.type,OWL.Class))
    b.add((X.new_class,RDFS.subClassOf,X.Person))
    b.add((X.new_class,RDFS.label,Literal('new class')))
    b.add((X.new_prop,RDF.type,OWL.ObjectProperty))
    b.add((X.new_prop,RDFS.label,Literal('new relation')))
    for individual,name,cls in [(X.person1,'Sam Example',X.Person),
                                 (X.ticket1,'Physical ticket 001',X.new_class)]:
        b.add((individual,RDF.type,OWL.NamedIndividual))
        b.add((individual,RDF.type,cls))
        b.add((individual,RDFS.label,Literal(name)))
    b.add((X.person1,X.rel,X.ticket1))
    b.add((X.ticket1,X.labeled,Literal('AB123',datatype=XSD.string)))
    b.add((X.existing_person,X.rel,X.ticket1))
    b.add((X.ticket1,X.new_prop,X.person1))
    return a,b


class DeltaTests(unittest.TestCase):
    def test_scopes_only_new_abox_assertions(self):
        base,final=fixture()
        data=e.select_new_assertions(base,final,Graph(),final)
        self.assertEqual(len(data['selected']['individual_class_assertion']),2)
        self.assertEqual(len(data['selected']['object_property_assertion']),3)
        self.assertEqual(len(data['selected']['data_property_assertion']),1)
        self.assertEqual(data['counts']['new_named_individuals_declared'],2)
        self.assertEqual(data['counts']['new_named_classes'],1)
        self.assertEqual(data['counts']['new_named_object_properties'],1)
        self.assertNotIn((X.new_class,RDFS.subClassOf,X.Person),sum(data['selected'].values(),[]))
        self.assertNotIn((X.existing_person,RDF.type,X.Person),sum(data['selected'].values(),[]))

    def test_no_tautological_predicate_in_object_evidence(self):
        base,final=fixture()
        d=e.select_new_assertions(base,final,Graph(),final)
        rows=e.comparisons(d['selected'],final,final)
        row=next(x for x in rows if x['category']=='object_property_assertion' and x['predicate']==str(X.rel))
        self.assertNotIn('possesses',row['evidence'])
        self.assertIn('possesses',row['reference'])

    def test_score_micro_macro_and_checks(self):
        base,final=fixture()
        sel=e.select_new_assertions(base,final,Graph(),final)
        rows=e.comparisons(sel['selected'],final,final)
        def fake_embed(texts):
            # Repeatable synthetic encoder; no claim of actual semantic quality.
            arr=[]
            for text in texts:
                h=hashlib.sha256(text.encode('utf8')).digest()
                arr.append(np.array(list(h[:16]),dtype=float)/255)
            return np.array(arr)
        scored=e.compute_scores(rows,fake_embed)
        summ=e.summarize(scored,0.70)
        self.assertEqual(summ['total_scored_assertions'],6)
        self.assertEqual(sum(x['count'] for x in summ['by_category'].values()),6)
        self.assertAlmostEqual(summ['mean_cosine'],sum(r['cosine'] for r in scored)/6,places=5)
        self.assertTrue(all('evidence' not in x for x in scored))
        self.assertTrue(all('AB123' not in str(x) for x in scored))

    def test_standalone_audit_and_embeddings_no_training(self):
        base,final=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'generated').mkdir();(root/'src').mkdir()
            (root/'src/module.ttl').write_text('@prefix : <https://example.org/test#> .\n',encoding='utf8')
            p0=root/'generated/module_augmented.ttl'
            p1=root/'generated/module_augmented_with_manifest_enriched.ttl'
            p2=root/'generated/manifest_tbox_extension.ttl'
            base.serialize(p0,format='turtle')
            # Preservation is a required pipeline invariant.
            with p1.open('wb') as stream:
                stream.write(p0.read_bytes())
                stream.write(b'\n\n')
                stream.write((final-base).serialize(format='turtle').encode('utf8'))
            Graph().serialize(p2,format='turtle')
            audit=e.run(root,audit_only=True)
            self.assertEqual(audit['status'],'PASS_SCOPED_DELTA_AUDIT_ONLY')
            self.assertIsNone(audit['results'])
            self.assertEqual(audit['original_subclass_axioms_scored'],0)
            def encoder(texts):
                return np.ones((len(texts),4))
            result=e.run(root,embedder=encoder)
            self.assertTrue(result['results']['metric_threshold_met'])
            self.assertEqual(result['results']['mean_cosine'],1)
            self.assertEqual(result['total_selected_assertions'],6)
            self.assertEqual(result['status'],e.STATUS)
            self.assertEqual(set(base),set(Graph().parse(p0,format='turtle')))

    def test_refuses_nonprefix_and_invalid_threshold(self):
        base,final=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'generated').mkdir()
            for name in ['module_augmented.ttl','module_augmented_with_manifest_enriched.ttl','manifest_tbox_extension.ttl']:
                (root/'generated'/name).write_text('not turtle')
            (root/'generated/module_augmented_with_manifest_enriched.ttl').write_text('different content')
            with self.assertRaisesRegex(ValueError,'prefix'):
                e.run(root,audit_only=True)
            with self.assertRaisesRegex(ValueError,'threshold'):
                e.run(root,threshold=1.01,audit_only=True)

if __name__=='__main__': unittest.main()
