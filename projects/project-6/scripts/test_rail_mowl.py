"""Offline unit and integration tests. Mock only the external mOWL/xAI calls;
assert the split, provenance, leakage, scoring and preservation using real RDFLib.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL, SKOS

import prepare_rail_mowl as pre
import evaluate_rail_mowl as eva


def make_fixture(root: Path):
    for folder in ("src","imports","generated","reports"):
        (root/folder).mkdir(parents=True,exist_ok=True)
    BFO="http://purl.obolibrary.org/obo/"
    base="https://example.org/testing/"
    g=Graph()
    edges={(base+f"C{i}",base+f"C{i+1}") for i in range(13)}
    edges|={(base+f"D{i}",base+f"C{i}") for i in range(9)}
    edges|={(pre.AGENT_NS+"passenger",BFO+"BFO_0000023")}
    for a,b in edges:
        g.add((URIRef(a),RDFS.subClassOf,URIRef(b)))
    g.add((URIRef(pre.AGENT_NS+"person"),RDF.type,OWL.Class))
    r=Graph()
    railway={
        (pre.BASE_NS+"train",BFO+"BFO_0000040"),
        (pre.BASE_NS+"passenger_train",pre.BASE_NS+"train"),
        (pre.BASE_NS+"rail_journey",BFO+"BFO_0000015"),
        (pre.BASE_NS+"rail_journey_plan",BFO+"BFO_0000031"),
        (pre.BASE_NS+"railway_station",BFO+"BFO_0000040"),
        (pre.BASE_NS+"rail_ticket",BFO+"BFO_0000031"),
        (pre.BASE_NS+"rail_reservation_information",BFO+"BFO_0000031"),
        (pre.BASE_NS+"rail_passenger_role",BFO+"BFO_0000023"),
        pre.RAIL_ROLE_PAIR
    }
    for a,b in railway:
        r.add((URIRef(a),RDFS.subClassOf,URIRef(b)))
        r.add((URIRef(a),RDF.type,OWL.Class))
    g.serialize(root/"src/module.ttl",format="turtle")
    r.serialize(root/"generated/rail_axioms_refined.ttl",format="turtle")
    import_path=root/"imports/fake.ttl"
    import_path.write_text(f'<{base}A> <{RDFS.subClassOf}> <{base}C13> .\n',encoding="utf-8")
    imports={"imports/fake.ttl":pre.file_hash(import_path)}
    proposal={"provenance":{"local_import_hashes":imports}}
    (root/"generated/rail_pattern_proposal.json").write_text(json.dumps(proposal),encoding="utf-8")
    sha={"source":pre.file_hash(root/"src/module.ttl"),"refined":pre.file_hash(root/"generated/rail_axioms_refined.ttl")}
    summary={"status":"PASS_QC_AND_SHACL_UNAPPROVED","sparql_passed":True,
       "shacl":{"executed":True,"conforms":True},
       "source_sha256":sha["source"],"refined_sha256":sha["refined"],
       "merged_into_source":False,"generated_query_count":1}
    (root/"reports/step4_summary.json").write_text(json.dumps(summary),encoding="utf-8")
    (root/"reports/step4_qc_results.json").write_text(json.dumps({"checks":[{"status":"PASS"}],"status":summary["status"]}),encoding="utf-8")
    (root/"reports/step4_shacl_report.txt").write_text("Validation Report\nConforms: True\n",encoding="utf-8")
    return g,r,proposal,sha


class Step5Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source,self.refined,self.proposal,self.hashes=make_fixture(self.root)
        self.patcher=patch.object(pre,"require_inputs",return_value=(
            self.source,Graph(),self.refined,self.proposal,{}, {},self.hashes))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def fake_train(self, path, *, epochs,dim,seed):
        graph=pre.load_ontology(path)
        classes=sorted(str(x) for x in graph.subjects(RDF.type,OWL.Class))
        vec=np.random.default_rng(seed).standard_normal((len(classes),dim)).astype("float32")
        return {iri:i for i,iri in enumerate(classes)}, vec, {"test_backend":"mocked_external_mowl"}

    def test_role_holdout_and_no_leakage(self):
        x=pre.plan(self.root)
        train=pre.pairs(pre.load_ontology(self.root/"generated/mowl_rail_train.ttl"))
        valid=pre.pairs(pre.load_ontology(self.root/"generated/mowl_rail_valid.ttl"))
        self.assertEqual(valid-train,valid)
        self.assertIn(pre.RAIL_ROLE_PAIR,valid)
        self.assertNotIn(pre.RAIL_ROLE_PAIR,train)
        self.assertEqual(x["rail_candidate_holdout_pairs"],1)
        for p in valid:
            self.assertFalse(pre.reachable(train,*p))

    def test_split_deterministic(self):
        a=pre.plan(self.root)
        b1=(self.root/"generated/mowl_rail_train.ttl").read_bytes()
        b2=(self.root/"generated/mowl_rail_valid.ttl").read_bytes()
        b=pre.plan(self.root)
        self.assertEqual(a,b)
        self.assertEqual(b1,(self.root/"generated/mowl_rail_train.ttl").read_bytes())
        self.assertEqual(b2,(self.root/"generated/mowl_rail_valid.ttl").read_bytes())

    def test_source_unmodified(self):
        pre.plan(self.root)
        self.assertEqual(self.hashes["source"],pre.file_hash(self.root/"src/module.ttl"))

    def test_step4_failure_not_accepted(self):
        p=self.root/"reports/step4_summary.json"
        s=json.loads(p.read_text());s["shacl"]["conforms"]=False;p.write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError,"Step 4"):
            pre.plan(self.root)

    def test_import_hash_change_fails(self):
        (self.root/"imports/fake.ttl").write_text('<x:a> <x:b> <x:c> .')
        with self.assertRaisesRegex(ValueError,"Local imports"):
            pre.plan(self.root)

    def test_holdout_excluded_if_transitive(self):
        a,b,c="http://z/a","http://z/b","http://z/c"
        pairs={(a,b),(b,c)}
        self.assertTrue(pre.reachable(pairs,a,c))
        self.assertFalse(pre.reachable(pairs,c,a))

    def test_cosine_math_and_invalid(self):
        self.assertEqual(eva.cosine(np.array([1.,0.]),np.array([1.,0.])),1)
        self.assertEqual(eva.cosine(np.array([1.,0.]),np.array([-1.,0.])),-1)
        self.assertIsNone(eva.cosine(np.array([0.,0.]),np.array([0.,1.])))

    def test_score_cold_start_is_not_fabricated(self):
        split=pre.plan(self.root)
        model=eva.evaluate(self.root,epochs=1,train_function=self.fake_train)
        self.assertEqual(model["status"],"PASS_COSINE_ONLY")
        hybrid=json.loads((self.root/"reports/step5_hybrid_review.json").read_text())
        self.assertEqual(hybrid["hybrid_scored_pairs"],0)
        self.assertEqual(sum(r["cosine"] is not None for r in hybrid["candidate_relationships"]),1)
        self.assertTrue(all(not row["approved"] for row in hybrid["candidate_relationships"]))

    def test_hybrid_llm_scores_only_held_out(self):
        def llm(request):
            return {"ratings":[{"id":r["id"],"plausibility":.8,"rationale":"Plausible subclass under BFO role.","concern":"Human review pending"} for r in request["candidates"]]}
        result=eva.evaluate(self.root,epochs=2,with_llm=True,train_function=self.fake_train,llm_function=llm)
        self.assertEqual(result["status"],"PASS_HYBRID_REVIEW_ONLY")
        hybrid=json.loads((self.root/"reports/step5_hybrid_review.json").read_text())
        self.assertEqual(hybrid["hybrid_scored_pairs"],1)
        self.assertEqual(hybrid["unscored_independent_cosine"],8)

    def test_invalid_llm_result_rejected(self):
        req={"candidates":[{"id":"edge-00"}]}
        with self.assertRaisesRegex(ValueError,"missing"):
            eva.validate_llm_reply({"ratings":[]},req)

    def test_save_and_reuse_vectors_without_train(self):
        eva.evaluate(self.root,epochs=2,train_function=self.fake_train)
        # Cached vectors should not depend on the presence of installed mOWL.
        reused=eva.evaluate(self.root,epochs=2,reuse_vectors=True)
        self.assertEqual(reused["status"],"PASS_COSINE_ONLY")

    def test_stale_cache_rejected(self):
        eva.evaluate(self.root,epochs=2,train_function=self.fake_train)
        with self.assertRaisesRegex(ValueError,"stale"):
            eva.evaluate(self.root,epochs=3,reuse_vectors=True)

if __name__=='__main__': unittest.main()
