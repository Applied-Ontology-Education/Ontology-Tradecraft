"""Project 6 synthetic manifest ingestion regression tests; no Grok/Java needed."""
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from copy import deepcopy

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, OWL

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location('manifest_ingest', HERE.with_name('ingest_passenger_manifest.py'))
ING = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = ING
SPEC.loader.exec_module(ING)


class ManifestIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ontology = ROOT / 'generated' / 'module_augmented.ttl'
        cls.template = ROOT / 'templates' / 'passenger_manifest_template.xlsx'
        if not cls.ontology.is_file():
            raise unittest.SkipTest('Run Project 6 Step 8 to create generated/module_augmented.ttl')
        if not cls.template.is_file():
            raise unittest.SkipTest('Missing synthetic passenger_manifest_template.xlsx')
        cls.records = ING.parse_manifest(cls.template)
        cls.base = Graph().parse(cls.ontology, format='turtle')

    def test_01_parses_both_sheets(self):
        self.assertEqual(len(self.records), 5)
        self.assertEqual(sum(x.mode == 'air' for x in self.records),2)
        self.assertEqual(sum(x.mode == 'rail' for x in self.records),3)

    def test_02_cross_mode_person_identity(self):
        air = next(x for x in self.records if x.values['person_id'] == 'P-001' and x.mode=='air')
        rail = next(x for x in self.records if x.values['person_id'] == 'P-001' and x.mode=='rail')
        self.assertEqual(ING._sid('person',air.values['person_id']),ING._sid('person',rail.values['person_id']))

    def test_03_ontological_schema_exists(self):
        ING.verify_ontology(self.base,self.records)

    def test_04_duplicate_entry_rejected(self):
        with self.assertRaisesRegex(ING.ManifestError,'duplicate entry_id'):
            ING.verify_records([self.records[0],self.records[0]])

    def test_05_conflicting_trip_rejected(self):
        v = dict(self.records[1].values)
        v['departure_name'] = 'Contradictory airport'
        with self.assertRaisesRegex(ING.ManifestError,'trip_id has conflicting'):
            ING.verify_records([self.records[0],replace(self.records[1],values=v)])

    def test_06_conflicting_person_name_rejected(self):
        v = dict(self.records[2].values)
        v['last_name']='Different'
        with self.assertRaisesRegex(ING.ManifestError,'conflicting names'):
            ING.verify_records([self.records[0],replace(self.records[2],values=v)])

    def test_07_conflicting_aircraft_rejected(self):
        v = dict(self.records[1].values)
        v['aircraft_tail_number'] = 'DIFFERENT'
        with self.assertRaisesRegex(ING.ManifestError, 'conflicting aircraft'):
            ING.verify_records([self.records[0], replace(self.records[1], values=v)])

    def test_08_date_time_requires_offset(self):
        with self.assertRaisesRegex(ING.ManifestError,'time zone offset'):
            ING._iso_datetime('2026-11-19T09:00:00','test')

    def test_08_air_boarding_pass_is_carrier(self):
        g,_=ING.build_abox(self.records)
        row=self.records[0].values
        bp=ING._sid('boarding_pass',row['boarding_pass_id'])
        info=ING._sid('reservation_information','air',row['entry_id'])
        self.assertIn((bp,ING.BFO.BFO_0000101,info),g)
        self.assertIn((ING._sid('person',row['person_id']),ING.AGENT.possesses,bp),g)

    def test_09_rail_journey_objects(self):
        g,_=ING.build_abox(self.records)
        r=self.records[2].values
        trip=ING._sid('trip','rail',r['trip_id'])
        self.assertIn((trip,RDF.type,ING.INTEL.rail_journey),g)
        self.assertIn((trip,ING.INTEL.departs_from_station,ING._sid('railway_station',r['departure_id'])),g)
        self.assertIn((trip,ING.INTEL.arrives_at_station,ING._sid('railway_station',r['arrival_id'])),g)
        self.assertIn((trip,ING.INTEL.occurs_aboard_train,ING._sid('passenger_train',r['trip_id'])),g)

    def test_10_planned_trip_not_asserted_actual(self):
        g,_=ING.build_abox(self.records)
        r=self.records[-1].values
        trip=ING._sid('trip','rail',r['trip_id'])
        plan=ING._sid('rail_journey_plan',r['entry_id'])
        self.assertNotIn((trip,RDF.type,ING.INTEL.rail_journey),g)
        self.assertNotIn((plan,ING.INTEL.specifies_rail_journey,trip),g)

    def test_11_no_threat_roles_or_new_schema(self):
        g,_=ING.build_abox(self.records)
        self.assertFalse(list(g.subjects(RDF.type, OWL.Class)))
        self.assertFalse(list(g.subjects(RDF.type, OWL.ObjectProperty)))
        self.assertFalse(list(g.subjects(RDF.type, OWL.DatatypeProperty)))
        self.assertFalse(list(g.subjects(RDF.type, ING.AGENT.Person_Of_Interest)))

    def test_12_validate_only_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            a,b,c=(Path(d)/f for f in ('a.ttl','b.ttl','report.json'))
            result=ING.ingest(self.template,self.ontology,a,b,c,validate_only=True)
            self.assertEqual(result['status'],'PASS_MANIFEST_VALIDATION_ONLY')
            self.assertTrue(all(not p.exists() for p in (a,b,c)))

    def test_13_generation_preserves_augmented_and_is_repeatable(self):
        original=ING._sha_file(self.ontology)
        with tempfile.TemporaryDirectory() as d:
            a,b,c=(Path(d)/f for f in ('a.ttl','b.ttl','report.json'))
            result=ING.ingest(self.template,self.ontology,a,b,c)
            self.assertEqual(result['counts']['new_named_individuals'],25)
            self.assertEqual(result['abox_triples'],126)
            self.assertEqual(result['combined_triples'],610)
            self.assertEqual(ING._sha_file(self.ontology),original)
            first=ING._sha_file(b)
            ING.ingest(self.template,self.ontology,a,b,c)
            self.assertEqual(ING._sha_file(b),first)
            self.assertTrue(b.read_bytes().startswith(self.ontology.read_bytes()))


if __name__=='__main__':
    unittest.main()
