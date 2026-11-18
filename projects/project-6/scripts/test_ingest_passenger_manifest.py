"""Tests for the trusted onboard/in-transit manifest contract (synthetic data)."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rdflib import Graph
from rdflib.namespace import RDF, OWL
import ingest_passenger_manifest as ing

ROOT = Path(__file__).resolve().parents[1]

class InTransitIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xlsx = ROOT/'templates/passenger_manifest_template.xlsx'
        cls.owl = ROOT/'generated/module_augmented.ttl'
        cls.records = ing.parse_manifest(cls.xlsx)
        cls.base = Graph().parse(cls.owl,format='turtle')

    def changed(self, index, **kw):
        return replace(self.records[index],values=dict(self.records[index].values,**kw))

    def test_01_five_onboard_rows_two_modes(self):
        self.assertEqual(len(self.records),5)
        self.assertEqual([sum(r.mode==m for r in self.records) for m in ('air','rail')],[2,3])

    def test_02_evidence_and_booking_status_removed(self):
        for r in self.records:
            self.assertNotIn('journey_status',r.values)
            self.assertNotIn('booking_status',r.values)
            self.assertNotIn('occurrence_evidence',r.values)
            self.assertNotIn('evidence_reference',r.values)

    def test_03_existing_ontology_vocabulary(self):
        ing.verify_ontology(self.base,self.records)

    def test_04_duplicate_manifest_entry_fails(self):
        with self.assertRaisesRegex(ing.ManifestError,'duplicate entry_id'):
            ing.verify_records(self.records+[self.records[0]])

    def test_05_conflicting_airframe_fails(self):
        with self.assertRaisesRegex(ing.ManifestError,'conflicting aircraft'):
            ing.verify_records([self.records[0],self.changed(1,aircraft_tail_number='OTHER')])

    def test_06_conflicting_names_fails(self):
        with self.assertRaisesRegex(ing.ManifestError,'conflicting names'):
            ing.verify_records([self.records[0],self.changed(2,person_id='P-001',last_name='NotExample')])

    def test_07_conflicting_route_fails(self):
        with self.assertRaisesRegex(ing.ManifestError,'conflicting route'):
            ing.verify_records([self.records[0],self.changed(1,departure_name='Wrong')])

    def test_08_offset_required(self):
        with self.assertRaisesRegex(ing.ManifestError,'time zone offset'):
            ing._iso_datetime('2026-11-17T14:00:00','x')

    def test_09_actual_air_process_and_airplane_participation(self):
        g, _ = ing.build_abox(self.records)
        row=self.records[0].values
        proc=ing._sid('trip','air',row['trip_id'])
        plane=ing._sid('airframe',row['aircraft_tail_number'])
        self.assertIn((proc,RDF.type,ing.BFO.BFO_0000015),g)
        self.assertIn((plane,ing.BFO.BFO_0000056,proc),g)

    def test_10_two_rail_journeys_typed_and_carried(self):
        g, result=ing.build_abox(self.records)
        self.assertEqual(result['counts']['unique_rail_journeys'],2)
        for r in self.records[2:]:
            d=r.values
            proc=ing._sid('trip','rail',d['trip_id'])
            train=ing._sid('passenger_train',d['trip_id'])
            self.assertIn((proc,RDF.type,ing.INTEL.rail_journey),g)
            self.assertIn((train,ing.BFO.BFO_0000056,proc),g)
            self.assertIn((proc,ing.INTEL.occurs_aboard_train,train),g)

    def test_11_all_passengers_participate_in_correct_process(self):
        g, report=ing.build_abox(self.records)
        for r in self.records:
            d=r.values
            self.assertIn((ing._sid('person',d['person_id']),ing.BFO.BFO_0000056,
                           ing._sid('trip',r.mode,d['trip_id'])),g)
        self.assertEqual(report['counts']['onboard_passenger_rows'],5)
        self.assertEqual(report['counts']['people'],5)

    def test_12_rail_ticket_material_bearer_still_requires_optional_reference(self):
        g, report=ing.build_abox(self.records)
        self.assertEqual(report['counts']['physical_rail_ticket_rows'],2)
        self.assertNotIn((ing._sid('rail_ticket','TKT-R003'),RDF.type,ing.INTEL.rail_ticket),g)

    def test_13_no_new_schema_or_threat_roles(self):
        g, _=ing.build_abox(self.records)
        self.assertFalse(list(g.subjects(RDF.type, OWL.Class)))
        self.assertFalse(list(g.subjects(RDF.type, OWL.ObjectProperty)))
        self.assertFalse(list(g.subjects(RDF.type, ing.AGENT.Person_Of_Interest)))

    def test_14_boarding_pass_information_bearer_intact(self):
        g,_=ing.build_abox(self.records)
        d=self.records[0].values
        self.assertIn((ing._sid('boarding_pass',d['boarding_pass_id']),ing.BFO.BFO_0000101,
                       ing._sid('reservation_information','air',d['entry_id'])),g)

    def test_15_validate_only_is_nonmutating(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/n for n in ['a.ttl','b.ttl','r.json']]
            result=ing.ingest(self.xlsx,self.owl,*paths,validate_only=True)
            self.assertEqual(result['status'],'PASS_MANIFEST_VALIDATION_ONLY')
            self.assertFalse(any(x.exists() for x in paths))

    def test_16_repeatable_and_protected_bytes(self):
        old=ing._sha_file(self.owl)
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/n for n in ['a.ttl','b.ttl','r.json']]
            report=ing.ingest(self.xlsx,self.owl,*paths)
            self.assertEqual(report['ontology_triples'],484)
            self.assertEqual(report['abox_triples'],142)
            self.assertEqual(report['combined_triples'],626)
            self.assertTrue(paths[1].read_bytes().startswith(self.owl.read_bytes()))
            h=ing._sha_file(paths[1]);ing.ingest(self.xlsx,self.owl,*paths)
            self.assertEqual(h,ing._sha_file(paths[1]))
            self.assertEqual(old,ing._sha_file(self.owl))

if __name__=='__main__': unittest.main()
