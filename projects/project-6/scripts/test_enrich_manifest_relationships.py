"""Tests for airline/rail in-transit relationship enrichment."""
import tempfile
import unittest
from pathlib import Path

from rdflib import Graph, Literal
from rdflib.namespace import RDF, OWL, RDFS, XSD
import ingest_passenger_manifest as ing
import enrich_manifest_relationships as en

ROOT=Path(__file__).resolve().parents[1]

class InTransitEnrichmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xlsx=ROOT/'templates/passenger_manifest_template.xlsx'
        cls.owl=ROOT/'generated/module_augmented.ttl'
        cls.abox=ROOT/'generated/manifest_instances.ttl'
        cls.records=ing.parse_manifest(cls.xlsx)
        cls.base=Graph().parse(cls.owl,format='turtle')
        cls.instances=Graph().parse(cls.abox,format='turtle')

    def test_01_prior_abox_matches_workbook(self):
        en._expected_core_records(self.records,self.base,self.instances)

    def test_02_schema_new_class_and_new_properties(self):
        schema,delta,audit=en.add_relationships(self.records,self.base,self.instances)
        self.assertEqual(len(schema),54)
        self.assertEqual(len(en.NEW_DATA_PROPS),5)
        self.assertEqual(len(en.NEW_OBJECT_PROPS),5)
        self.assertIn((ing.INTEL.air_flight,RDF.type,OWL.Class),schema)
        self.assertIn((ing.INTEL.air_flight,RDFS.subClassOf,ing.BFO.BFO_0000015),schema)
        self.assertFalse(list(delta.subjects(RDF.type,OWL.Class)))
        self.assertEqual(audit['new_tbox_triples'],54)

    def test_03_actual_flight_and_planning_act_distinct(self):
        _,delta,a=en.add_relationships(self.records,self.base,self.instances)
        tid=self.records[0].values['trip_id']
        flight=ing._sid('trip','air',tid);act=ing._sid('act_of_flight_plan',tid)
        info=ing._sid('flight_plan',tid);plane=ing._sid('airframe','N123AB')
        self.assertNotEqual(flight,act)
        self.assertIn((flight,RDF.type,ing.INTEL.air_flight),delta)
        self.assertIn((act,RDF.type,ing.AGENT.Act_of_Flight_Plan),delta)
        self.assertIn((plane,ing.BFO.BFO_0000056,act),delta)
        self.assertIn((flight,ing.INTEL.occurs_aboard_airplane,plane),delta)
        self.assertIn((flight,ing.INTEL.has_flight_plan_information,info),delta)
        self.assertIn((info,en.IS_ABOUT,act),delta)
        self.assertEqual(a['actual_air_flights'],1)

    def test_04_flight_plan_ice_data(self):
        _,delta,_=en.add_relationships(self.records,self.base,self.instances)
        info=ing._sid('flight_plan',self.records[0].values['trip_id'])
        self.assertIn((info,ing.INTEL.has_flight_service_code,Literal('AA101',datatype=XSD.string)),delta)
        self.assertTrue(list(delta.objects(info,ing.INTEL.has_planned_air_departure_time)))

    def test_05_physical_document_remains_optional(self):
        _,delta,a=en.add_relationships(self.records,self.base,self.instances)
        self.assertEqual(a['flight_plan_document_bearers'],0)
        tid=self.records[0].values['trip_id']
        _,delta,a=en.add_relationships(self.records,self.base,self.instances,{tid:'FPDOC-AA101'})
        doc=ing._sid('physical_flight_plan_document','FPDOC-AA101')
        self.assertIn((doc,RDF.type,en.IAO_MATERIAL_BEARER),delta)
        self.assertIn((doc,en.IS_CARRIER_OF,ing._sid('flight_plan',tid)),delta)
        self.assertEqual(a['flight_plan_document_bearers'],1)

    def test_06_rail_trains_are_already_participating(self):
        _,delta,a=en.add_relationships(self.records,self.base,self.instances)
        self.assertEqual(a['rail_train_participation_links'],2)
        for r in self.records[2:]:
            tid=r.values['trip_id'];node=ing._sid('passenger_train',tid)
            self.assertIn((node,ing.BFO.BFO_0000056,ing._sid('trip','rail',tid)),self.instances)
        self.assertFalse(list(delta.triples((None,ing.BFO.BFO_0000056,ing._sid('trip','rail',tid)))))

    def test_07_nex103_new_york_is_arrival(self):
        _,delta,_=en.add_relationships(self.records,self.base,self.instances)
        d=self.records[-1].values
        train=ing._sid('passenger_train',d['trip_id'])
        self.assertIn((train,ing.INTEL.has_train_departure_station_name,
                       Literal('Philadelphia 30th Street',datatype=XSD.string)),delta)
        self.assertIn((train,ing.INTEL.has_train_arrival_station_name,
                       Literal('New York Penn Station',datatype=XSD.string)),delta)

    def test_08_rail_plans_link_to_actual_trips(self):
        _,delta,_=en.add_relationships(self.records,self.base,self.instances)
        for r in self.records[2:]:
            d=r.values
            plan=ing._sid('rail_journey_plan',d['entry_id'])
            self.assertIn((plan,ing.INTEL.plans_to_use_train,
                           ing._sid('passenger_train',d['trip_id'])),delta)
            self.assertIn((plan,ing.INTEL.specifies_rail_journey,
                           ing._sid('trip','rail',d['trip_id'])),self.instances)

    def test_09_unknown_physical_document_fails(self):
        with self.assertRaisesRegex(ing.ManifestError,'not found'):
            en.add_relationships(self.records,self.base,self.instances,{'UNKNOWN':'DOC'})

    def test_10_duplicate_document_trip_rejected(self):
        with self.assertRaisesRegex(ing.ManifestError,'unique nonempty'):
            en.parse_kv_arguments(['AA=DOC','AA=DOC'])

    def test_11_missing_actual_flight_fails(self):
        altered=self.instances-Graph()
        flight=ing._sid('trip','air',self.records[0].values['trip_id'])
        altered.remove((flight,RDF.type,ing.BFO.BFO_0000015))
        with self.assertRaisesRegex(ing.ManifestError,'does not exactly match'):
            en.add_relationships(self.records,self.base,altered)

    def test_12_no_write_on_validate_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/n for n in ['schema.ttl','abox.ttl','all.ttl','report.json']]
            r=en.enrich(self.xlsx,self.owl,self.abox,*paths,validate_only=True)
            self.assertEqual(r['status'],'PASS_RELATIONSHIPS_VALIDATED_ONLY')
            self.assertFalse(any(p.exists() for p in paths))

    def test_13_full_output_deterministic(self):
        before=tuple(map(ing._sha_file,(self.xlsx,self.owl,self.abox)))
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/n for n in ['schema.ttl','abox.ttl','all.ttl','report.json']]
            tid=self.records[0].values['trip_id']
            result=en.enrich(self.xlsx,self.owl,self.abox,*paths,flight_documents={tid:'FPDOC-AA101'})
            self.assertEqual(result['combined_triples'],709)
            self.assertEqual(result['new_abox_triples'],29)
            self.assertEqual(result['enriched_abox_triples'],171)
            self.assertTrue(paths[2].read_bytes().startswith(self.owl.read_bytes()))
            h=ing._sha_file(paths[2]);en.enrich(self.xlsx,self.owl,self.abox,*paths,flight_documents={tid:'FPDOC-AA101'})
            self.assertEqual(ing._sha_file(paths[2]),h)
            self.assertEqual(tuple(map(ing._sha_file,(self.xlsx,self.owl,self.abox))),before)

    def test_14_optional_physical_ticket_reference_is_preserved(self):
        _,delta,_=en.add_relationships(self.records,self.base,self.instances)
        row=self.records[2].values
        self.assertIn((ing._sid('rail_ticket',row['ticket_id']),ing.INTEL.has_physical_rail_ticket_evidence_reference,
                       Literal(row['physical_ticket_evidence_reference'],datatype=XSD.string)),delta)

if __name__=='__main__': unittest.main()
