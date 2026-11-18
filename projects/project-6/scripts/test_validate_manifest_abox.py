"""Structural validator regression tests for in-transit manifest ABox."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

from rdflib import Graph, Literal, BNode
from rdflib.namespace import RDF, OWL, RDFS, XSD

from ingest_passenger_manifest import _sid, INTEL, AGENT, BFO
from enrich_manifest_relationships import _schema_graph, NEW_DATA_PROPS, NEW_OBJECT_PROPS
from validate_manifest_abox import (load_inputs, validate_assertions, ValidationError,
    build_shapes, SH, shacl_validate, _stable, _flat_parts, _serial_flat,
    atomic_json, file_hash)

ROOT=Path(__file__).resolve().parents[1]

class InTransitValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workbook=ROOT/'templates/passenger_manifest_template.xlsx'
        cls.context=load_inputs(ROOT,cls.workbook)
        cls.records=cls.context['records']

    def altered(self, triple=None):
        ctx=dict(self.context)
        ctx['graphs']=dict(ctx['graphs'])
        abox=Graph(); abox+=ctx['graphs']['enriched_abox']
        if triple: abox.remove(triple)
        ctx['graphs']['enriched_abox']=abox
        combined=Graph();combined+=self.context['graphs']['combined']
        if triple:combined.remove(triple)
        ctx['graphs']['combined']=combined
        return ctx

    def test_01_exact_reviewed_source_and_augmented_sizes(self):
        g=self.context['graphs']
        self.assertEqual(len(g['source']),421)
        self.assertEqual(len(g['augmented']),484)

    def test_02_all_onboard_assertions_pass_without_evidence(self):
        v=validate_assertions(self.context)
        self.assertEqual(v['warning_count'],0)
        self.assertEqual(v['counts']['manifest_passenger_participations'],5)
        self.assertEqual(v['counts']['actual_air_flights'],1)
        self.assertEqual(v['counts']['actual_rail_journeys'],2)
        self.assertEqual(v['status'],'PASS_ONBOARD_IN_TRANSIT_ASSERTIONS')

    def test_03_strict_occurrence_flag_is_compatible(self):
        v=validate_assertions(self.context,strict_occurrence=True)
        self.assertEqual(v['warnings'],[])

    def test_04_missing_air_passenger_participation_fails(self):
        r=self.records[0];d=r.values
        t=(_sid('person',d['person_id']),BFO.BFO_0000056,_sid('trip','air',d['trip_id']))
        with self.assertRaisesRegex(ValidationError,'Missing expected assertion'):
            validate_assertions(self.altered(t))

    def test_05_missing_rail_passenger_participation_fails(self):
        r=self.records[-1];d=r.values
        t=(_sid('person',d['person_id']),BFO.BFO_0000056,_sid('trip','rail',d['trip_id']))
        with self.assertRaisesRegex(ValidationError,'Missing expected assertion'):
            validate_assertions(self.altered(t))

    def test_06_missing_rail_process_fails(self):
        d=self.records[-1].values
        t=(_sid('trip','rail',d['trip_id']),RDF.type,INTEL.rail_journey)
        with self.assertRaisesRegex(ValidationError,'Missing expected assertion'):
            validate_assertions(self.altered(t))

    def test_07_missing_actual_flight_class_fails(self):
        d=self.records[0].values
        t=(_sid('trip','air',d['trip_id']),RDF.type,INTEL.air_flight)
        with self.assertRaisesRegex(ValidationError,'Missing expected assertion'):
            validate_assertions(self.altered(t))

    def test_08_rail_station_direction_mismatch_fails(self):
        d=self.records[-1].values
        t=(_sid('trip','rail',d['trip_id']),INTEL.arrives_at_station,
           _sid('railway_station',d['arrival_id']))
        with self.assertRaisesRegex(ValidationError,'Missing expected assertion'):
            validate_assertions(self.altered(t))

    def test_09_missing_aircraft_actual_participation_fails(self):
        d=self.records[0].values
        t=(_sid('airframe',d['aircraft_tail_number']),BFO.BFO_0000056,
           _sid('trip','air',d['trip_id']))
        with self.assertRaisesRegex(ValidationError,'Missing expected assertion'):
            validate_assertions(self.altered(t))

    def test_10_schema_has_air_flight_class_and_properties(self):
        g=_schema_graph(Graph())
        self.assertEqual(len(g),54)
        self.assertEqual(len(NEW_DATA_PROPS),5)
        self.assertEqual(len(NEW_OBJECT_PROPS),5)
        self.assertIn((INTEL.air_flight,RDFS.subClassOf,BFO.BFO_0000015),g)

    def test_11_schema_cannot_shadow_augmented(self):
        base=Graph()
        base.add((INTEL.has_train_arrival_station_name,RDF.type,OWL.ObjectProperty))
        with self.assertRaises(Exception):_schema_graph(base)

    def test_12_shacl_includes_actual_flight(self):
        shapes=build_shapes(self.context['graphs']['enriched_abox'])
        self.assertIn((INTEL.occurs_aboard_airplane,),{(x,) for x in shapes.objects(None,SH.path)})
        self.assertIn((INTEL.occurs_aboard_train,),{(x,) for x in shapes.objects(None,SH.path)})

    def test_13_shacl_invoked_without_inference(self):
        captured={}
        def fake_shacl(**kwargs):
            captured.update(kwargs);return True,Graph(),'Conforms: True'
        with mock.patch.dict('sys.modules',{'pyshacl':SimpleNamespace(validate=fake_shacl)}):
            result=shacl_validate(Graph(),Graph())
        self.assertTrue(result['conforms'])
        self.assertEqual(captured['inference'],'none')

    def test_14_bnodes_are_separated_for_import_reasoning(self):
        b=BNode('same'); a=Graph();c=Graph()
        a.add((b,RDFS.label,Literal('one')));c.add((b,RDFS.label,Literal('two')))
        flat=_flat_parts([('one',a),('two',c)])
        self.assertEqual(len(set(flat.subjects(RDFS.label))),2)
        self.assertEqual(len(Graph().parse(data=_serial_flat(flat),format='turtle')),len(flat))

    def test_15_robot_log_print_works_with_windows_cp1252_stdout(self):
        """The non-ASCII arrow previously caused UnicodeEncodeError before ROBOT launched."""
        import io
        from validate_manifest_abox import run_robot

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            log = work / 'extract.log'
            output = work / 'module.ttl'

            class FakeProcess:
                def wait(self, timeout=None):
                    output.write_text('robot output', encoding='utf-8')
                    return 0

            encoded_bytes = io.BytesIO()
            windows_stdout = io.TextIOWrapper(
                encoded_bytes, encoding='cp1252', errors='strict', write_through=True
            )
            with mock.patch('validate_manifest_abox.subprocess.Popen', return_value=FakeProcess()):
                with mock.patch('sys.stdout', windows_stdout):
                    result = run_robot(
                        ['robot', 'extract', '--method', 'STAR'],
                        log, timeout=30, output=output
                    )
            self.assertEqual(result['return_code'], 0)
            self.assertTrue(log.is_file())
            self.assertIn(b'->', encoded_bytes.getvalue())

    def test_16_atomic_report_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'reports/x.json'
            atomic_json(p,{'status':'PASS'})
            self.assertIn('PASS',p.read_text())
            self.assertEqual(len(file_hash(p)),64)

if __name__=='__main__':unittest.main()
