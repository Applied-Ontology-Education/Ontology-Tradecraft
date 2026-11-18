"""Offline regression tests for Project 6 full TBox+manifest one-click orchestration.

The fake executor never runs Grok, mOWL or HermiT. One integration test runs
actual v3 ingestion and enrichment scripts against the supplied synthetic XLSX.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import run_all as runner


FIXTURES = Path(__file__).resolve().parents[1] / 'test_fixtures' / 'manifest_v3'


class FullPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'src').mkdir()
        (self.root / 'src/module.ttl').write_text('# unchanged source\n', encoding='utf8')
        (self.root / 'imports').mkdir()
        for name in runner.IMPORT_NAMES:
            (self.root / 'imports' / name).write_text('# import '+name+'\n', encoding='utf8')
        (self.root / 'scripts').mkdir()
        for script in runner.required_script_paths(self.root, runner.stage_plan()):
            script.write_text('# stage stub\n', encoding='utf8')
        (self.root / 'templates').mkdir()
        (self.root / 'templates/passenger_manifest_template.xlsx').write_bytes(b'workbook-for-fake-run')
        self.source_hash = runner.sha256_file(self.root / 'src/module.ttl')
        self.workbook = self.root / 'templates/passenger_manifest_template.xlsx'

    def fake_stage(self, root, stage):
        for relative in stage.outputs:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fake generated output\n', encoding='utf8')
        report = {'status': stage.status, 'source_unchanged': True,
                  'source_sha256': runner.sha256_file(root / 'src/module.ttl')}
        if stage.key == '05b_mowl_and_grok':
            report.update(llm_executed=True, mowl_training_completed=True)
        if stage.key == '05c_grounded_grok':
            report.update(llm_executed=True, mowl_training_reexecuted=False)
        if stage.key == '06_scoped_hermit':
            report.update(reasoning_scope='SOURCE_PLUS_STAR_LOCALITY_MODULE_OF_IMPORTS',
                          baseline_reasoner_passed=True, augmented_reasoner_passed=True)
        if stage.key == '07_controlled_merge':
            out = root / 'generated/module_augmented.ttl'
            out.write_bytes((root / 'src/module.ttl').read_bytes() + b'# rail extension\n')
            report.update(output='generated/module_augmented.ttl',
                          merged_sha256=runner.sha256_file(out), source_overwritten=False,
                          merged_triples=484, added_rail_triples=63, source_triples=421,
                          new_named_individuals=0)
        if stage.key == '08_manifest_ingestion':
            report.pop('source_sha256')
            report.update(version=runner.V3_SCRIPTS[stage.script],
                          ontology_sha256=runner.sha256_file(root / 'generated/module_augmented.ttl'),
                          source_and_augmented_ontology_unchanged=True,
                          no_threat_or_person_of_interest_classification=True,
                          generated_classes_or_properties=0,
                          travel_occurrence_policy='MANIFEST_ONBOARD_IN_TRANSIT',
                          counts={'rows':5,'people':5,'onboard_passenger_rows':5},
                          input_sha256=runner.sha256_file(root / 'templates/passenger_manifest_template.xlsx'),
                          abox_triples=142)
        if stage.key == '09_manifest_enrichment':
            report.update(version=runner.V3_SCRIPTS[stage.script],
                          source_sha256=runner.sha256_file(root / 'generated/module_augmented.ttl'),
                          protected_ontology_and_manifest_files_unchanged=True,
                          flight_and_rail_occurrence_assumed_from_manifest=True,
                          original_abox_sha256=runner.sha256_file(root / 'generated/manifest_instances.ttl'),
                          workbook_sha256=runner.sha256_file(root / 'templates/passenger_manifest_template.xlsx'),
                          augmented_triples=484, new_tbox_triples=54,
                          enriched_abox_triples=171, combined_triples=709)
        if stage.key == '09a_manifest_cosine':
            report.update(scope='NEW_EXCEL_ABOX_ASSERTIONS_ONLY',
                          original_classes_scored=0,
                          original_subclass_axioms_scored=0,
                          rail_tbox_subclass_axioms_scored=0,
                          baseline_sha256=runner.sha256_file(root / 'generated/module_augmented.ttl'),
                          final_sha256=runner.sha256_file(root / runner.FINAL_OUTPUT),
                          selected_assertions={'individual_class_assertion':25,
                                               'object_property_assertion':45,
                                               'data_property_assertion':50},
                          results={'mean_cosine':0.8,'threshold':0.70,
                                   'metric_threshold_met':True,
                                   'total_scored_assertions':120})
        if stage.key == '10_manifest_validation':
            report.pop('source_sha256')
            report.update(version=runner.V3_SCRIPTS[stage.script],
                          source_or_augmented_overwritten=False,
                          manifest_assertions={'status':'PASS_ONBOARD_IN_TRANSIT_ASSERTIONS','warnings':[]},
                          shacl={'conforms':True,'violations':0},
                          reasoning={'owl_reasoner_executed':True,
                                     'scope':'SOURCE_PLUS_STAR_LOCALITY_MODULE_OF_SIX_LOCAL_IMPORTS',
                                     **{key:{'return_code':0,'timeout':False} for key in
                                        ('extract','baseline_reasoner','candidate_reasoner')}},
                          protected_hashes={'workbook':runner.sha256_file(root / 'templates/passenger_manifest_template.xlsx'),
                                            **{name:runner.sha256_file(root / relative)
                                            for name,relative in {
                            'source':'src/module.ttl',
                            'augmented':'generated/module_augmented.ttl',
                            'initial_abox':'generated/manifest_instances.ttl',
                            'schema':'generated/manifest_tbox_extension.ttl',
                            'enriched_abox':'generated/manifest_instances_enriched.ttl',
                            'combined':runner.FINAL_OUTPUT}.items()}},
                          counts={'combined_triples':709})
            source_prefix = (root / 'generated/module_augmented.ttl').read_bytes()
            if not (root / runner.FINAL_OUTPUT).read_bytes().startswith(source_prefix):
                (root / runner.FINAL_OUTPUT).write_bytes(source_prefix + b'# manifest extension\n')
            report['protected_hashes']['combined'] = runner.sha256_file(root / runner.FINAL_OUTPUT)
        path = root / stage.report
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding='utf8')

    def run_fake(self, **kwargs):
        return runner.run_pipeline(self.root, check_runtime=False,
                                   stage_executor=self.fake_stage, **kwargs)

    def test_plan_has_fourteen_stages_with_correct_order(self):
        plan = runner.stage_plan()
        self.assertEqual(len(plan), 14)
        self.assertEqual([s.key for s in plan[-5:]], ['07_controlled_merge', '08_manifest_ingestion',
                                                     '09_manifest_enrichment','09a_manifest_cosine', '10_manifest_validation'])
        self.assertFalse(plan[-4].accepts_project_root)
        self.assertFalse(plan[-3].accepts_project_root)
        self.assertTrue(plan[-2].accepts_project_root)
        self.assertTrue(plan[-1].accepts_project_root)

    def test_no_cached_mowl_vectors_and_hard_timeouts(self):
        plan = runner.stage_plan()
        stage = next(s for s in plan if s.key == '05b_mowl_and_grok')
        self.assertIn('--with-llm', stage.args)
        self.assertNotIn('--reuse-vectors', stage.args)
        self.assertEqual(stage.args[stage.args.index('--epochs')+1], '200')
        self.assertEqual(plan[-1].timeout, 1380)

    def test_manifest_document_flag_is_forwarded_only_to_enrichment(self):
        stages = runner.stage_plan(flight_documents=('FL-20261117-AA101=FPDOC-AA101',))
        self.assertNotIn('--flight-document', next(s for s in stages if s.key=='08_manifest_ingestion').args)
        self.assertIn('--flight-document', next(s for s in stages if s.key=='09_manifest_enrichment').args)
        self.assertNotIn('--flight-document', next(s for s in stages if s.key=='09a_manifest_cosine').args)
        self.assertNotIn('--flight-document', stages[-1].args)

    def test_preflight_does_not_archive_and_hashes_eight_inputs(self):
        (self.root / 'reports').mkdir()
        (self.root / 'reports/old.json').write_text('older')
        result = self.run_fake(preflight_only=True)
        self.assertEqual(result['status'], 'PASS_PREFLIGHT_ONLY')
        self.assertEqual(len(result['inputs']), 8)
        self.assertEqual(result['steps'], 14)
        self.assertTrue((self.root / 'reports/old.json').exists())
        self.assertFalse((self.root / '.pipeline-archives').exists())

    def test_clean_mock_success_preserves_source_and_manifest(self):
        old = self.workbook.read_bytes()
        result = self.run_fake()
        self.assertEqual(result['status'], 'PASS_FULL_TBOX_ABOX_VALIDATED_COSINE_MET')
        self.assertEqual(len(result['stages']), 14)
        self.assertEqual(self.source_hash, runner.sha256_file(self.root / 'src/module.ttl'))
        self.assertEqual(self.workbook.read_bytes(), old)
        self.assertEqual(result['output'], runner.FINAL_OUTPUT)
        self.assertEqual(result['final_combined_triples'], 709)
        self.assertTrue((self.root / result['output']).is_file())

    def test_subthreshold_cosine_is_reported_honestly_and_reasoner_runs(self):
        def report_low(root, stage):
            self.fake_stage(root, stage)
            if stage.key == '09a_manifest_cosine':
                p = root / stage.report
                data = json.loads(p.read_text())
                data['results']['mean_cosine'] = 0.42
                data['results']['metric_threshold_met'] = False
                p.write_text(json.dumps(data))
        result = runner.run_pipeline(self.root, check_runtime=False, stage_executor=report_low)
        self.assertEqual(result['status'], 'PASS_FULL_TBOX_ABOX_VALIDATED_COSINE_BELOW_TARGET')
        self.assertFalse(result['cosine_threshold_met'])
        self.assertEqual(result['cosine_scored_assertions'], 120)
        self.assertEqual(result['stages'][-1]['stage'], '10_manifest_validation')

    def test_cosine_validation_rejects_broad_imported_holdout(self):
        def wrong_scope(root, stage):
            self.fake_stage(root, stage)
            if stage.key == '09a_manifest_cosine':
                p=root/stage.report
                d=json.loads(p.read_text())
                d['original_subclass_axioms_scored']=165
                p.write_text(json.dumps(d))
        with self.assertRaisesRegex(RuntimeError,'only new ABox'):
            runner.run_pipeline(self.root, check_runtime=False, stage_executor=wrong_scope)

    def test_prior_generated_reports_are_archived(self):
        self.run_fake()
        (self.root / 'generated/injected.txt').write_text('stale')
        result = self.run_fake()
        dest = self.root / result['previous_outputs_archive']
        self.assertTrue((dest / 'generated/injected.txt').exists())
        self.assertFalse((self.root / 'generated/injected.txt').exists())

    def test_missing_workbook_fails_without_archiving(self):
        self.workbook.unlink()
        (self.root / 'generated').mkdir()
        (self.root / 'generated/old.ttl').write_text('keep me')
        with self.assertRaisesRegex(FileNotFoundError, 'manifest'):
            self.run_fake()
        self.assertTrue((self.root / 'generated/old.ttl').is_file())

    def test_workbook_inside_generated_refused(self):
        (self.root / 'generated').mkdir()
        (self.root / 'generated/input.xlsx').write_bytes(b'bad place')
        with self.assertRaisesRegex(ValueError, 'must not be inside'):
            self.run_fake(manifest='generated/input.xlsx')

    def test_missing_stage_script_prevents_run(self):
        (self.root / 'scripts/enrich_manifest_relationships.py').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'enrich_manifest'):
            self.run_fake()

    def test_invalid_flight_document_flag_rejected(self):
        with self.assertRaisesRegex(ValueError, 'TRIP_ID=DOCUMENT_ID'):
            runner.stage_plan(flight_documents=('BAD',))

    def test_fail_fast_when_ingestion_fails(self):
        stages = []
        def broken(root, stage):
            stages.append(stage.key)
            if stage.key == '08_manifest_ingestion':
                raise RuntimeError('bad input sheet')
            self.fake_stage(root, stage)
        with self.assertRaisesRegex(RuntimeError, 'bad input sheet'):
            runner.run_pipeline(self.root, check_runtime=False, stage_executor=broken)
        self.assertEqual(stages[-1], '08_manifest_ingestion')
        self.assertNotIn('09_manifest_enrichment', stages)
        self.assertEqual(json.loads((self.root / 'reports/step8_run_summary.json').read_text())['status'], 'FAIL_FAST')

    def test_hash_protection_catches_workbook_mutation(self):
        def mutate(root, stage):
            self.fake_stage(root, stage)
            if stage.key == '08_manifest_ingestion':
                self.workbook.write_bytes(b'changed manifest')
        with self.assertRaisesRegex(RuntimeError, 'Protected original'):
            runner.run_pipeline(self.root, check_runtime=False, stage_executor=mutate)
        self.assertFalse(json.loads((self.root / 'reports/step8_run_summary.json').read_text())['source_unchanged'])

    def test_reject_validation_with_shacl_violations(self):
        def fail_shacl(root, stage):
            self.fake_stage(root, stage)
            if stage.key == '10_manifest_validation':
                path = root / stage.report
                data = json.loads(path.read_text())
                data['shacl']['conforms'] = False
                data['shacl']['violations'] = 1
                path.write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, 'SHACL'):
            runner.run_pipeline(self.root, check_runtime=False, stage_executor=fail_shacl)

    def test_reject_validation_with_failed_hermit(self):
        def fail_hermit(root, stage):
            self.fake_stage(root, stage)
            if stage.key == '10_manifest_validation':
                path = root / stage.report
                data = json.loads(path.read_text())
                data['reasoning']['candidate_reasoner']['return_code'] = 1
                path.write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, 'candidate_reasoner'):
            runner.run_pipeline(self.root, check_runtime=False, stage_executor=fail_hermit)

    def test_actual_v3_ingestion_and_enrichment_integrate_with_runner(self):
        # Real input parsing, RDF generation, report hashes, and CLI wiring.
        # Only Grok/mOWL/ROBOT/HermiT steps are simulated.
        package = FIXTURES
        shutil.copy2(package / 'module.ttl', self.root / 'src/module.ttl')
        shutil.copy2(package / 'passenger_manifest_template.xlsx', self.workbook)
        for name in ('ingest_passenger_manifest.py', 'enrich_manifest_relationships.py'):
            shutil.copy2(Path(__file__).resolve().parent / name, self.root / 'scripts' / name)
        def stage_executor(root, stage):
            if stage.key in {'08_manifest_ingestion', '09_manifest_enrichment'}:
                runner.execute_stage(root, stage)
            elif stage.key == '07_controlled_merge':
                self.fake_stage(root, stage)
                output = root / 'generated/module_augmented.ttl'
                shutil.copy2(package / 'module_augmented.ttl', output)
                report_path = root / stage.report
                data = json.loads(report_path.read_text())
                data['merged_sha256'] = runner.sha256_file(output)
                report_path.write_text(json.dumps(data))
            else:
                self.fake_stage(root, stage)
        result = runner.run_pipeline(self.root, check_runtime=False,
            stage_executor=stage_executor,
            flight_documents=('FL-20261117-AA101=FPDOC-AA101',))
        self.assertEqual(result['status'], 'PASS_FULL_TBOX_ABOX_VALIDATED_COSINE_MET')
        self.assertEqual(result['manifest_rows'], 5)
        self.assertEqual(result['manifest_people'], 5)
        self.assertEqual(result['tbox_triples'], 484)
        self.assertEqual(result['final_combined_triples'], 709)
        self.assertEqual(result['enriched_abox_triples'], 171)
        self.assertEqual(result['manifest_extension_triples'], 54)
        self.assertEqual(runner.sha256_file(self.workbook), result['manifest_sha256'])
        self.assertEqual(runner.sha256_file(self.root/'src/module.ttl'), result['source_sha256'])
        from rdflib import Graph
        combined = Graph().parse(self.root / runner.FINAL_OUTPUT, format='turtle')
        self.assertEqual(len(combined), 709)

    def test_subprocess_command_honors_individual_stage_cli(self):
        stub = self.root / 'scripts/stub.py'
        stub.write_text('import sys; print("ARGS:"+str(sys.argv), flush=True)\n', encoding='utf8')
        (self.root / 'reports').mkdir()
        stage = runner.Stage('00_test','stub.py',('--input','some.xlsx'),
                             'reports/none','PASS',(),30,accepts_project_root=False)
        runner.execute_stage(self.root,stage)
        log = (self.root / 'reports/pipeline_logs/00_test.log').read_text()
        self.assertIn('ARGS:', log)
        self.assertNotIn('--project-root', log)


if __name__ == '__main__':
    unittest.main()
