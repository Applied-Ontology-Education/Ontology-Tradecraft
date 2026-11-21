"""Offline checks for the isolated Project 6 run_all v9 integration.

These tests do NOT execute Java, Grok, or mOWL. They exercise the driver wiring,
review-file preflight, and the v9 result-status acceptance rules.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
MODULE = HERE / 'scripts' / 'run_all.py'
spec = importlib.util.spec_from_file_location('run_all_v9_driver', MODULE)
run_all = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = run_all
spec.loader.exec_module(run_all)
REFERENCE = Path(__file__).resolve().parent / 'fixtures' / 'approved_anchors_REFERENCE.json'

class RunAllV9IntegrationTests(unittest.TestCase):
    def make_root(self, path):
        root = Path(path)
        (root / 'src').mkdir()
        (root / 'imports').mkdir()
        (root / 'scripts').mkdir()
        (root / 'config').mkdir()
        (root / 'reports/experiments/rail_mowl_v9_reviewed').mkdir(parents=True)
        (root / 'src/module.ttl').write_text('example', encoding='utf-8')
        for n in run_all.IMPORT_NAMES:
            (root / 'imports' / n).write_text(n, encoding='utf-8')
        for st in run_all.stage_plan():
            (root / 'scripts' / st.script).write_text('# test fixture\n', encoding='utf-8')
        for n in ('experiment_rail_mowl_v8.py', 'audit_rail_superclass_support.py'):
            (root / 'scripts' / n).write_text('# test fixture\n', encoding='utf-8')
        shutil.copyfile(REFERENCE, root / run_all.V9_ANCHORS_CONFIG)
        return root

    def test_added_only_after_merge(self):
        plan = run_all.stage_plan()
        self.assertEqual(plan[-1].key, '08_v9_rail_mowl')
        self.assertEqual(plan[-2].key, '07_controlled_merge')
        v9 = plan[-1]
        self.assertEqual(v9.script, 'experiment_rail_mowl_v9.py')
        self.assertIn(run_all.V9_ANCHORS_CONFIG, v9.args)
        self.assertEqual(v9.args[v9.args.index('--native-epochs')+1], '1000')
        self.assertEqual(v9.args[v9.args.index('--projection-epochs')+1], '100')

    def test_preflight_preserves_all_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.make_root(d)
            sentinel = root / 'reports/experiment_already_completed.json'
            sentinel.write_text('frozen', encoding='utf-8')
            result = run_all.run_pipeline(root, check_runtime=False, preflight_only=True)
            self.assertEqual(result['status'], 'PASS_PREFLIGHT_ONLY')
            self.assertEqual(result['steps'], len(run_all.stage_plan()))
            self.assertEqual(sentinel.read_text(), 'frozen')
            self.assertIn(run_all.V9_ANCHORS_CONFIG, result['inputs'])
            self.assertFalse((root/'.pipeline-archives').exists())

    def test_preflight_fails_closed_on_missing_review(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.make_root(d)
            (root/run_all.V9_ANCHORS_CONFIG).unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Missing reviewed v9 anchor'):
                run_all.run_pipeline(root, check_runtime=False, preflight_only=True)
            self.assertFalse((root/'.pipeline-archives').exists())

    def test_v9_below_threshold_is_valid_completed_experiment(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.make_root(d)
            inputs=run_all.fingerprint_inputs(root,run_all.stage_plan())
            stage=run_all.stage_plan()[-1]
            for item in stage.outputs:
                target=root/item;target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('test',encoding='utf-8')
            record={
                'version':run_all.V9_EXPECTED_VERSION,
                'status':'TARGET_OR_CONTROLS_NOT_MET',
                'folds_completed':4,'outer_test_sample_size':8,
                'dev_pairs_per_fold':[40]*4,'configs_per_fold':10,
                'all_8_rail_targets_unseen_during_model_selection':True,
                'all_inner_development_targets_unseen_by_inner_mowl_and_projection':True,
                'outer_named_subclass_leakage_detected':False,
                'development_named_subclass_leakage_detected':False,
                'review_file_sha256':inputs[run_all.V9_ANCHORS_CONFIG],
                'input_sha256':{'src/module.ttl':inputs['src/module.ttl']},
                'native_8_class_mean':0.6037,'native_auc':0.953125,
                'native_gap':0.45154, 'v9_8_class_mean':0.6037,
                'v9_auc':0.953125,'v9_gap':0.45154,
                'native_top1_of_four':7,'v9_top1_of_four':7,
                'target_0_70_met':False,
            }
            (root/stage.report).write_text(json.dumps(record),encoding='utf-8')
            self.assertEqual(run_all.read_and_validate_stage(root,stage,inputs)['status'],record['status'])
            record['outer_named_subclass_leakage_detected']=True
            (root/stage.report).write_text(json.dumps(record),encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'v9 completeness'):
                run_all.read_and_validate_stage(root,stage,inputs)

if __name__=='__main__':
    unittest.main()
