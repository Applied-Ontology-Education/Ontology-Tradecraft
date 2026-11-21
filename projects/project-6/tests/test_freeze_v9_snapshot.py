import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/freeze_v9_snapshot.py'
spec=importlib.util.spec_from_file_location('freeze',SCRIPT)
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

class FreezeTests(unittest.TestCase):
    def test_checks_full_run_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for rel in f.UPSTREAM:
                p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(b'example frozen input')
            for d in (root/'imports',root/'generated/experiments/rail_mowl_v9_reviewed',root/f.BASE):
                d.mkdir(parents=True,exist_ok=True)
            (root/'imports'/'example.ttl').write_text('example')
            (root/'generated/experiments/rail_mowl_v9_reviewed'/'fold_1_outer_train.ttl').write_text('graph')
            summary={'experiment_mode':'NESTED_DEV_SEVEN_APPROVED_ANCHORS','configs_per_fold':10,
                     'folds_completed':4,'projection_fallback_to_native_folds':4,
                     'native_8_class_mean':0.6037136092782021,'native_auc':.953125,
                     'native_gap':.4515440862160176,'native_top1_of_four':7,
                     'input_sha256':{'src/module.ttl':f.hashfile(root/'src/module.ttl')},
                     'review_file_sha256':f.hashfile(root/'reports/experiments/rail_anchor_review/approved_anchors.json')}
            p=root/f.BASE/'eight_class_summary.json'
            p.write_text(json.dumps(summary))
            ss,paths=f.collect(root)
            self.assertIn('scripts/experiment_rail_mowl_v9.py',paths)
            with patch('sys.argv',['freeze', '--project-root',str(root)]):
                self.assertEqual(f.main(),0)
            archive=root/f.FREEZE/'v9_frozen_snapshot.zip'
            with ZipFile(archive) as z:
                self.assertIn('FROZEN_MANIFEST.json',z.namelist())
                self.assertIn('scripts/experiment_rail_mowl_v9.py',z.namelist())
            with patch('sys.argv',['freeze', '--project-root',str(root)]):
                with self.assertRaises(FileExistsError):f.main()

if __name__=='__main__': unittest.main()
