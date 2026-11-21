#!/usr/bin/env python3
"""Freeze Project 6 full v9 as a provenance-checked, no-overwrite ZIP snapshot.

From projects/project-6:
    python -u scripts/freeze_v9_snapshot.py

This copies existing, completed v9 experiment artifacts. It never changes
or deletes the live v9 experiment. It refuses to overwrite a frozen archive.
The archive may contain private ontology and project data: keep it private.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

BASE='reports/experiments/rail_mowl_v9_reviewed'
FREEZE='reports/experiments/v9_frozen_2026_11_19'
UPSTREAM=(
    'src/module.ttl', 'generated/module_augmented.ttl',
    'generated/rail_axioms_refined.ttl', 'generated/mowl_rail_train.ttl',
    'generated/mowl_rail_valid.ttl','reports/step5_split_report.json',
    'scripts/run_all.py','scripts/experiment_rail_mowl_v8.py',
    'scripts/experiment_rail_mowl_v9.py','scripts/audit_rail_superclass_support.py',
    'reports/experiments/rail_anchor_review/approved_anchors.json'
)

def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def collect(root):
    summary_path=root/BASE/'eight_class_summary.json'
    if not summary_path.is_file():raise FileNotFoundError(summary_path)
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('experiment_mode')!='NESTED_DEV_SEVEN_APPROVED_ANCHORS' or summary.get('configs_per_fold')!=10 or summary.get('folds_completed')!=4 or summary.get('projection_fallback_to_native_folds')!=4:
        raise ValueError('Only freeze the completed full v9 run, not a 3-epoch smoke test')
    if abs(summary.get('native_8_class_mean',-1)-0.6037136092782021)>1e-7 or abs(summary.get('native_auc',-1)-.953125)>1e-7 or abs(summary.get('native_gap',-1)-.4515440862160176)>1e-7:
        raise ValueError('Unexpected v9 baseline metrics; refusing to label as approved frozen run')
    files=set(UPSTREAM)
    for folder in (root/BASE,root/'generated/experiments/rail_mowl_v9_reviewed',root/'imports'):
        if not folder.is_dir():raise FileNotFoundError(folder)
        files.update(p.relative_to(root).as_posix() for p in folder.rglob('*') if p.is_file())
    for rel in files:
        if not (root/rel).is_file():raise FileNotFoundError(root/rel)
    for rel,sha in summary.get('input_sha256',{}).items():
        if hashfile(root/rel)!=sha:raise RuntimeError('Source file changed since v9 full run: '+rel)
    review='reports/experiments/rail_anchor_review/approved_anchors.json'
    if hashfile(root/review)!=summary['review_file_sha256']:
        raise RuntimeError('Approved anchors changed since v9 run')
    return summary,sorted(files)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    root=args.project_root.resolve()
    summary,files=collect(root)
    dest=root/FREEZE
    archive=dest/'v9_frozen_snapshot.zip'
    manifest_file=dest/'manifest.json'
    if archive.exists() or manifest_file.exists():
        raise FileExistsError('v9 frozen snapshot already exists; refusing to overwrite '+str(dest))
    dest.mkdir(parents=True,exist_ok=True)
    manifest={
        'status':'FROZEN_SNAPSHOT_CREATED','snapshot_name':'Project 6 v9 full run 2026-11-19',
        'mode':summary['experiment_mode'],'native_mean':summary['native_8_class_mean'],
        'native_auc':summary['native_auc'],'native_gap':summary['native_gap'],
        'native_top1_of_four':summary['native_top1_of_four'],
        'native_fallback_folds':summary['projection_fallback_to_native_folds'],
        'frozen_results_not_a_full_owl_reasoner_proof':True,
        'files':{rel:hashfile(root/rel) for rel in files},
    }
    manifest_blob=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode('utf-8')
    try:
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for rel in files:z.write(root/rel,arcname=rel)
            z.writestr('FROZEN_MANIFEST.json',manifest_blob)
        with zipfile.ZipFile(archive,'r') as z:
            if z.testzip() is not None:raise RuntimeError('ZIP CRC verification failed')
        manifest_file.write_bytes(manifest_blob)
    except BaseException:
        if archive.exists():archive.unlink()
        if manifest_file.exists():manifest_file.unlink()
        raise
    print('[v9 freeze] Frozen file count:',len(files))
    print('[v9 freeze] Immutable checkpoint ZIP:',archive)
    print('[v9 freeze] SHA256:',hashfile(archive))
    print('[v9 freeze] No original files changed. Do not rerun v9 if preserving live results.',flush=True)
    return 0

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as error:
        print('[v9 freeze] ERROR:',str(error),file=sys.stderr,flush=True)
        sys.exit(2)
