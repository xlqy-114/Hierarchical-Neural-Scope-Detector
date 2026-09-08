#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

REQUIRED_RESOURCE_FILES = [
    'official_full_sample_ids_6019.json',
    'v73_9_hgb_online_feature_contract.csv',
    'v73_9_hgb_runtime_collection_columns.csv',
    'v73_9_hgb_generated_feature_contract.csv',
    'v73_9_hgb_generated_consistency_source_columns.csv',
    'v73_9_label_and_raw_ingredient_contract.csv',
    'v73_16_standalone_required_stage_hook_specs.txt',
    'v73_16_standalone_required_stage_hook_summary.json',
    'v49_s_full_target_specs.txt',
    'v49_s_primary_target_name.txt',
    'v50_s_full_target_weights_global_random.txt',
    'v50_c_target_weights_global_random.txt',
    'v49_d_full_targets.txt',
    'v50_d_target_weights_global_random.txt',
    'v50_global_random_scope_weights.txt',
    'v50_internal_scd_global_random_scope_manifest.json',
]
FORBIDDEN_PREFIXES = ('score__','alarm__','threshold__')
FORBIDDEN_EXACT = {'hgbmax'}
FORBIDDEN_SUBSTRINGS = ('lineA_old','lineA_speed','linea_old','linea_speed')

def fail(msg: str):
    raise SystemExit('[v73.16 preflight ERROR] ' + msg)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root', type=Path, required=True)
    ap.add_argument('--package-dir', type=Path, required=True)
    ap.add_argument('--run-sample-ids', type=Path, required=True)
    ap.add_argument('--expected-rows', type=int, required=True)
    ap.add_argument('--model-path', type=Path, required=True)
    ap.add_argument('--output-json', type=Path, required=True)
    args=ap.parse_args()
    res=args.package_dir/'resources'
    missing=[]
    for name in REQUIRED_RESOURCE_FILES:
        if not (res/name).is_file():
            missing.append(str(res/name))
    for p in [args.run_sample_ids,args.model_path,args.repo_root/'drivevla/eval_share/evaluation.py',args.repo_root/'drivevla/eval_share/gt/gt_traj.pkl',args.repo_root/'drivevla/eval_share/gt/gt_traj_mask.pkl']:
        if not p.exists():
            missing.append(str(p))
    if missing:
        fail('missing required files/paths:\n  ' + '\n  '.join(missing))
    detector_like=list(res.glob('v35*detector*'))+list(res.glob('*detector*reuse*.tar.gz'))+list(res.glob('*.joblib'))
    if detector_like:
        fail('detector/model artifacts are present in standalone no-detector resources: ' + ', '.join(map(str, detector_like)))
    ids=json.loads(args.run_sample_ids.read_text(encoding='utf-8'))
    if not isinstance(ids, list):
        fail('run sample ids JSON must be a list')
    if len(ids)!=args.expected_rows:
        fail(f'expected {args.expected_rows} ids for this run, got {len(ids)}')
    if len(set(map(str,ids)))!=len(ids):
        fail('duplicate sample ids in run id file')
    contract=pd.read_csv(res/'v73_9_hgb_online_feature_contract.csv')
    collect=pd.read_csv(res/'v73_9_hgb_runtime_collection_columns.csv')
    gen=pd.read_csv(res/'v73_9_hgb_generated_feature_contract.csv')
    feats=contract['feature'].astype(str).tolist()
    forbidden=[f for f in feats if f in FORBIDDEN_EXACT or f.startswith(FORBIDDEN_PREFIXES) or any(x in f for x in FORBIDDEN_SUBSTRINGS)]
    if forbidden:
        fail('forbidden detector-derived feature names in model contract: ' + ', '.join(forbidden[:20]))
    stage_summary=json.loads((res/'v73_16_standalone_required_stage_hook_summary.json').read_text(encoding='utf-8'))
    summary={
        'version':'v73_16_standalone_resource_preflight',
        'package_dir':str(args.package_dir),
        'run_sample_ids':str(args.run_sample_ids),
        'expected_rows':args.expected_rows,
        'sample_id_count':len(ids),
        'first_id':ids[0] if ids else '',
        'last_id':ids[-1] if ids else '',
        'model_feature_count':int(contract['feature'].nunique()),
        'runtime_collection_feature_count':int(collect['feature'].nunique()),
        'generated_feature_count':int(gen['feature'].nunique()),
        'required_stage_hook_count':int(stage_summary.get('required_stage_count', -1)),
        'detector_artifact_required':False,
        'detector_artifact_present_in_package':False,
        'historical_output_dependency_required':False,
        'pass':True,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
if __name__=='__main__':
    main()
