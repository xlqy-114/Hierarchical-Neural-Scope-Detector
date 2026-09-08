#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, tarfile, zipfile, shutil
from pathlib import Path
import pandas as pd

EXPECTED_SEEDS = list(range(954301, 954311))
REQ_V35 = [
    'lineA_old_extra_trees_high_recall_online_all_no_temporal',
    'lineA_old_hgb_online_all_no_temporal',
    'lineA_speed_alter_hgb_online_compact',
]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def load_jsonl_ids(path: Path):
    ids=[]
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            sid=str(r.get('sample_id','')).strip()
            if not sid:
                raise SystemExit(f'Missing sample_id in {path}')
            ids.append(sid)
    return ids

def normalize_pred_key(x: str) -> str:
    x=str(x)
    return x[:-11] if x.endswith('_trajectory') else x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--repo-root', required=True)
    args=ap.parse_args()
    root=Path(args.root); repo=Path(args.repo_root)
    inputs=root/'inputs'; work=root/'work'; ext=work/'inputs_extracted'
    out_summary=root/'preflight_summary.json'
    if out_summary.exists():
        out_summary.unlink()
    ext.mkdir(parents=True, exist_ok=True)
    files={
      'freeze':inputs/'R1C1_Table5_freeze_training_inputs_v54formal10.tar.gz',
      'contract':inputs/'Table5_REPRODUCIBILITY_CONTRACT_EXACT_REPLAY_20260829.zip',
      'v35':inputs/'v35_1_exported_selected_detectors_for_reuse.tar.gz',
      'clean':inputs/'v73_16_clean_reference_official6019_maxtok192_for_collaborator_check.tar.gz',
    }
    missing=[str(p) for p in files.values() if not p.is_file()]
    required_paths={
      'subset1200':repo/'output/OpenDriveVLA-0.5B/subsets/dev_stress_1200/subset_ids.json',
      'cal_json':repo/'output/OpenDriveVLA-0.5B/20260506_Subset1200_S0S1_v35_1_singlefault_10seeds_pos800900/calibration/results/detector_thresholds.json',
      'output_sanity':repo/'output/OpenDriveVLA-0.5B/20260506_Subset1200_S0S1_v35_1_singlefault_10seeds_pos800900/calibration/output_sanity_thresholds.json',
      'registry_effectiveness':repo/'output/OpenDriveVLA-0.5B/20260507_FullS_v36_6_required_scan_full_leaf/v36_swide_scan_per_target_effectiveness.csv',
      'registry_targets':repo/'output/OpenDriveVLA-0.5B/20260507_FullS_v36_6_required_scan_full_leaf/target_registry/v36_swide_targets_full_leaf.csv',
      'model':repo/'ckpts/OpenDriveVLA-0.5B',
      'nuscenes':repo/'data/nuscenes',
    }
    missing += [str(p) for p in required_paths.values() if not p.exists()]
    if missing:
        raise SystemExit('Missing required inputs:\n'+'\n'.join(missing))

    # Deterministic fresh extraction.
    if ext.exists():
        shutil.rmtree(ext)
    ext.mkdir(parents=True, exist_ok=True)
    with tarfile.open(files['freeze'],'r:gz') as tf: tf.extractall(ext)
    with zipfile.ZipFile(files['contract']) as z: z.extractall(ext)
    with tarfile.open(files['v35'],'r:gz') as tf: tf.extractall(ext)
    clean_dir=ext/'clean_reference_official6019'; clean_dir.mkdir()
    with tarfile.open(files['clean'],'r:gz') as tf: tf.extractall(clean_dir)

    fr=ext/'R1C1_Table5_freeze_training_inputs_v54formal10'
    feat=fr/'v54_online_deployable_feature_matrix.csv'
    lab=fr/'v54_posthoc_label_table_harmful_severe.csv'
    score=fr/'v54_detector_score_cache.csv'
    flist=fr/'old_exact_2605_features.csv'
    for p in [feat,lab,score,flist]:
        if not p.is_file(): raise SystemExit(f'Missing extracted file: {p}')

    fh=pd.read_csv(feat, nrows=0).columns.tolist()
    fkeys=pd.read_csv(feat, usecols=['seed','sample_id'])
    ldf=pd.read_csv(lab, usecols=['seed','sample_id','severe_harmful','injected_effective'])
    old=pd.read_csv(flist)
    old_features=old['feature'].astype(str).tolist()
    if len(fkeys)!=12000 or sorted(fkeys.seed.astype(int).unique().tolist())!=EXPECTED_SEEDS or fkeys.sample_id.nunique()!=1200:
        raise SystemExit('Historical 10-seed feature matrix contract failed')
    if len(fh)!=2641 or len(old_features)!=2605 or len(set(old_features))!=2605:
        raise SystemExit('Feature-count contract failed')
    absent=[c for c in old_features if c not in fh]
    if absent: raise SystemExit(f'old_exact features absent from matrix: {absent[:20]}')
    if len(ldf)!=12000 or int(ldf.severe_harmful.sum())!=400:
        raise SystemExit('Historical severe-label contract failed')
    if ldf.duplicated(['seed','sample_id']).any() or fkeys.duplicated(['seed','sample_id']).any():
        raise SystemExit('Duplicate historical (seed,sample_id) keys')

    # Verify the exact Table-5 severe-B historical candidate contract.
    contract_root=ext/'table5_reproducibility_contract/artifact'
    replay=pd.read_csv(contract_root/'replay/table5_replay_vs_historical.csv')
    cand=replay[(replay.label.astype(str)=='severe_harmful') &
                (replay.candidate_name.astype(str)=='alarm__model__severe_harmful__old_exact__extra_trees_depth5__rt1p0') &
                (replay.combination.astype(str)=='base_or_model_and_not_veto') &
                (replay.veto_signal.astype(str)=='veto__hgbmax_le_0.005') &
                (replay.scope.astype(str)=='ALL')]
    if len(cand)!=1 or not bool(cand.iloc[0]['match']) or tuple(int(cand.iloc[0][k]) for k in ['historical_tp','historical_fn','historical_fp'])!=(194,1,18):
        raise SystemExit('Exact Table5 severe-B candidate not verified by replay contract')
    contract_old=pd.read_csv(contract_root/'schema/old_exact_2605_features.csv')['feature'].astype(str).tolist()
    if contract_old != old_features:
        raise SystemExit('old_exact feature order differs between training input and Table5 contract')
    modelrows=pd.read_csv(contract_root/'policy/v61_feature_model_oof_results_by_scope.csv')
    hist=modelrows[(modelrows.label=='severe_harmful')&(modelrows.scope=='ALL')&(modelrows.feature_set=='old_exact')&(modelrows.model=='extra_trees_depth5')&(modelrows.target_train_recall==1.0)]
    if len(hist)!=1: raise SystemExit('Historical severe old_exact model row not unique')
    historical_threshold=float(hist.iloc[0].threshold)

    # v35 detector artifacts.
    v35root=ext/'v35_1_exported_detector_artifacts'
    for name in REQ_V35:
        d=v35root/name
        for fn in ['model.joblib','feature_columns.json','threshold.json']:
            if not (d/fn).is_file(): raise SystemExit(f'Missing v35 artifact {d/fn}')

    # Full clean reference: authoritative sample IDs come from per-sample metrics.
    metric_path=clean_dir/'per_sample_planning_metrics.jsonl'
    pred_path=clean_dir/'pred_trajs_dict.json'
    clean_metric_ids=load_jsonl_ids(metric_path)
    if len(clean_metric_ids)!=6019 or len(set(clean_metric_ids))!=6019:
        raise SystemExit('Clean per-sample metric reference is not exact 6019 unique samples')
    pred=json.loads(pred_path.read_text(encoding='utf-8'))
    pred_ids=[normalize_pred_key(k) for k in pred.keys()]
    if len(pred_ids)!=6019 or len(set(pred_ids))!=6019:
        raise SystemExit('Clean trajectory reference is not exact 6019 unique samples after key normalization')
    if set(pred_ids)!=set(clean_metric_ids):
        raise SystemExit('Clean trajectory and per-sample metric sample sets differ')

    # Historical subset must equal the actual 1200 development sample set.
    subset=json.loads(required_paths['subset1200'].read_text(encoding='utf-8'))
    if not isinstance(subset,list) or len(subset)!=1200 or len(set(map(str,subset)))!=1200:
        raise SystemExit('Original subset_ids.json must be an exact JSON list of 1200 unique IDs')
    subset=[str(x) for x in subset]
    hist_samples=sorted(set(fkeys.sample_id.astype(str)))
    if set(subset)!=set(hist_samples):
        only_subset=sorted(set(subset)-set(hist_samples))[:10]
        only_hist=sorted(set(hist_samples)-set(subset))[:10]
        raise SystemExit(f'Historical 10-seed development sample set != subset1200; only_subset={only_subset}; only_history={only_hist}')
    clean_set=set(clean_metric_ids)
    miss=[x for x in subset if x not in clean_set]
    if miss: raise SystemExit(f'Original subset contains IDs absent from full clean reference: {miss[:10]}')
    unused=[x for x in clean_metric_ids if x not in set(subset)]
    if len(unused)!=4819:
        raise SystemExit(f'Expected 4819 unused samples, got {len(unused)}')

    summary={
      'version':'R1C1_Table5_FrozenPolicy_Broader4819_v1_5_preflight',
      'pass':True,
      'historical_training_rows':12000,
      'historical_training_seeds':EXPECTED_SEEDS,
      'historical_unique_samples':1200,
      'historical_severe_positives':400,
      'historical_samples_exactly_match_subset1200':True,
      'old_exact_feature_count':2605,
      'historical_current5_table5_severe_B_candidate':{'tp':194,'fn':1,'fp':18,'threshold':historical_threshold,'combination':'base_or_model_and_not_veto','veto':'hgbmax<=0.005'},
      'clean_reference_samples':6019,
      'clean_pred_key_normalization':'strip trailing _trajectory; normalized pred set equals per_sample_planning_metrics sample_id set',
      'original_subset_samples':1200,
      'unused_samples':4819,
      'input_sha256':{k:sha256(v) for k,v in files.items()},
      'paths':{k:str(v) for k,v in required_paths.items()},
      'extracted_root':str(ext),
    }
    out_summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
