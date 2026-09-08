#!/usr/bin/env python3
import argparse, csv, hashlib, io, json, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_CONTRACT_SHA256 = "a122b393e24c39d6e177c46d48556dfbb95c2072ba5c5ce9e2a6cf950c36c87f"
EXPECTED_COMBINATION = "base_or_model_and_not_veto"
EXPECTED_VETO_NAME = "veto__hgbmax_le_0.005"
EXPECTED_CANDIDATE = "alarm__model__severe_harmful__old_exact__extra_trees_depth5__rt1p0"
EXPECTED_HIST_TP_FN_FP = (194, 1, 18)


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def metrics(y, pred):
    y=np.asarray(y,dtype=int); p=np.asarray(pred,dtype=int)
    tp=int(((p==1)&(y==1)).sum()); fn=int(((p==0)&(y==1)).sum())
    fp=int(((p==1)&(y==0)).sum()); tn=int(((p==0)&(y==0)).sum())
    return {
        'tp':tp,'fn':fn,'fp':fp,'tn':tn,
        'recall': tp/(tp+fn) if tp+fn else None,
        'far': fp/(fp+tn) if fp+tn else None,
        'precision': tp/(tp+fp) if tp+fp else None,
    }


def load_contract(contract_zip: Path):
    digest=sha256(contract_zip)
    if digest != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError(f"Unexpected Table5 contract SHA256: {digest}; expected {EXPECTED_CONTRACT_SHA256}")
    with zipfile.ZipFile(contract_zip,'r') as z:
        prefix='table5_reproducibility_contract/artifact/'
        model_df=pd.read_csv(io.BytesIO(z.read(prefix+'policy/v61_feature_model_oof_results_by_scope.csv')))
        pts=pd.read_csv(io.BytesIO(z.read(prefix+'policy/table5_reported_operating_points.csv')))
        replay=pd.read_csv(io.BytesIO(z.read(prefix+'replay/table5_replay_vs_historical.csv')))
    q=model_df[(model_df.label=='severe_harmful') & (model_df.scope=='ALL') &
               (model_df.feature_set=='old_exact') & (model_df.model=='extra_trees_depth5') &
               (np.isclose(model_df.target_train_recall.astype(float),1.0))]
    if len(q)!=1:
        raise RuntimeError(f"Expected exactly one historical severe old_exact ExtraTrees rt1.0 row, got {len(q)}")
    row=q.iloc[0]
    threshold=float(row.threshold)
    if str(row.candidate_name)!=EXPECTED_CANDIDATE:
        raise RuntimeError(f"Unexpected candidate: {row.candidate_name}")
    p=pts[(pts.label=='severe_harmful') & (pts.operating_point=='B')]
    if len(p)!=1:
        raise RuntimeError('Could not uniquely identify Table5 severe-harmful B operating point')
    pp=p.iloc[0]
    if (int(pp.tp),int(pp.fn),int(pp.fp))!=EXPECTED_HIST_TP_FN_FP:
        raise RuntimeError(f"Unexpected reported severe-B counts: {(pp.tp,pp.fn,pp.fp)}")
    rr=replay[(replay.label=='severe_harmful') & (replay.scope=='ALL') &
              (replay.candidate_name==EXPECTED_CANDIDATE) &
              (replay.combination==EXPECTED_COMBINATION) &
              (replay.veto_signal==EXPECTED_VETO_NAME)]
    if len(rr)!=1 or not bool(rr.iloc[0]['match']):
        raise RuntimeError('Historical severe-B exact replay candidate not uniquely verified')
    return threshold, {
        'contract_sha256':digest,
        'candidate_name':EXPECTED_CANDIDATE,
        'combination':EXPECTED_COMBINATION,
        'veto_signal':EXPECTED_VETO_NAME,
        'historical_threshold':threshold,
        'historical_current5_counts': {'tp':194,'fn':1,'fp':18,'tn':5787},
        'historical_current5_recall':194/195,
        'historical_current5_far':18/(18+5787),
        'historical_threshold_origin':'Table5 exact-replay contract generated before held-out 4819 evaluation',
    }


def stratify(df, group_col, pred_col, y_col='severe_harmful'):
    rows=[]
    vals=df[group_col].fillna('none').astype(str)
    for val in sorted(vals.unique()):
        g=df[vals==val]
        m=metrics(g[y_col],g[pred_col])
        rows.append({group_col:val,'rows':len(g),'positive_total':int(g[y_col].sum()),'negative_total':int((g[y_col]==0).sum()),**m})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',default='/root/autodl-tmp/DriveVLA')
    ap.add_argument('--run-dir',default='/root/autodl-tmp/DriveVLA/reviewer_exp/R1C1_broader4819/output/formal_seed970001')
    ap.add_argument('--contract-zip',default='/root/autodl-tmp/DriveVLA/reviewer_exp/R1C1_broader4819/inputs/Table5_REPRODUCIBILITY_CONTRACT_EXACT_REPLAY_20260829.zip')
    args=ap.parse_args()
    run=Path(args.run_dir); contract=Path(args.contract_zip)
    out=run/'reviewer_evaluation_historical_table5B'
    out.mkdir(parents=True,exist_ok=True)

    hist_thr, provenance=load_contract(contract)
    per_path=run/'reviewer_evaluation'/'r1c1_formal_per_sample_evaluation.csv'
    compact_path=run/'reviewer_online_export'/'r1c1_online_policy_compact.csv'
    if not per_path.exists() or not compact_path.exists():
        raise FileNotFoundError('Need completed v1.5 evaluation and online compact records; do not rerun inference')
    df=pd.read_csv(per_path)
    compact=pd.read_csv(compact_path)
    required=['sample_id','severe_harmful','feature_model_score','base_alarm','hgbmax_score','feature_model_threshold','final_alarm','fault_op','exact_layer']
    miss=[c for c in required if c not in df.columns]
    if miss: raise KeyError(f'Missing required columns: {miss}')
    if len(df)!=4819 or df.sample_id.nunique()!=4819:
        raise RuntimeError(f'Expected 4819 unique held-out samples; got rows={len(df)} unique={df.sample_id.nunique()}')
    if set(df.sample_id)!=set(compact.sample_id):
        raise RuntimeError('Per-sample evaluation and online compact sample sets differ')

    # IMPORTANT: no threshold search is performed here. The threshold is read from the historical
    # exact-replay artifact whose SHA256 was already frozen in the preflight before this held-out run.
    df['historical_table5B_feature_threshold']=hist_thr
    df['historical_table5B_feature_alarm']=(pd.to_numeric(df.feature_model_score,errors='coerce').fillna(-np.inf)>=hist_thr).astype(int)
    base=pd.to_numeric(df.base_alarm,errors='coerce').fillna(0).astype(int)
    hgbmax=pd.to_numeric(df.hgbmax_score,errors='coerce').fillna(np.inf)
    df['historical_table5B_veto']=((base==1)&(hgbmax<=0.005)).astype(int)
    df['historical_table5B_pre_veto_alarm']=np.maximum(base.to_numpy(),df.historical_table5B_feature_alarm.to_numpy()).astype(int)
    df['historical_table5B_final_alarm']=((df.historical_table5B_pre_veto_alarm==1)&(df.historical_table5B_veto==0)).astype(int)

    y=df.severe_harmful.astype(int)
    corrected=metrics(y,df.historical_table5B_final_alarm)
    feature_only=metrics(y,df.historical_table5B_feature_alarm)
    base_only=metrics(y,base)
    pre_veto=metrics(y,df.historical_table5B_pre_veto_alarm)
    wrong_impl=metrics(y,pd.to_numeric(df.final_alarm,errors='coerce').fillna(0).astype(int))
    veto_count=int(df.historical_table5B_veto.sum())
    veto_pos=int(((df.historical_table5B_veto==1)&(y==1)).sum())
    veto_neg=int(((df.historical_table5B_veto==1)&(y==0)).sum())

    # Sanity: score source is the exact score saved during the held-out run; no inference/model refit here.
    observed_runtime_thresholds=sorted(pd.to_numeric(df.feature_model_threshold,errors='coerce').dropna().unique().tolist())
    summary={
      'version':'R1C1_Table5_FrozenPolicy_Broader4819_v1_6_historical_Table5B_reevaluation',
      'status':'corrected operating-point evaluation; held-out inference is reused unchanged',
      'rows':len(df),
      'unique_samples':int(df.sample_id.nunique()),
      'effective_injection_count':int(pd.to_numeric(df.injected_effective,errors='coerce').fillna(0).astype(int).sum()) if 'injected_effective' in df else None,
      'severe_positive':int(y.sum()),
      'severe_negative':int((y==0).sum()),
      'policy_provenance':provenance,
      'correction_note':(
          'The v1.0-v1.5 deployment wrapper mistakenly recalibrated the feature-model threshold on the retained 10-seed development material. '
          'This reevaluation restores the pre-existing historical Table 5 severe-harmful B threshold from the exact-replay contract. '
          'No held-out score/label is used to choose or tune the threshold; the 4819 inference, model scores, base detector scores, veto scores, and labels are reused unchanged.'
      ),
      'model_note':(
          'The deployable ExtraTrees estimator is the final-refit model frozen before the 4819 test using retained historical development material under the Table-5 old_exact/ExtraTrees contract. '
          'The operating threshold used here is the historical current5 Table-5-B threshold, not the erroneous v1.0-v1.5 recalibrated threshold.'
      ),
      'erroneous_wrapper_thresholds_observed':observed_runtime_thresholds,
      'erroneous_wrapper_metrics_for_audit_only':wrong_impl,
      'historical_table5B_corrected_policy':corrected,
      'historical_table5B_feature_only':feature_only,
      'historical_table5B_base_only':base_only,
      'historical_table5B_pre_veto':pre_veto,
      'veto_audit':{'count':veto_count,'vetoed_positive':veto_pos,'vetoed_negative':veto_neg},
      'no_test_driven_threshold_search':True,
    }
    (out/'historical_table5B_policy_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    policy_manifest = {
      'policy_name':'historical_Table5_severe_B_deployable_refit',
      'target':'severe_harmful',
      'feature_set':'old_exact',
      'feature_count':2605,
      'model_family':'extra_trees_depth5',
      'selected_k':256,
      'feature_model_threshold':hist_thr,
      'threshold_source':'pre-existing Table5 exact-replay contract; severe_harmful/ALL/old_exact/extra_trees_depth5/rt1.0',
      'combination':'(base_alarm OR feature_model_alarm) AND NOT veto',
      'base_detector':'lineA_old_extra_trees_high_recall_online_all_no_temporal',
      'veto':'base_alarm == 1 AND max(old_hgb_score, speed_hgb_score) <= 0.005',
      'test_threshold_tuning':False,
      'heldout_inference_reused_without_rerun':True,
      'deployable_model_file':'../../../../frozen_policy/table5_severe_old_exact_extratrees_final.joblib',
      'note':'The final-refit estimator was frozen before held-out inference; this manifest corrects only the wrapper operating threshold to the pre-existing historical Table5-B value.'
    }
    (out/'historical_table5B_policy_manifest.json').write_text(json.dumps(policy_manifest,indent=2),encoding='utf-8')
    pd.DataFrame([{
        'policy':'historical_Table5_severe_B',
        'threshold':hist_thr,
        **corrected,
        'rows':len(df),'severe_positive':int(y.sum()),'severe_negative':int((y==0).sum()),
    }]).to_csv(out/'r1c1_formal_metrics_historical_table5B.csv',index=False)
    (out/'r1c1_formal_metrics_historical_table5B.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    df.to_csv(out/'r1c1_formal_per_sample_historical_table5B.csv',index=False)
    stratify(df,'fault_op','historical_table5B_final_alarm').to_csv(out/'fault_operator_stratified_metrics_historical_table5B.csv',index=False)
    stratify(df,'exact_layer','historical_table5B_final_alarm').to_csv(out/'exact_layer_stratified_metrics_historical_table5B.csv',index=False)

    print(json.dumps({
      'pass':True,
      'rows':len(df),
      'severe_positive':int(y.sum()),
      'historical_table5B_threshold':hist_thr,
      'corrected_policy':corrected,
      'veto_audit':summary['veto_audit'],
      'output_dir':str(out),
      'no_test_driven_threshold_search':True,
    },indent=2))

if __name__=='__main__': main()
