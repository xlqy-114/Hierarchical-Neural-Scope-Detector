#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier

BASE='lineA_old_extra_trees_high_recall_online_all_no_temporal'
HGB_OLD='lineA_old_hgb_online_all_no_temporal'
HGB_SPEED='lineA_speed_alter_hgb_online_compact'
VETO_THR=0.005
TARGET_RECALL=1.0
K=256
SEEDS=list(range(954301,954311))
FINAL_RANDOM_STATE=0

def sha256(path: Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def sanitize(X):
    X=np.asarray(X,dtype=np.float64)
    X[~np.isfinite(X)]=np.nan
    np.clip(X,-1e6,1e6,out=X)
    return X

def build_model(seed:int):
    return Pipeline([
        ('impute',SimpleImputer(strategy='median')),
        ('var',VarianceThreshold(0.0)),
        ('select',SelectKBest(score_func=f_classif,k=K)),
        ('model',ExtraTreesClassifier(n_estimators=160,max_depth=5,min_samples_leaf=3,n_jobs=-1,class_weight='balanced_subsample',random_state=seed)),
    ])

def score(model,X):
    p=model.predict_proba(X)
    return p[:,1] if p.ndim==2 and p.shape[1]==2 else np.zeros(len(X),dtype=float)

def threshold_for_100pct_recall(y, s):
    pos=np.asarray(s)[np.asarray(y).astype(int)==1]
    pos=pos[np.isfinite(pos)]
    if len(pos)==0: raise ValueError('No positive score available for 100% recall threshold')
    # Exact equivalent of historical choose_threshold_for_recall(..., 1.0):
    # the largest threshold that keeps every training positive is min positive score.
    return float(np.min(pos))

def metrics(y,p):
    y=np.asarray(y).astype(int); p=np.asarray(p).astype(int)
    tp=int(((y==1)&(p==1)).sum()); fn=int(((y==1)&(p==0)).sum())
    fp=int(((y==0)&(p==1)).sum()); tn=int(((y==0)&(p==0)).sum())
    return {'tp':tp,'fn':fn,'fp':fp,'tn':tn,'recall':tp/(tp+fn) if tp+fn else None,'far':fp/(fp+tn) if fp+tn else None,'precision':tp/(tp+fp) if tp+fp else None}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); args=ap.parse_args()
    root=Path(args.root); ext=root/'work/inputs_extracted'; src=ext/'R1C1_Table5_freeze_training_inputs_v54formal10'
    out=root/'frozen_policy'; out.mkdir(parents=True,exist_ok=True)
    feat_path=src/'v54_online_deployable_feature_matrix.csv'; lab_path=src/'v54_posthoc_label_table_harmful_severe.csv'; score_path=src/'v54_detector_score_cache.csv'
    features=pd.read_csv(src/'old_exact_2605_features.csv')['feature'].astype(str).tolist()
    if len(features)!=2605 or len(set(features))!=2605: raise SystemExit('old_exact feature contract invalid')
    usecols=['seed','sample_id']+features
    feat=pd.read_csv(feat_path,usecols=usecols)
    lab=pd.read_csv(lab_path,usecols=['seed','sample_id','severe_harmful','injected_effective'])
    df=feat.merge(lab,on=['seed','sample_id'],how='inner',validate='one_to_one')
    if len(df)!=12000 or sorted(df.seed.astype(int).unique().tolist())!=SEEDS: raise SystemExit('historical development join contract failed')
    X=sanitize(df[features].to_numpy(dtype=np.float64,copy=True)); y=df.severe_harmful.astype(int).to_numpy(); seedarr=df.seed.astype(int).to_numpy()
    oof=np.full(len(df),np.nan); fold_rows=[]; fold_thresholds=[]
    for seed in SEEDS:
        test=seedarr==seed; train=~test
        model=build_model(seed=int(seed)%100000)
        model.fit(X[train],y[train])
        tr=score(model,X[train]); te=score(model,X[test]); oof[test]=te
        thr=threshold_for_100pct_recall(y[train],tr); fold_thresholds.append(thr)
        pm=metrics(y[test],te>=thr)
        fold_rows.append({'heldout_seed':seed,'train_rows':int(train.sum()),'test_rows':int(test.sum()),'train_positive':int(y[train].sum()),'test_positive':int(y[test].sum()),'fold_train_threshold_rt1p0':thr,**pm})
        print(f'[freeze] heldout={seed} threshold={thr:.12g} test_recall={pm["recall"]:.6f} test_far={pm["far"]:.6f}',flush=True)
    if not np.isfinite(oof).all(): raise SystemExit('OOF scores incomplete')
    deploy_thr=float(np.median(np.asarray(fold_thresholds,dtype=float)))

    # Historical v35 score cache supplies the exact base/veto signals on development data.
    sc=pd.read_csv(score_path)
    piv=sc.pivot_table(index=['seed','sample_id'],columns='detector',values=['score','alarm'],aggfunc='first')
    piv.columns=[f'{a}__{b}' for a,b in piv.columns]; piv=piv.reset_index()
    dev=pd.DataFrame({'seed':df.seed.astype(int),'sample_id':df.sample_id.astype(str),'severe_harmful':y,'feature_model_score_oof':oof})
    dev=dev.merge(piv,on=['seed','sample_id'],how='left',validate='one_to_one')
    for c in [f'alarm__{BASE}',f'score__{HGB_OLD}',f'score__{HGB_SPEED}']:
        if c not in dev.columns or dev[c].isna().any(): raise SystemExit(f'Missing development base/veto signal: {c}')
    dev['feature_model_alarm']=(dev.feature_model_score_oof>=deploy_thr).astype(int)
    dev['base_alarm']=pd.to_numeric(dev[f'alarm__{BASE}']).astype(int)
    dev['hgbmax_score']=np.maximum(pd.to_numeric(dev[f'score__{HGB_OLD}']).to_numpy(),pd.to_numeric(dev[f'score__{HGB_SPEED}']).to_numpy())
    dev['veto']=((dev.base_alarm==1)&(dev.hgbmax_score<=VETO_THR)).astype(int)
    dev['pre_veto_alarm']=np.maximum(dev.base_alarm,dev.feature_model_alarm)
    dev['final_alarm']=((dev.pre_veto_alarm==1)&(dev.veto==0)).astype(int)
    dev_metrics=metrics(y,dev.final_alarm.to_numpy())
    dev.to_csv(out/'development_oof_frozen_policy_predictions.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(out/'development_oof_fold_metrics.csv',index=False)

    # Fit one deployable model on all historical development rows after all policy choices are frozen.
    final_model=build_model(seed=FINAL_RANDOM_STATE); final_model.fit(X,y)
    model_path=out/'table5_severe_old_exact_extratrees_final.joblib'; joblib.dump(final_model,model_path,compress=3)
    feature_json=out/'old_exact_2605_features.json'; feature_json.write_text(json.dumps(features,ensure_ascii=False,indent=2),encoding='utf-8')
    # Export final selected 256 names for inspection only; runtime still uses exact 2605 contract.
    var=final_model.named_steps['var']; sel=final_model.named_steps['select']
    var_names=np.asarray(features,dtype=object)[var.get_support()]
    selected=var_names[sel.get_support()].tolist()
    pd.DataFrame({'index':range(len(selected)),'feature':selected}).to_csv(out/'selected_256_features_final.csv',index=False)

    # Historical current5 threshold is provenance only and is intentionally not applied to the refit model.
    contract=ext/'table5_reproducibility_contract/artifact/policy/v61_feature_model_oof_results_by_scope.csv'
    h=pd.read_csv(contract); h=h[(h.label=='severe_harmful')&(h.scope=='ALL')&(h.feature_set=='old_exact')&(h.model=='extra_trees_depth5')&(h.target_train_recall==1.0)]
    historical_current5_threshold=float(h.iloc[0].threshold)
    manifest={
      'policy_version':'R1C1_Table5_FrozenPolicy_Broader4819_v1_5',
      'freeze_timing':'completed_before_any_4819_heldout_inference_or_label_inspection',
      'label':'severe_harmful',
      'feature_set':'old_exact','feature_count':2605,'model_name':'extra_trees_depth5','selected_k':256,
      'model_hyperparameters':{'n_estimators':160,'max_depth':5,'min_samples_leaf':3,'class_weight':'balanced_subsample','random_state_final_fit':FINAL_RANDOM_STATE},
      'preprocessing':['float64','nonfinite_to_nan','clip_-1e6_1e6','median_impute','variance_threshold_0','SelectKBest_f_classif_256'],
      'target_train_recall':1.0,
      'threshold_calibration':'10-seed leave-one-seed-out; per-fold threshold=min positive training score (exact rt=1.0 equivalent); deployment threshold=median of 10 fold thresholds',
      'fold_thresholds':fold_thresholds,'deployment_threshold':deploy_thr,
      'historical_current5_oof_threshold_provenance_only':historical_current5_threshold,
      'combination':'base_or_model_and_not_veto','base_detector':BASE,
      'veto_definition':'base_alarm == 1 AND max(old_hgb_score, speed_hgb_score) <= 0.005',
      'veto_hgb_old_detector':HGB_OLD,'veto_hgb_speed_detector':HGB_SPEED,'veto_hgbmax_threshold':VETO_THR,
      'historical_development_rows':12000,'historical_development_seeds':SEEDS,'historical_development_unique_samples':1200,'historical_development_severe_positives':int(y.sum()),
      'development_oof_frozen_policy_metrics':dev_metrics,
      'information_boundary':'runtime model uses current-inference online features and v35.1 runtime scores only; clean/GT/fault metadata are excluded',
      'model_sha256':'','feature_contract_sha256':'',
    }
    manifest['model_sha256']=sha256(model_path); manifest['feature_contract_sha256']=sha256(feature_json)
    (out/'frozen_policy.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'FROZEN_BEFORE_TEST.txt').write_text('This policy was frozen before any held-out 4819 inference or held-out label evaluation. Do not modify model, threshold, veto, feature contract, or alarm logic after starting the formal test.\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
