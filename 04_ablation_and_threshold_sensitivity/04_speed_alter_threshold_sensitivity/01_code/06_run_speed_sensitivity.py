#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strong speed-alter threshold sensitivity at the reported low-FAR operating point.

The continuous speed metric candidate is NOT chosen by current-label fit.  It must
be supplied explicitly after independent confirmation from the historical behavior
audit.  For each alternative speed threshold, the detector is refit from scratch
under the same five-seed LOSO procedure.  The inherited harmful criterion remains:
delta_l2_avg > 0.5 OR delta_collision_avg > 0.
"""
from __future__ import annotations
import argparse, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import sklearn
warnings.filterwarnings("ignore")

SEEDS=[955916,955917,955918,955919,955920]
DETECTORS=[
    "lineA_speed_alter_hgb_online_compact",
    "lineA_old_hgb_online_all_no_temporal",
    "lineA_old_extra_trees_high_recall_online_all_no_temporal",
]
EXPECTED=(205,4,7,5784)
HISTORICAL_CANDIDATE="segment_no_origin__none__signed_error_degradation"

def sid(x):
    s=str(x)
    return s[:-11] if s.endswith("_trajectory") else s

def load_inline(campaign:Path, seed_root:str):
    rows=[]
    for seed in SEEDS:
        p=campaign/seed_root/f"seed{seed}"/"results"/"v53_inline_detector_records.jsonl"
        if not p.is_file():
            p=campaign/seed_root/f"seed{seed}"/"results"/"v54_inline_detector_records.jsonl"
        if not p.is_file():
            raise FileNotFoundError(p)
        n=0
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            o=json.loads(line)
            a=o.get("analysis_only") or {}
            dm=a.get("delta_metrics") or {}
            dets=o.get("detectors") or {}
            r={
                "seed":seed,
                "sample_id":sid(o.get("sample_id") or o.get("raw_sample_id")),
                "delta_l2_avg":float(dm["delta_l2_avg"]),
                "delta_collision_avg":float(dm["delta_collision_avg"]),
            }
            for d in DETECTORS:
                x=dets.get(d) or {}
                r[f"score__{d}"]=float(x["score"])
                r[f"alarm__{d}"]=int(x["alarm"])
            rows.append(r)
            n+=1
        if n!=1200:
            raise RuntimeError(f"seed {seed}: expected 1200 rows, got {n}")
    df=pd.DataFrame(rows).sort_values(["seed","sample_id"]).reset_index(drop=True)
    if len(df)!=6000 or df.duplicated(["seed","sample_id"]).any():
        raise RuntimeError("6000-row inline detector alignment failed")
    return df

def sanitize_np(X):
    X=np.asarray(X,dtype=np.float64)
    X[~np.isfinite(X)]=np.nan
    return np.clip(X,-1e6,1e6)

def model(seed):
    return Pipeline([
        ("sanitize",FunctionTransformer(sanitize_np,validate=False)),
        ("impute",SimpleImputer(strategy="median")),
        ("var",VarianceThreshold(0.0)),
        ("select",SelectKBest(score_func=f_classif,k=6)),
        ("model",HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=seed,
        )),
    ])

def positive_score(m,X):
    p=m.predict_proba(X)
    return p[:,1] if p.shape[1]==2 else np.zeros(len(X),dtype=float)

def choose_training_recall_threshold(y,s,target=1.0):
    vals=np.unique(s[np.isfinite(s)])
    if int(y.sum())==0 or len(vals)==0:
        return 1.0
    best=float(vals.min()-1e-12)
    bestfp=math.inf
    for t in vals:
        pred=s>=t
        tp=int(((pred==1)&(y==1)).sum())
        fn=int(((pred==0)&(y==1)).sum())
        fp=int(((pred==1)&(y==0)).sum())
        rec=tp/(tp+fn) if tp+fn else 0.0
        if rec+1e-12>=target and (fp<bestfp or (fp==bestfp and float(t)>best)):
            bestfp=fp
            best=float(t)
    return best

def fit_oof(df,y,cols):
    scores=np.full(len(df),np.nan,dtype=float)
    fold_thresholds=[]
    seedarr=df["seed"].to_numpy(int)
    for held in SEEDS:
        te=seedarr==held
        tr=~te
        m=model(held%100000)
        m.fit(df.loc[tr,cols],y[tr])
        train_scores=positive_score(m,df.loc[tr,cols])
        test_scores=positive_score(m,df.loc[te,cols])
        scores[te]=test_scores
        fold_thresholds.append(
            choose_training_recall_threshold(y[tr],train_scores,1.0)
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("OOF score vector is incomplete")
    return scores,float(np.median(fold_thresholds)),fold_thresholds

def metrics(y,p):
    tp=int(((p==1)&(y==1)).sum())
    fn=int(((p==0)&(y==1)).sum())
    fp=int(((p==1)&(y==0)).sum())
    tn=int(((p==0)&(y==0)).sum())
    rec=tp/(tp+fn) if tp+fn else np.nan
    far=fp/(fp+tn) if fp+tn else np.nan
    return tp,fn,fp,tn,rec,far

ap=argparse.ArgumentParser()
ap.add_argument("--campaign-dir",required=True)
ap.add_argument("--seed-root",default="A1_internal_neural_scdout_v54_2_online_singlefault")
ap.add_argument("--candidate-values",required=True)
ap.add_argument("--historical-definition-json",required=True)
ap.add_argument("--output-dir",required=True)
ap.add_argument("--clean-grid",default="0.30,0.35,0.40,0.45,0.50")
ap.add_argument("--gt-grid",default="0.10,0.15,0.20,0.25,0.30")
ap.add_argument("--require-sklearn-1-2-2",action="store_true")
args=ap.parse_args()

if args.require_sklearn_1_2_2 and sklearn.__version__!="1.2.2":
    raise RuntimeError(f"formal run requires sklearn 1.2.2, current={sklearn.__version__}")

histdef=json.load(open(args.historical_definition_json,encoding="utf-8"))
selected=histdef.get("selected_candidate")
if selected!=HISTORICAL_CANDIDATE:
    raise RuntimeError(
        f"historical audit did not confirm expected candidate: {selected}"
    )

vals=pd.read_csv(args.candidate_values)
vals=vals[vals["candidate"].eq(HISTORICAL_CANDIDATE)].copy()
if len(vals)!=6000:
    raise RuntimeError(f"expected 6000 rows for historical candidate, got {len(vals)}")
vals["sample_id"]=vals["sample_id"].map(sid)
vals=vals.sort_values(["seed","sample_id"]).reset_index(drop=True)

df=load_inline(Path(args.campaign_dir),args.seed_root)
m=df.merge(
    vals[[
        "seed","sample_id",
        "fault_vs_clean_mean_step_norm_abs_delta",
        "degradation_vs_clean_gt_mean_step_norm_abs_delta",
        "speed_alter_historical",
    ]],
    on=["seed","sample_id"],
    how="left",
    validate="one_to_one",
)
if m.isna().any().any():
    bad=[c for c in m.columns if m[c].isna().any()]
    raise RuntimeError(f"missing values after merge: {bad}")

score_cols=(
    [f"score__{d}" for d in sorted(DETECTORS)]
    + [f"alarm__{d}" for d in sorted(DETECTORS)]
)

old=(
    (m["delta_l2_avg"].to_numpy(float)>0.5)
    | (m["delta_collision_avg"].to_numpy(float)>0)
).astype(int)
c=m["fault_vs_clean_mean_step_norm_abs_delta"].to_numpy(float)
g=m["degradation_vs_clean_gt_mean_step_norm_abs_delta"].to_numpy(float)

def label(tc,tg):
    return (old | ((c>=tc)&(g>=tg))).astype(int)

# Exact current-label replay of the historical speed-alter definition.
y0=label(0.40,0.20)
hist=m["speed_alter_historical"].to_numpy(int)
if not np.array_equal(y0,hist):
    raise RuntimeError(
        f"confirmed historical speed definition does not reproduce saved labels: "
        f"mismatches={int((y0!=hist).sum())}"
    )
if int(y0.sum())!=209:
    raise RuntimeError(f"expected 209 original speed-alter positives, got {int(y0.sum())}")
print("[OK] confirmed historical speed definition reproduces 209/6000 saved labels exactly")

settings=[]
for tc in [float(x) for x in args.clean_grid.split(",") if x.strip()]:
    settings.append(("fault_vs_clean_speed_threshold",tc,0.20))
for tg in [float(x) for x in args.gt_grid.split(",") if x.strip()]:
    if abs(tg-0.20)<1e-12:
        continue
    settings.append(("gt_speed_degradation_threshold",0.40,tg))

rows=[]
seedrows=[]
oof=m[["seed","sample_id"]].copy()

for varied,tc,tg in settings:
    y=label(tc,tg)
    print(
        f"[RUN] speed thresholds: fault-vs-clean>={tc:g}, "
        f"GT-degradation>={tg:g}, positives={int(y.sum())}",
        flush=True,
    )
    scores,thr,foldthr=fit_oof(m,y,score_cols)
    pred=(scores>=thr).astype(int)
    tp,fn,fp,tn,rec,far=metrics(y,pred)

    if abs(tc-0.40)<1e-12 and abs(tg-0.20)<1e-12:
        got=(tp,fn,fp,tn)
        if got!=EXPECTED:
            raise RuntimeError(
                f"EXACT TABLE-5 SPEED REPLAY FAILED: expected={EXPECTED}, got={got}"
            )
        print("[OK] exact Table-5 speed low-FAR replay matched:",got)

    rows.append({
        "varied_parameter":varied,
        "speed_fault_vs_clean_threshold":tc,
        "speed_gt_degradation_threshold":tg,
        "inherited_harmful_l2_threshold":0.5,
        "inherited_collision_rule":">0 (fixed)",
        "positive_total":int(y.sum()),
        "negative_total":int((y==0).sum()),
        "tp":tp,"fn":fn,"fp":fp,"tn":tn,
        "recall":rec,"far":far,
        "decision_threshold":thr,
        "fold_training_thresholds":json.dumps([float(x) for x in foldthr]),
        "historical_step_construction":"segment_no_origin",
        "historical_gt_degradation_form":"signed_error_degradation",
        "configuration":"selected detector scores + HGB; model-only; 5-seed LOSO",
        "sklearn_version":sklearn.__version__,
    })

    tag=f"c{str(tc).replace('.','p')}__g{str(tg).replace('.','p')}"
    oof[f"label__{tag}"]=y
    oof[f"score__{tag}"]=scores
    oof[f"pred__{tag}"]=pred

    for seed in SEEDS:
        mask=m["seed"].to_numpy(int)==seed
        stp,sfn,sfp,stn,srec,sfar=metrics(y[mask],pred[mask])
        seedrows.append({
            "varied_parameter":varied,
            "speed_fault_vs_clean_threshold":tc,
            "speed_gt_degradation_threshold":tg,
            "seed":seed,
            "positive_total":int(y[mask].sum()),
            "negative_total":int((y[mask]==0).sum()),
            "tp":stp,"fn":sfn,"fp":sfp,"tn":stn,
            "recall":srec,"far":sfar,
        })

out=Path(args.output_dir)
out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/"speed_threshold_sensitivity_summary.csv",index=False)
pd.DataFrame(seedrows).to_csv(out/"speed_threshold_sensitivity_seedwise.csv",index=False)
oof.to_csv(out/"speed_threshold_sensitivity_oof_predictions.csv",index=False)
m[[
    "seed","sample_id","delta_l2_avg","delta_collision_avg",
    "fault_vs_clean_mean_step_norm_abs_delta",
    "degradation_vs_clean_gt_mean_step_norm_abs_delta",
    "speed_alter_historical",
]].to_csv(out/"speed_continuous_metrics_6000.csv",index=False)

(out/"run_summary.json").write_text(
    json.dumps({
        "rows":6000,
        "seeds":SEEDS,
        "historically_confirmed_candidate":HISTORICAL_CANDIDATE,
        "baseline_expected_tp_fn_fp_tn":EXPECTED,
        "sklearn_version":sklearn.__version__,
        "note":"Speed-specific thresholds are varied one at a time; inherited harmful threshold remains fixed at 0.5 m and collision condition at >0.",
    },indent=2),
    encoding="utf-8",
)

print("[OK] speed threshold sensitivity complete")
print(
    pd.DataFrame(rows)[[
        "varied_parameter",
        "speed_fault_vs_clean_threshold",
        "speed_gt_degradation_threshold",
        "positive_total","tp","fn","fp","tn","recall","far",
    ]].to_string(index=False)
)
