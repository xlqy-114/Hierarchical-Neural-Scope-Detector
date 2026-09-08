#!/usr/bin/env python3
from __future__ import annotations
import math
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def build_extra_trees_pipeline(k: int, seed: int):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("select", SelectKBest(score_func=f_classif, k=int(k))),
        ("model", ExtraTreesClassifier(
            n_estimators=160,
            max_depth=5,
            min_samples_leaf=3,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=int(seed),
        )),
    ])

def predict_score(model, X):
    if hasattr(model, "predict_proba"):
        p=model.predict_proba(X)
        if getattr(p,"ndim",1)==2 and p.shape[1]>=2:
            return np.asarray(p[:,1],dtype=np.float64)
        return np.asarray(p,dtype=np.float64).reshape(-1)
    if hasattr(model,"decision_function"):
        z=np.asarray(model.decision_function(X),dtype=np.float64).reshape(-1)
        return 1.0/(1.0+np.exp(-z))
    return np.asarray(model.predict(X),dtype=np.float64).reshape(-1)

def choose_threshold_for_recall(y, score, target=1.0):
    y=np.asarray(y,dtype=np.int64)
    score=np.asarray(score,dtype=np.float64)
    vals=np.unique(score[np.isfinite(score)])
    if int(y.sum())==0 or len(vals)==0:
        return 1.0
    best=float(vals.min()-1e-12); best_fp=math.inf
    for t in vals:
        pred=score>=t
        tp=int(((pred==1)&(y==1)).sum()); fn=int(((pred==0)&(y==1)).sum())
        fp=int(((pred==1)&(y==0)).sum())
        rec=tp/(tp+fn) if tp+fn else 0.0
        if rec+1e-12>=target and (fp<best_fp or (fp==best_fp and float(t)>best)):
            best_fp=fp; best=float(t)
    return best

def top5_robust_score(X, med, scale):
    X=np.asarray(X,dtype=np.float64)
    Z=np.abs((X-med)/scale)
    if Z.shape[1]==0:
        return np.zeros(Z.shape[0],dtype=np.float64)
    k=min(5,Z.shape[1])
    part=np.partition(Z,-k,axis=1)[:,-k:]
    return np.nanmean(part,axis=1)
