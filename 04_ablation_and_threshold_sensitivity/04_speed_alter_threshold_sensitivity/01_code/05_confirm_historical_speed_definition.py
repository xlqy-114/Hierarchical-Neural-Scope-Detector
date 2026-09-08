#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confirm the exact historical speed-alter continuous metric definition.

The historical audit CSV contains the original behavior metrics and flags.  This
script uses the original clean trajectories to determine whether the step metric
was constructed from:
  (a) only adjacent predicted waypoints (5 segments), or
  (b) origin -> first waypoint plus adjacent waypoints (6 segments).

It also verifies the exact GT-degradation formula and the 0.40 / 0.20 flag logic.
No model is trained here.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def sid(x):
    s=str(x)
    return s[:-11] if s.endswith("_trajectory") else s

def canonical_traj(x):
    a=np.asarray(x,dtype=np.float64)
    while a.ndim>2 and a.shape[0]==1:
        a=a[0]
    if a.ndim!=2 or a.shape[1]<2 or a.shape[0]<6:
        raise RuntimeError(f"invalid clean trajectory shape {a.shape}")
    return a[:6,:2]

def step_mean_no_origin(t):
    return float(np.linalg.norm(np.diff(t,axis=0),axis=1).mean())

def step_mean_with_origin(t):
    z=np.vstack([np.zeros((1,2),dtype=np.float64),t])
    return float(np.linalg.norm(np.diff(z,axis=0),axis=1).mean())

ap=argparse.ArgumentParser()
ap.add_argument("--audit-csv",required=True)
ap.add_argument("--clean-trajs",required=True)
ap.add_argument("--output",required=True)
args=ap.parse_args()

audit=pd.read_csv(args.audit_csv)
required=[
    "sample_id",
    "beh__clean_step_mean",
    "beh__fault_vs_gt__mean_step_norm_abs_delta",
    "beh__clean_vs_gt__mean_step_norm_abs_delta",
    "beh__degradation_vs_clean_gt__mean_step_norm_abs_delta",
    "beh__fault_vs_clean__mean_step_norm_abs_delta",
    "beh__flag_fault_vs_clean_speed",
    "beh__flag_gt_speed_degradation",
    "beh__flag_speed_alter",
    "v31_old_harmful",
    "v31_speed_alter_harmful",
]
missing=[c for c in required if c not in audit.columns]
if missing:
    raise RuntimeError(f"historical audit is missing required columns: {missing}")

clean0=json.load(open(args.clean_trajs,encoding="utf-8"))
clean={sid(k):canonical_traj(v) for k,v in clean0.items()}
if len(clean)!=6019:
    raise RuntimeError(f"expected 6019 clean trajectories, got {len(clean)}")

# 1) Determine the historical step construction directly from the historical
#    clean_step_mean column.
no_err=[]
with_err=[]
matched=0
for r in audit.itertuples(index=False):
    s=sid(getattr(r,"sample_id"))
    v=getattr(r,"beh__clean_step_mean")
    if s not in clean or pd.isna(v):
        continue
    v=float(v)
    no_err.append(abs(v-step_mean_no_origin(clean[s])))
    with_err.append(abs(v-step_mean_with_origin(clean[s])))
    matched+=1

if matched==0:
    raise RuntimeError("no historical rows could be compared with clean trajectories")
no_err=np.asarray(no_err)
with_err=np.asarray(with_err)

no_exact=int((no_err<1e-9).sum())
with_exact=int((with_err<1e-9).sum())

if no_exact != matched:
    raise RuntimeError(
        f"historical clean_step_mean does not exactly match 5-segment no-origin "
        f"construction: {no_exact}/{matched}"
    )
if with_exact == matched:
    raise RuntimeError(
        "origin-inclusive and no-origin step constructions are indistinguishable "
        "in this historical audit"
    )

# 2) Verify historical GT degradation is SIGNED:
# fault-vs-GT step error minus clean-vs-GT step error.
fgt=pd.to_numeric(audit["beh__fault_vs_gt__mean_step_norm_abs_delta"],errors="coerce")
cgt=pd.to_numeric(audit["beh__clean_vs_gt__mean_step_norm_abs_delta"],errors="coerce")
deg=pd.to_numeric(audit["beh__degradation_vs_clean_gt__mean_step_norm_abs_delta"],errors="coerce")
valid=np.isfinite(fgt)&np.isfinite(cgt)&np.isfinite(deg)
signed=(fgt-cgt)
deg_err=np.abs(deg[valid]-signed[valid])
if int((deg_err<1e-9).sum()) != int(valid.sum()):
    raise RuntimeError(
        f"signed GT-degradation formula mismatch on "
        f"{int(valid.sum()-(deg_err<1e-9).sum())} historical rows"
    )

# 3) Verify original flags and final speed-alter label construction.
c=pd.to_numeric(audit["beh__fault_vs_clean__mean_step_norm_abs_delta"],errors="coerce")
g=pd.to_numeric(audit["beh__degradation_vs_clean_gt__mean_step_norm_abs_delta"],errors="coerce")
flag_c=audit["beh__flag_fault_vs_clean_speed"].fillna(0).astype(int)
flag_g=audit["beh__flag_gt_speed_degradation"].fillna(0).astype(int)
flag_both=audit["beh__flag_speed_alter"].fillna(0).astype(int)
old=audit["v31_old_harmful"].fillna(0).astype(int)
final=audit["v31_speed_alter_harmful"].fillna(0).astype(int)

pred_c=(c>=0.40).fillna(False).astype(int)
pred_g=(g>=0.20).fillna(False).astype(int)
pred_both=(pred_c.astype(bool)&pred_g.astype(bool)).astype(int)
pred_final=(old.astype(bool)|pred_both.astype(bool)).astype(int)

checks={
    "fault_vs_clean_speed_flag_mismatch":int((pred_c!=flag_c).sum()),
    "gt_speed_degradation_flag_mismatch":int((pred_g!=flag_g).sum()),
    "speed_alter_condition_mismatch":int((pred_both!=flag_both).sum()),
    "final_speed_alter_label_mismatch":int((pred_final!=final).sum()),
}
if any(checks.values()):
    raise RuntimeError(f"historical threshold/flag logic mismatch: {checks}")

result={
    "historical_audit_rows":int(len(audit)),
    "historical_unique_sample_ids":int(audit["sample_id"].astype(str).nunique()),
    "historical_duplicate_sample_ids":int((audit["sample_id"].astype(str).value_counts()>1).sum()),
    "clean_trajectory_rows_compared":int(matched),
    "no_origin_step_mean_exact_matches_at_1e-9":no_exact,
    "origin_inclusive_step_mean_exact_matches_at_1e-9":with_exact,
    "no_origin_step_mean_max_abs_error":float(no_err.max()),
    "origin_inclusive_step_mean_max_abs_error":float(with_err.max()),
    "gt_degradation_signed_formula_valid_rows":int(valid.sum()),
    "gt_degradation_signed_formula_exact_matches_at_1e-9":int((deg_err<1e-9).sum()),
    "flag_logic_checks":checks,
    "confirmed_historical_definition":{
        "step_construction":"segment_no_origin",
        "mask_mode":"none",
        "gt_degradation_form":"signed_error_degradation",
        "fault_vs_clean_threshold":0.40,
        "gt_degradation_threshold":0.20,
        "speed_specific_condition":"fault_vs_clean_mean_step_norm_abs_delta >= 0.40 AND degradation_vs_clean_gt_mean_step_norm_abs_delta >= 0.20",
        "final_speed_alter_label":"old_harmful OR speed_specific_condition",
    },
    "selected_candidate":"segment_no_origin__none__signed_error_degradation",
}

Path(args.output).parent.mkdir(parents=True,exist_ok=True)
Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")

print("[OK] historical speed definition uniquely confirmed")
print(f"[OK] historical audit rows: {len(audit)}; unique sample IDs: {audit['sample_id'].astype(str).nunique()}")
print(f"[OK] no-origin clean step mean exact match: {no_exact}/{matched}")
print(f"[OK] origin-inclusive clean step mean exact match: {with_exact}/{matched}")
print(f"[OK] signed GT-degradation exact match: {(deg_err<1e-9).sum()}/{valid.sum()}")
print("[OK] threshold/flag logic mismatch counts:", checks)
print("[OK] selected candidate: segment_no_origin__none__signed_error_degradation")
