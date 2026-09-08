#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parents[1]
TRAINER = HERE / "tools_uniad" / "train_all_eligible_c1score_c2_weight_sweep_v4_4_9_2.py"

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r)+"\n")

def make_seed(root, seed, n, feature_names):
    d=root/f"all_prefix_6019_seed{seed}_synthetic_extv431"; d.mkdir(parents=True)
    rng=np.random.default_rng(seed); rows=[]; events=[]; debug=[]
    for i in range(n):
        nf=int(i%17==0); validity=int((not nf) and i%29==0); unsafe=int((not nf) and (i%11==0 or validity))
        feats={name:float(rng.normal()) for name in feature_names}; feats["feature__act_nonfinite_count_sample_sum"]=float(nf)
        feats["feature__synthetic_risk_signal"]=float(2.5*unsafe+rng.normal(scale=.7))
        row={"sample_i":i,"sample_id":f"{seed}_{i}","label__post_repair_unsafe_1p0_or_gate":unsafe,
             "label__repaired_output_illegal_or_missing":validity,"label__gate_reject":validity,
             "label__gate_accept":int(not validity),"label__repaired_output_nonfinite":int(nf and i%2==0),
             "label__repaired_severe_l2max_gt_1p0_or_illegal":unsafe,**feats}; rows.append(row)
        eff=int(i%5!=0); debug.append({"sample_i":i,"sample_id":row["sample_id"],"debug__target_present":int(i%5!=1),"debug__fault_applied_count":eff,"debug__fault_effective_count":eff})
        if i%7==0:
            events.append({"sample_i":i,"sample_id":row["sample_id"],"trace_order":i+1,"module_name":"planning_head.attn","module_class":"MultiheadAttention","uniad_group":"U7_planning_head","module_aware_role":"attention","repair_triggered":1,"repair_modified":1,"nonfinite_count_sample":nf,"huge_count_sample":int(not nf),"abnormal_count_sample":1,"abnormal_ratio_sample":.01,"pre_repair_finite_ratio_sample":.99,"pre_repair_absmax_finite_sample":100.,"post_repair_finite_ratio_full":1.,"post_repair_nonfinite_count_full":0,"post_repair_abnormal_count_full":0,"post_repair_abnormal_ratio_full":0.,"replace_output_ok":1,"repair_reason":"finite_extremes_clip+median","repair_method":"hybrid","repair_idx":1})
    pd.DataFrame(rows).to_csv(d/"detector_train_activation_only_v4_3_1.csv",index=False)
    write_jsonl(d/"debug_repair_events_v4_3_1.jsonl",events); write_jsonl(d/"debug_metadata_v4_3_1.jsonl",debug)

def test_contract():
    spec=importlib.util.spec_from_file_location("v4492_contract",TRAINER); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)
    activation=[m.INTERNAL_NF_COL]+[f"feature__act_contract_{i}" for i in range(5448)]; repair=[f"feature__repair_contract_{i}" for i in range(202)]
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"c.json"; p.write_text(json.dumps({"features":activation+repair})); cols,a=m.resolve_activation_contract(p,5449,4780,669,202,"feature__repair_")
    assert cols==activation and a["combined_source_feature_count"]==5651
    print("CONTRACT_5651_SPLIT_TEST_PASS")

def main():
    test_contract()
    with tempfile.TemporaryDirectory(prefix="v4492_test_") as td:
        root=Path(td); data=root/"data"; data.mkdir(); feats=["feature__act_nonfinite_count_sample_sum","feature__synthetic_risk_signal"]+[f"feature__synthetic_{i}" for i in range(18)]
        contract=root/"contract.json"; legacy=["feature__repair_legacy_a","feature__repair_legacy_b","feature__repair_legacy_c"]; contract.write_text(json.dumps({"features":feats+legacy}))
        for seed in [1,2,3,4,5]: make_seed(data,seed,120,feats)
        out=root/"out"; cmd=[sys.executable,str(TRAINER),"--data-root",str(data),"--frozen-contract-json",str(contract),"--out-dir",str(out),"--train-seeds","1,2,3","--val-seeds","4","--test-seeds","5","--expected-activation-features","20","--expected-reviewed-features","12","--expected-runtime-features","8","--expected-legacy-repair-features","3","--c2-positive-weights","1,2,5","--c2-hard-negative-count","30","--c2-random-negative-count","30","--c1-max-iter","25","--c2-max-iter","20","--max-leaf-nodes","15","--min-samples-leaf","3"]
        subprocess.run(cmd,check=True)
        pop=pd.read_csv(out/"population_summary_v4_4_9_2.csv"); assert (pop.internal_nonfinite>0).all()
        preds=pd.read_csv(out/"fixed_split_test_predictions_v4_4_9_2.csv"); assert preds[(preds.internal_nonfinite_seen>0)&(preds.learned_eligible>0)].empty
        nf=preds[preds.internal_nonfinite_seen>0]; alarm=[c for c in preds if c.startswith("alarm__")]; assert alarm and (nf[alarm].min(axis=1)==1).all()
        assert not any(c.startswith("c1_candidate") for c in preds.columns)
        assert np.array_equal(preds["c2_scored"].to_numpy(dtype=np.int8), preds["learned_eligible"].to_numpy(dtype=np.int8))
        score_cols=[c for c in preds if c.startswith("score__c2_pw")]; assert len(score_cols)==3
        subset=json.loads((out/"c2_training_subset_audit_v4_4_9_2.json").read_text()); assert subset["selected_positive"]>0 and subset["selected_negative"]>0
        metrics=pd.read_csv(out/"fixed_split_test_metrics_v4_4_9_2.csv"); assert "selected_model_variant" in metrics and len(metrics)>=10
        manifest=json.loads((out/"model_manifest_v4_4_9_2.json").read_text()); assert manifest["no_hard_gate"] is True and len(manifest["c2_models"])==3
        cfg=json.loads((out/"run_config_v4_4_9_2.json").read_text()); assert cfg["no_oof"] is True and cfg["test_used_for_selection"] is False
        print("ALL_ELIGIBLE_NO_HARD_GATE_TEST_PASS")
        print("WEIGHT_SWEEP_VALIDATION_SELECTION_TEST_PASS")
        print("SYNTHETIC_PIPELINE_TEST_PASS")
if __name__=="__main__": main()
