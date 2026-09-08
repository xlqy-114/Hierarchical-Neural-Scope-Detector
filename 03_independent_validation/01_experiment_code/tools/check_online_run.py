#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,gzip,csv,math
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--expected-rows',type=int,required=True); ap.add_argument('--policy-dir',required=True); ap.add_argument('--output-dir',required=True); args=ap.parse_args()
    run=Path(args.run_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    p=run/'results/v53_inline_detector_records.jsonl'
    if not p.is_file(): raise SystemExit(f'Missing inline records: {p}')
    records=[]
    with p.open('r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): records.append(json.loads(line))
    if len(records)!=args.expected_rows: raise SystemExit(f'Expected {args.expected_rows} online rows, got {len(records)}')
    sids=[str(r.get('sample_id','')) for r in records]
    if len(set(sids))!=len(sids): raise SystemExit('Duplicate sample IDs in online records')
    policy=json.loads((Path(args.policy_dir)/'frozen_policy.json').read_text(encoding='utf-8'))
    fcols=json.loads((Path(args.policy_dir)/'old_exact_2605_features.json').read_text(encoding='utf-8'))
    compact=[]
    # Write exact 2605 online values as gzip CSV for independent replay.
    gzpath=out/'r1c1_table5_online_feature_matrix_2605.csv.gz'
    with gzip.open(gzpath,'wt',encoding='utf-8',newline='') as gf:
        w=csv.writer(gf); w.writerow(['sample_id','row_source']+fcols)
        for r in records:
            t=r.get('r1c1_table5_frozen_policy') or {}
            if not t: raise SystemExit(f'Missing frozen Table5 online decision for sample {r.get("sample_id")}')
            if str(t.get('model_sha256'))!=str(policy.get('model_sha256')): raise SystemExit('Online model hash differs from frozen policy')
            flat=r.get('deployable_feature_row') or {}
            vals=[flat.get(c,'') for c in fcols]
            w.writerow([r.get('sample_id',''),r.get('row_source','')]+vals)
            a=r.get('analysis_only') or {}; faults=a.get('faults') or []
            f0=faults[0] if faults and isinstance(faults[0],dict) else {}
            compact.append({
              'sample_id':r.get('sample_id',''),'row_source':r.get('row_source',''),
              'injected_effective':int(a.get('is_positive_effective',0) or 0),'exact_layer':a.get('exact_layer',''),
              'fault_target_exact':a.get('fault_target_exact',''),'fault_op':f0.get('bit_op',f0.get('op','')),'bit_count':f0.get('bit_count',''),
              'feature_model_score':t.get('feature_model_score'),'feature_model_threshold':t.get('feature_model_threshold'),'feature_model_alarm':t.get('feature_model_alarm'),
              'base_alarm':t.get('base_alarm'),'old_hgb_score':t.get('old_hgb_score'),'speed_hgb_score':t.get('speed_hgb_score'),'hgbmax_score':t.get('hgbmax_score'),
              'veto':t.get('veto'),'pre_veto_alarm':t.get('pre_veto_alarm'),'final_alarm':t.get('final_alarm'),
              'feature_absent_value_count':t.get('feature_absent_value_count'),'feature_nonfinite_value_count':t.get('feature_nonfinite_value_count'),
            })
    cdf=pd.DataFrame(compact); cdf.to_csv(out/'r1c1_online_policy_compact.csv',index=False)
    summary={'pass':True,'rows':len(records),'unique_samples':len(set(sids)),'final_alarm_count':int(pd.to_numeric(cdf.final_alarm).sum()),'feature_alarm_count':int(pd.to_numeric(cdf.feature_model_alarm).sum()),'base_alarm_count':int(pd.to_numeric(cdf.base_alarm).sum()),'veto_count':int(pd.to_numeric(cdf.veto).sum()),'effective_injection_count':int(cdf.injected_effective.sum()),'feature_matrix_gz':str(gzpath)}
    (out/'r1c1_online_run_check.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
