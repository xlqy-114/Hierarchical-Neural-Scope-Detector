#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def load_clean_ids(path:Path):
    out=[]
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            sid=str(r.get('sample_id','')).strip()
            if not sid: raise SystemExit('Clean metric row missing sample_id')
            out.append(sid)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--repo-root',required=True); args=ap.parse_args()
    root=Path(args.root); repo=Path(args.repo_root); ext=root/'work/inputs_extracted'
    subset_path=repo/'output/OpenDriveVLA-0.5B/subsets/dev_stress_1200/subset_ids.json'
    clean_metrics=ext/'clean_reference_official6019/per_sample_planning_metrics.jsonl'
    subset=json.loads(subset_path.read_text(encoding='utf-8'))
    if not isinstance(subset,list) or len(subset)!=1200 or len(set(map(str,subset)))!=1200: raise SystemExit('Original subset contract failed')
    full_ids=load_clean_ids(clean_metrics); used=set(str(x) for x in subset)
    if len(full_ids)!=6019 or len(set(full_ids))!=6019: raise SystemExit('Full clean reference contract failed')
    miss=sorted(used-set(full_ids))
    if miss: raise SystemExit(f'Historical subset contains IDs absent from full reference: {miss[:10]}')
    unused=[x for x in full_ids if x not in used]
    if len(unused)!=4819: raise SystemExit(f'Expected 4819 unused samples, got {len(unused)}')
    out=root/'subsets'; out.mkdir(parents=True,exist_ok=True)
    p4819=out/'reviewer_R1C1_unused4819_ids.json'; p4819.write_text(json.dumps(unused,ensure_ascii=False,indent=2),encoding='utf-8')
    p12=out/'reviewer_R1C1_smoke12_ids.json'; p12.write_text(json.dumps(unused[:12],ensure_ascii=False,indent=2),encoding='utf-8')
    audit={'version':'R1C1_unused4819_v1_1','official_full_count':6019,'historical_subset_count':1200,'unused_count':4819,'overlap_count':len(set(unused)&used),'selection_rule':'all sample_id values in frozen full-6019 clean per-sample metrics absent from historical subset1200; ordering inherited from full reference; no labels/scores/fault outcomes used','unused_sha256':sha256(p4819),'smoke12_sha256':sha256(p12)}
    if audit['overlap_count']!=0: raise SystemExit('Unexpected overlap between held-out and historical sample sets')
    (out/'subset_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
