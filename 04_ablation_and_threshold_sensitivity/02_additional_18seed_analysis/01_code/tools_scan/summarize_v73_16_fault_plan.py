#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--campaign-dir', type=Path, required=True)
    ap.add_argument('--seed-root', required=True)
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--sample-count', type=int, required=True)
    ap.add_argument('--output-json', type=Path, required=True)
    args=ap.parse_args()
    rows=[]
    for seed in [s.strip().replace('seed','') for s in args.seeds.split(',') if s.strip()]:
        p=args.campaign_dir/args.seed_root/f'seed{seed}'/'results'/'fault_plan.json'
        if not p.is_file():
            rows.append({'seed':seed,'fault_plan_path':str(p),'available':False})
            continue
        d=json.loads(p.read_text(encoding='utf-8'))
        pos=int(d.get('positive_count', len(d.get('positive_sample_ids', []) or [])) or 0)
        rows.append({
            'seed':seed,
            'fault_plan_path':str(p),
            'available':True,
            'sample_count':int(args.sample_count),
            'injected_sample_count':pos,
            'no_fault_count':int(args.sample_count)-pos,
            'actual_injection_ratio':float(pos)/float(args.sample_count) if args.sample_count else None,
            'positive_min':d.get('positive_min'),
            'positive_max':d.get('positive_max'),
            'target_mode':d.get('target_mode'),
        })
    out={'version':'v73_16_fault_plan_manifest','campaign_dir':str(args.campaign_dir),'seed_root':args.seed_root,'sample_count':int(args.sample_count),'seeds':[r['seed'] for r in rows],'seed_plans':rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(out, indent=2, ensure_ascii=False))
if __name__=='__main__':
    main()
