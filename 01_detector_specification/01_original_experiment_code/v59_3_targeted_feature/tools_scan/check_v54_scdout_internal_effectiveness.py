#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict effectiveness check for v49 unified S+C+D/D_out internal neural run."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict

LEGAL_SCOPES = {"S0_only", "S1_only", "C_only", "D_only", "no_fault"}
PREFIX_BY_SCOPE = {"S0_only":"can_bus", "S1_only":"s1:", "C_only":"c:", "D_only":"d:"}

def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def check_seed(seed_dir: Path, expected_rows: int, strict: bool):
    rec_path = seed_dir/'results'/'detector_records.jsonl'
    plan_path = seed_dir/'results'/'fault_plan.json'
    out = {
        'seed_dir': str(seed_dir), 'exists': rec_path.exists(), 'fault_plan_exists': plan_path.exists(),
        'rows': 0, 'pass': 0, 'bad_count': 0, 'bad_examples': [],
        'scope_counts': {}, 'spec_count_distribution': {},
        'attempted_total': 0, 'applied_total': 0, 'effective_total': 0,
        'by_scope': {}, 'by_target_prefix': {}, 'reason_counts': {},
    }
    if not rec_path.exists():
        out['bad_examples'].append({'reason':'missing_detector_records', 'path': str(rec_path)})
        out['bad_count']=1
        return out
    scope_counts=Counter(); spec_dist=Counter(); by_scope=defaultdict(lambda: Counter({'rows':0,'spec':0,'attempted':0,'applied':0,'effective':0,'bad':0}))
    pref_counts=Counter(); reason_counts=Counter(); bad=[]
    for r in read_jsonl(rec_path):
        out['rows'] += 1
        scope=str(r.get('fault_scope','no_fault'))
        scope_counts[scope]+=1
        faults=r.get('faults') or []
        spec=int(r.get('fault_spec_count', len(faults)) or 0)
        attempted=int(r.get('fault_attempted_count', len(faults)) or 0)
        applied=int(r.get('fault_applied_count',0) or 0)
        effective=int(r.get('fault_effective_count',0) or 0)
        is_pos=int(r.get('is_positive',0) or 0)
        spec_dist[str(spec)]+=1
        out['attempted_total']+=attempted; out['applied_total']+=applied; out['effective_total']+=effective
        bs=by_scope[scope]; bs['rows']+=1; bs['spec']+=spec; bs['attempted']+=attempted; bs['applied']+=applied; bs['effective']+=effective
        for f in faults:
            tgt=str(f.get('target',''))
            if tgt == 'can_bus': pref='can_bus'
            elif ':' in tgt: pref=tgt.split(':',1)[0]+':'
            else: pref=tgt or 'unknown'
            pref_counts[pref]+=1
            reason_counts[str(f.get('reason',''))]+=1
        if strict:
            if scope not in LEGAL_SCOPES:
                bad.append({'sample_id': r.get('sample_id'), 'reason':'illegal_scope', 'scope':scope})
            if scope == 'no_fault':
                if is_pos != 0 or spec != 0 or attempted != 0 or applied != 0 or effective != 0:
                    bad.append({'sample_id': r.get('sample_id'), 'reason':'no_fault_has_fault_counts', 'scope':scope, 'is_positive':is_pos, 'spec':spec, 'attempted':attempted, 'applied':applied, 'effective':effective})
            else:
                if is_pos != 1 or spec != 1 or attempted != 1 or applied != 1 or effective != 1:
                    bad.append({'sample_id': r.get('sample_id'), 'reason':'positive_not_one_effective_fault', 'scope':scope, 'is_positive':is_pos, 'spec':spec, 'attempted':attempted, 'applied':applied, 'effective':effective, 'faults':faults[:2]})
                expected=PREFIX_BY_SCOPE.get(scope)
                if expected:
                    if expected == 'can_bus': ok=any(str(f.get('target',''))=='can_bus' for f in faults)
                    else: ok=any(str(f.get('target','')).startswith(expected) for f in faults)
                    if not ok:
                        bad.append({'sample_id': r.get('sample_id'), 'reason':'target_prefix_mismatch', 'scope':scope, 'expected_prefix':expected, 'faults':faults[:2]})
    if expected_rows and out['rows'] != expected_rows:
        bad.append({'reason':'row_count_mismatch','expected_rows':expected_rows,'observed_rows':out['rows']})
    out['scope_counts']=dict(scope_counts)
    out['spec_count_distribution']=dict(spec_dist)
    out['by_scope']={k:dict(v) for k,v in by_scope.items()}
    out['by_target_prefix']=dict(pref_counts)
    out['reason_counts']=dict(reason_counts)
    out['bad_count']=len(bad)
    out['bad_examples']=bad[:20]
    out['pass']=int(len(bad)==0)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--campaign-dir', required=True)
    ap.add_argument('--seed-root', default='A1_internal_neural_scdout_singlefault')
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--expected-rows', type=int, default=1200)
    ap.add_argument('--output-json', required=True)
    ap.add_argument('--strict', action='store_true')
    args=ap.parse_args()
    campaign=Path(args.campaign_dir)
    seeds=[s.strip().replace('seed','') for s in args.seeds.split(',') if s.strip()]
    per=[check_seed(campaign/args.seed_root/f'seed{s}', args.expected_rows, args.strict) for s in seeds]
    summary={
        'version':'v49_scd_internal_effectiveness_check',
        'campaign_dir':str(campaign),'seed_root':args.seed_root,'seeds':seeds,
        'expected_rows':args.expected_rows,'strict':int(args.strict),'seed_count':len(seeds),
        'all_finished':int(all(x.get('exists') for x in per)),
        'all_pass':int(all(int(x.get('pass',0))==1 for x in per)),
        'total_rows':sum(int(x.get('rows',0)) for x in per),
        'total_bad_count':sum(int(x.get('bad_count',0)) for x in per),
        'aggregate_scope_counts':dict(sum((Counter(x.get('scope_counts',{})) for x in per), Counter())),
        'aggregate_target_prefix_counts':dict(sum((Counter(x.get('by_target_prefix',{})) for x in per), Counter())),
        'aggregate_reason_counts':dict(sum((Counter(x.get('reason_counts',{})) for x in per), Counter())),
        'per_seed':per,
    }
    out=Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.strict and not summary['all_pass']:
        raise SystemExit(2)
if __name__=='__main__': main()
