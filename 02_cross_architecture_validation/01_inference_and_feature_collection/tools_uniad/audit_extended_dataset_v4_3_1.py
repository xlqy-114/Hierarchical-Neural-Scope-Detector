#!/usr/bin/env python3
"""Audit UniAD2 v4.3.1 extended feature fault+repair detector dataset outputs."""
import argparse, csv, json, math
from collections import Counter
from pathlib import Path

FORBIDDEN = [
    'repair_', 'gate_', 'label_', 'target_', 'fault_', 'bit_', 'clean_', 'delta_',
    'harmful', 'severe', 'applied', 'effective', 'event_id',
    'high_contam', 'module_aware', 'replacement', 'before', 'after',
]
# Note: module path tokens inside feature__act_hook__... are expected and are not leakage.


def read_json(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)

def count_jsonl(path):
    p=Path(path)
    if not p.exists(): return 0
    n=0
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): n+=1
    return n

def newest_out_dir(root: Path):
    dirs=[p for p in root.iterdir() if p.is_dir()]
    if not dirs: raise FileNotFoundError(f'No output dirs under {root}')
    return max(dirs, key=lambda p:p.stat().st_mtime)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=None)
    ap.add_argument('--out-root', type=Path, default=Path('/root/autodl-tmp/UniAD2/outputs/direct_fault_repair_extended_features_v4_3_1'))
    args=ap.parse_args()
    out_dir=args.out_dir or newest_out_dir(args.out_root)
    summary_path=out_dir/'fault_repair_extended_feature_dataset_summary_v4_3_1.json'
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing summary: {summary_path}')
    summary=read_json(summary_path)
    files=summary.get('files', {})
    print('OUT_DIR=', out_dir)
    print('SELECTED_DONE=', summary.get('counters',{}).get('selected_cases_done'))
    print('PREFIX_PROCESSED=', summary.get('counters',{}).get('prefix_samples_processed'))
    print('REPAIRED_NONFINITE=', summary.get('counters',{}).get('repaired_output_nonfinite'))
    print('GATE_REJECT=', summary.get('counters',{}).get('gate_reject_count'))
    for key in ['detector_activation_features','detector_output_features','repair_outcome_labels','debug_metadata']:
        p=Path(files.get(key,''))
        print(f'{key}_lines=', count_jsonl(p), 'path=', p)
    audit=summary.get('leakage_audit', {})
    print('FEATURE_KEY_COUNT=', audit.get('feature_key_count'))
    print('FORBIDDEN_HIT_COUNT=', audit.get('forbidden_hit_count'))
    if int(audit.get('forbidden_hit_count',0) or 0):
        print('FORBIDDEN_HITS_HEAD=', audit.get('forbidden_hits', [])[:20])
    train=Path(files.get('detector_train_activation_only_csv',''))
    if train.exists():
        with train.open('r', encoding='utf-8', newline='') as f:
            rdr=csv.DictReader(f)
            labels=Counter()
            rows=0
            feat_bad=[]
            for row in rdr:
                rows+=1
                for k,v in row.items():
                    if k.startswith('label__'):
                        if str(v).strip() in {'1','1.0'}:
                            labels[k]+=1
                    if k.startswith('feature__'):
                        lk=k.lower()
                        if any(bad in lk for bad in FORBIDDEN):
                            feat_bad.append(k)
            print('TRAIN_ACTIVATION_ROWS=', rows)
            print('LABEL_POSITIVE_COUNTS=', dict(labels.most_common(20)))
            print('TRAIN_FEATURE_FORBIDDEN_KEYS=', sorted(set(feat_bad))[:50])
    print('AUDIT_DONE')

if __name__ == '__main__':
    main()
