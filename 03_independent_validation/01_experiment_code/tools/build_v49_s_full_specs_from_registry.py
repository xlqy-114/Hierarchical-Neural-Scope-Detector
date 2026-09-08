#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build v49 S Full-S target specs/weights from v36/v38 full-leaf registry.

Output is compatible with inference_drivevla_subset1200_cde_v44_ood_singlefault.py,
which reuses the historical `s1:` activation hook runtime for arbitrary S-module
module suffixes.  Although the event prefix is still `s1:`, the logical target
list here covers S1-S5 full-leaf targets from the v36/v38 registry.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List

CANONICAL_LAYERS = [
    "S1_backbone", "S1_neck", "S2_bev_front", "S2_can_bus_mlp", "S2_encoder_l0",
    "S3_encoder_l1_l5", "S4_decoder", "S4_track_head", "S4_seg_map_head", "S5_scene_head_misc",
]

def read_registry(path: Path) -> List[Dict[str, Any]]:
    rows=[]
    with path.open('r', encoding='utf-8') as f:
        reader=csv.DictReader(f)
        for r in reader:
            if 'pass' in r and str(r.get('pass','1')).strip().lower() not in {'1','true','yes'}:
                continue
            logical=str(r.get('logical_name') or r.get('target_name') or r.get('name') or '').strip()
            layer=str(r.get('s_layer') or r.get('fault_s_layer') or r.get('layer') or '').strip()
            suffix=str(r.get('module_name') or r.get('suffix') or r.get('module') or '').strip()
            cls=str(r.get('module_class') or r.get('class') or 'unknown').strip()
            if not logical or not layer or not suffix:
                continue
            rows.append({'logical_name':logical, 's_layer':layer, 'suffix':suffix, 'module_class':cls})
    if not rows:
        raise RuntimeError(f'No usable S full-leaf targets found in {path}')
    # unique logical names, preserve layer/suffix first occurrence
    seen=set(); out=[]
    for r in rows:
        if r['logical_name'] in seen: continue
        seen.add(r['logical_name']); out.append(r)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--registry-csv', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--layer-balance-scale', type=int, default=1000)
    ap.add_argument('--exclude-logical-substrings', default='query_interact',
                    help='Comma-separated logical-name substrings to exclude from formal target pool. Default removes conditional S4 query_interact targets that may be planned but not runtime-hit.')
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows=read_registry(Path(args.registry_csv))
    excludes=[x.strip() for x in str(args.exclude_logical_substrings or '').split(',') if x.strip()]
    before=len(rows)
    if excludes:
        rows=[r for r in rows if not any(s in r['logical_name'] for s in excludes)]
    if not rows:
        raise RuntimeError('All S targets were excluded; check --exclude-logical-substrings')
    excluded_count=before-len(rows)
    rows.sort(key=lambda r: (r['s_layer'], r['logical_name']))
    layer_counts=Counter(r['s_layer'] for r in rows)
    weights={}
    for r in rows:
        c=max(1, int(layer_counts[r['s_layer']]))
        # Approximate uniform-by-layer inside the S leaf scope.  Each layer has
        # similar total weight; individual target weight is inverse to count.
        weights[r['logical_name']]=max(1, int(round(float(args.layer_balance_scale)/float(c))))
    specs=','.join([f"{r['logical_name']}={r['suffix']}" for r in rows])
    weight_text=','.join([f"{r['logical_name']}:{weights[r['logical_name']]}" for r in rows])
    (out/'v49_s_full_target_specs.txt').write_text(specs, encoding='utf-8')
    (out/'v49_s_full_target_weights_layer_balanced.txt').write_text(weight_text, encoding='utf-8')
    first=rows[0]['logical_name']
    (out/'v49_s_primary_target_name.txt').write_text(first, encoding='utf-8')
    with (out/'v49_s_full_target_map.csv').open('w', newline='', encoding='utf-8') as f:
        fieldnames=['logical_name','s_layer','suffix','module_class','sampling_weight']
        w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows:
            rr=dict(r); rr['sampling_weight']=weights[r['logical_name']]; w.writerow(rr)
    summary={
        'version':'v49_s_full_specs_from_registry_v51_target_cleanup',
        'registry_csv':str(Path(args.registry_csv)),
        'target_count':len(rows),
        'excluded_logical_substrings': excludes,
        'excluded_count': excluded_count,
        'layer_counts':dict(layer_counts),
        'weight_policy':'approx_uniform_by_layer_with_integer_inverse_count_weights',
        'layer_balance_scale':int(args.layer_balance_scale),
        'primary_target_name':first,
        'outputs':{
            'specs_txt':str(out/'v49_s_full_target_specs.txt'),
            'weights_txt':str(out/'v49_s_full_target_weights_layer_balanced.txt'),
            'map_csv':str(out/'v49_s_full_target_map.csv'),
            'primary_target_name_txt':str(out/'v49_s_primary_target_name.txt'),
        }
    }
    (out/'v49_s_full_specs_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__=='__main__':
    main()
