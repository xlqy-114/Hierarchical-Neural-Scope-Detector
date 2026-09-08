#!/usr/bin/env python3
"""Build UniAD2 v4.3.1 extended feature contract from v4.3.0 scan outputs.

The contract is a reviewed, explicit table of usable online feature sources. It does not
use keyword-only leakage filtering. It derives source rows from the scan hook summary and
classifies them into strict stable activation features and shape-variable scalar stats.
"""
import argparse, csv, json, math, os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

VERSION='uniad2_v4_3_1_feature_contract'
DEFAULT_SCAN_ROOT=Path('/root/autodl-tmp/UniAD2/outputs/feature_source_scan_v4_3_0')
DEFAULT_OUT_ROOT=Path('/root/autodl-tmp/UniAD2/outputs/feature_contract_v4_3_1')

def json_safe(x):
    try:
        import numpy as np
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, np.floating):
            x=float(x)
    except Exception:
        pass
    if isinstance(x, Path): return str(x)
    if isinstance(x, float):
        if math.isnan(x): return 'nan'
        if math.isinf(x): return 'inf' if x>0 else '-inf'
    if isinstance(x, dict): return {str(k): json_safe(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [json_safe(v) for v in x]
    return x

def write_json(path:Path,obj:Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2, allow_nan=False)

def write_csv(path:Path, rows:List[Dict[str,Any]], fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys=[]; seen=set()
        for r in rows:
            for k in r:
                if k not in seen:
                    keys.append(k); seen.add(k)
        fieldnames=keys
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: json_safe(r.get(k,'')) for k in fieldnames})

def latest_scan_dir(root:Path)->Path:
    cands=[]
    for p in root.glob('*'):
        if p.is_dir() and (p/'uniad2_hook_call_summary_v4_3_0.csv').exists():
            cands.append(p)
    if not cands:
        raise FileNotFoundError(f'No v4.3.0 scan dirs under {root}')
    return sorted(cands, key=lambda p:p.stat().st_mtime)[-1]

def read_csv(path:Path):
    with path.open('r',encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))

def sanitize_name(s:str)->str:
    out=[]
    for ch in str(s): out.append(ch.lower() if ch.isalnum() else '_')
    text=''.join(out)
    while '__' in text: text=text.replace('__','_')
    return text.strip('_')[:180] or 'unknown'

def fixed_group(name:str, cls:str='', fallback:str=''):
    n=str(name or '').lower(); c=str(cls or '').lower()
    if n.startswith('motion_head') or '.motion_head' in n or 'motionformer' in n: return 'U5_motion_head'
    if n.startswith('memory_bank') or '.memory_bank' in n or 'temporal' in n or 'memory' in n: return 'U9_online_track_memory'
    if 'planning' in n or 'plan' in n or 'sdc' in n: return 'U7_planning_head'
    if 'occ' in n or 'occupancy' in n: return 'U6_occ_head'
    if 'map' in n or 'seg' in n: return 'U4_map_seg_head'
    if 'track' in n or 'det' in n or 'bbox' in n: return 'U3_track_det_head'
    if 'bev' in n or (('encoder' in n) and 'pts_bbox_head' in n): return 'U2_bev_encoder'
    if 'img_backbone' in n or 'img_neck' in n or 'backbone' in n or 'neck' in n: return 'U1_img_backbone_neck'
    if fallback: return str(fallback)
    if 'transformer' in n or 'decoder' in n or 'encoder' in c: return 'U8_transformer_generic'
    return 'U0_other'

def fixed_role(name:str, cls:str='', fallback:str=''):
    n=str(name or '').lower(); c=str(cls or '').lower()
    if 'memory' in n or 'temporal' in n: return 'temporal_memory'
    if 'planning' in n or 'sdc' in n: return 'planning_head'
    if 'attention' in n or 'attn' in n or 'multiheadattention' in c: return 'attention'
    if 'ffn' in n or 'feedforward' in n: return 'ffn'
    if 'decoder' in n: return 'decoder'
    if 'encoder' in n: return 'encoder'
    if 'norm' in c or 'norm' in n: return 'normalization'
    if 'linear' in c or 'conv' in c: return 'projection_or_conv'
    if fallback: return str(fallback)
    return 'generic'

STAT_NAMES=['available','called','call_count','finite_ratio','nonfinite_count_sample','absmax_sample','mean_abs_sample','std_sample','q90_abs_sample','q95_abs_sample','q99_abs_sample','zero_ratio_sample','large_1e3_ratio_sample','large_1e4_ratio_sample','group_source','role_source']

def feature_names(module_name, group, role):
    sid=sanitize_name(module_name); g=sanitize_name(group); r=sanitize_name(role)
    base=f'feature__act_hook__{sid}'
    out=[]
    for stat in STAT_NAMES:
        if stat=='group_source': out.append((stat, f'feature__act_group_source__{g}__from__{sid}__absmax_sample'))
        elif stat=='role_source': out.append((stat, f'feature__act_role_source__{r}__from__{sid}__absmax_sample'))
        else: out.append((stat, f'{base}__{stat}'))
    return out

def as_int(x, default=0):
    try: return int(float(x))
    except Exception: return default

def as_float(x, default=0.0):
    try:
        v=float(x); return v if math.isfinite(v) else default
    except Exception: return default

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scan-dir', type=Path, default=None)
    ap.add_argument('--scan-root', type=Path, default=DEFAULT_SCAN_ROOT)
    ap.add_argument('--out-root', type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument('--availability-threshold', type=float, default=0.80)
    ap.add_argument('--include-shape-variable-u5-u9', action='store_true', default=True)
    ap.add_argument('--tag', type=str, default='latest')
    args=ap.parse_args()
    scan_dir=args.scan_dir or latest_scan_dir(args.scan_root)
    hook_path=scan_dir/'uniad2_hook_call_summary_v4_3_0.csv'
    rows=read_csv(hook_path)
    selected=[]; manifest=[]; blocked=[]
    for r in rows:
        name=r.get('module_name',''); cls=r.get('module_class','')
        fg=fixed_group(name, cls, r.get('inferred_group',''))
        fr=fixed_role(name, cls, r.get('inferred_role',''))
        online=as_float(r.get('availability_ratio',0.0)) >= args.availability_threshold
        tensor_ok=as_int(r.get('tensor_ok_for_numeric_feature',0)) == 1
        stable=as_int(r.get('shape_stable_enough',0)) == 1
        recommended=as_int(r.get('recommended_source',0)) == 1
        shape_variable_scalar = bool(online and tensor_ok and (not stable) and fg in {'U5_motion_head','U9_online_track_memory'})
        allow = bool(recommended or (args.include_shape_variable_u5_u9 and shape_variable_scalar))
        source_class = 'strict_activation_stable' if recommended else ('shape_variable_scalar_stats' if shape_variable_scalar else 'blocked_or_review')
        source_row={
            'module_name': name, 'module_class': cls,
            'scan_inferred_group': r.get('inferred_group',''), 'scan_inferred_role': r.get('inferred_role',''),
            'fixed_group': fg, 'fixed_role': fr,
            'availability_ratio': r.get('availability_ratio',''),
            'shape_unique_count': r.get('shape_unique_count',''), 'shape': r.get('shape',''), 'dtype': r.get('dtype',''),
            'numel': r.get('numel',''), 'candidate_reason': r.get('candidate_reason',''),
            'source_class': source_class,
            'online_available': int(online), 'tensor_ok_for_numeric_feature': int(tensor_ok), 'shape_stable_enough': int(stable),
            'selected_for_extended_logger': int(allow),
        }
        if allow:
            selected.append(source_row)
        else:
            blocked.append(source_row)
        for stat, fname in feature_names(name, fg, fr):
            manifest.append({
                **source_row,
                'feature_name': fname,
                'stat_name': stat,
                'prefix': 'feature__',
                'source_branch': 'detector',
                'collection_stage': 'pre_repair_or_repair_independent_forward_hook',
                'uses_clean_reference': 0,
                'uses_fault_metadata': 0,
                'uses_repair_internal': 0,
                'uses_gate_decision': 0,
                'uses_label_or_delta': 0,
                'allowed_activation_only': int(allow),
                'allowed_post_repair_output': int(allow),
                'allowed_for_training_y': 0,
                'decision': 'allow' if allow else 'needs_review',
                'reason': 'online_activation_hook_output' if recommended else ('shape_variable_scalar_stats_u5_u9' if shape_variable_scalar else 'not_selected_by_contract'),
                'review_status': 'v4_3_1_auto_contract',
            })
    allowed_cols=sorted({m['feature_name'] for m in manifest if int(m['allowed_activation_only'])==1})
    out_dir=args.out_root/args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary={
        'version': VERSION,
        'scan_dir': str(scan_dir),
        'hook_summary': str(hook_path),
        'total_scan_sources': len(rows),
        'selected_source_count': len(selected),
        'strict_stable_source_count': sum(1 for r in selected if r['source_class']=='strict_activation_stable'),
        'shape_variable_scalar_source_count': sum(1 for r in selected if r['source_class']=='shape_variable_scalar_stats'),
        'manifest_feature_count': len(manifest),
        'allowed_activation_only_feature_count': len(allowed_cols),
        'selected_group_counts': dict(Counter(r['fixed_group'] for r in selected)),
        'selected_role_counts': dict(Counter(r['fixed_role'] for r in selected)),
        'notes': [
            'Feature sources are selected source-by-source from v4.3.0 scan, not by keyword-only filtering.',
            'motion_head.* is forced to U5_motion_head; memory_bank/temporal is forced to U9_online_track_memory.',
            'shape-variable U5/U9 tensors are allowed only as fixed scalar statistics.',
            'No clean reference, fault metadata, repair event, gate decision, or label field is allowed as X.',
        ],
    }
    contract={**summary, 'selected_sources': selected, 'allowed_activation_only_columns': allowed_cols}
    write_json(out_dir/'feature_contract_v4_3_1.json', contract)
    write_csv(out_dir/'feature_source_manifest_v4_3_1.csv', manifest)
    write_csv(out_dir/'selected_feature_sources_v4_3_1.csv', selected)
    write_csv(out_dir/'blocked_or_review_feature_sources_v4_3_1.csv', blocked)
    write_json(out_dir/'allowed_activation_only_columns_v4_3_1.json', allowed_cols)
    write_json(out_dir/'feature_contract_summary_v4_3_1.json', summary)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), flush=True)
    print('OUT_DIR=', out_dir, flush=True)

if __name__=='__main__':
    main()
