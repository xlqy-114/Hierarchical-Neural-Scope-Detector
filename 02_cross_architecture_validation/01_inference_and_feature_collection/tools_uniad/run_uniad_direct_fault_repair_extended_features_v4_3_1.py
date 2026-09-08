#!/usr/bin/env python3
"""UniAD2 v4.0.1 direct fault+repair detector-dataset inference.

Purpose:
  Generate a seed-controlled fault plan and apply clean-calibrated numerical
  containment/restoration inside forward hooks. The script compares repaired outputs
  labels against the clean model in the same run, without a separate fault-only inference pass.

Important constraints:
  - The repair logic does NOT use bit position/fault metadata for detection or repair.
    Fault metadata is generated and saved for reproducibility/debug only; repair and detector features remain fault-agnostic.
  - Clean calibration thresholds are clean-only numerical contracts.
  - This repair version implements non-parametric, fault-agnostic methods plus
  module-aware high-contamination containment:
      detect_only, clip, median_impute, hybrid
    It intentionally does not use naive torch.nan_to_num as a silent repair.
"""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
HELPER_PATH = THIS_DIR / "uniad_v351_runtime_helpers.py"
spec = importlib.util.spec_from_file_location("uniad_v351_runtime_helpers", str(HELPER_PATH))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)  # type: ignore

VERSION = "uniad2_direct_fault_repair_extended_features_v4_3_1"
DEFAULT_V351_DIR = Path('/root/autodl-tmp/UniAD2/outputs/fault_inference_v3_5_1/full6019_seed958101')
DEFAULT_CAL_DIR = Path('/root/autodl-tmp/UniAD2/outputs/clean_calibration_v3_5_3/seeds_958201_958204')
DEFAULT_OUT_ROOT = Path('/root/autodl-tmp/UniAD2/outputs/direct_fault_repair_extended_features_v4_3_1')
DEFAULT_REGISTRY = Path('/root/autodl-tmp/UniAD2/outputs/uniad2_hook_registry_v3_5_1/uniad2_hook_registry_v3_5_1.csv')


def json_safe(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return 'nan'
        if math.isinf(obj):
            return 'inf' if obj > 0 else '-inf'
        return obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if torch.is_tensor(obj):
        return json_safe(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return str(obj)


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2, allow_nan=False)


def append_jsonl(path: Path, row: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: json_safe(r.get(k, '')) for k in fieldnames})


def read_jsonl_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_thresholds(cal_dir: Path, margin: float) -> Dict[str, Dict[str, Any]]:
    p = cal_dir / 'clean_hook_thresholds_all_seeds_v3_5_3.csv'
    if not p.exists():
        raise FileNotFoundError(f'Missing clean thresholds: {p}')
    target_margin = float(margin)
    out: Dict[str, Dict[str, Any]] = {}
    with p.open('r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            try:
                m = float(row.get('margin', 'nan'))
            except Exception:
                continue
            if abs(m - target_margin) > 1e-9:
                continue
            name = row['module_name']
            out[name] = {
                'hook_index': int(row.get('hook_index', -1)),
                'target_id': row.get('target_id', ''),
                'module_name': name,
                'module_class': row.get('module_class', ''),
                'uniad_group': row.get('uniad_group', ''),
                'margin': m,
                'threshold_absmax': float(row.get('threshold_absmax', 'nan')),
                'clean_absmax_max': float(row.get('clean_absmax_max_across_seeds', 'nan')),
                'clean_absmax_p9999': float(row.get('clean_absmax_p9999_max_across_seeds', 'nan')),
            }
    if not out:
        raise RuntimeError(f'No thresholds loaded for margin={margin} from {p}')
    return out


def parse_group_filter(text: str):
    s = str(text or '').strip()
    if not s or s.lower() in {'all', '*'}:
        return None
    return {x.strip() for x in s.split(',') if x.strip()}


def parse_case_ids(text: str) -> List[int]:
    ids = []
    for x in str(text or '').split(','):
        x = x.strip()
        if x:
            ids.append(int(x))
    return sorted(set(ids))


def load_original_baseline(v351_dir: Path) -> Dict[int, Dict[str, Any]]:
    raw_path = v351_dir / 'uniad2_clean_fault_raw_ingredients_v3_5_1.jsonl'
    rows = {}
    for row in read_jsonl_rows(raw_path):
        try:
            i = int(row.get('sample_i'))
        except Exception:
            continue
        rows[i] = row
    return rows


def baseline_delta_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    d = row.get('delta', {}) if isinstance(row, dict) else {}
    if not isinstance(d, dict):
        d = {}
    out = {}
    for k in ['traj_delta_available', 'traj_delta_reason', 'traj_delta_l2_mean', 'traj_delta_l2_max', 'traj_delta_l2_final', 'traj_delta_steps']:
        out[f'baseline_{k}'] = d.get(k, '')
    for k in ['output_illegal_or_missing', 'output_missing', 'output_nonfinite', 'output_shape_invalid', 'fault_applied_count', 'fault_effective_count']:
        out[f'baseline_{k}'] = row.get(k, '') if isinstance(row, dict) else ''
    # Preserve optional collision/speed fields if present in future/raw variants.
    for k, v in row.items() if isinstance(row, dict) else []:
        lk = str(k).lower()
        if any(tag in lk for tag in ['collision', 'speed', 'heading']):
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[f'baseline_{k}'] = v
    return out


def load_selected_cases(args, original_plan: Dict[int, Any], baseline_rows: Dict[int, Dict[str, Any]]) -> List[int]:
    if args.case_ids.strip():
        return parse_case_ids(args.case_ids)
    mode = args.selected_mode
    candidates = []
    if mode == 'nan_cases':
        for i, row in baseline_rows.items():
            try:
                if int(row.get('output_nonfinite', 0) or 0) == 1:
                    candidates.append(i)
            except Exception:
                pass
    elif mode == 'fault_cases':
        candidates = sorted(original_plan.keys())
    elif mode == 'all_prefix':
        candidates = list(range(0, int(args.case_limit))) if args.case_limit > 0 else list(range(0, len(baseline_rows)))
    else:
        raise ValueError(f'Unsupported selected_mode: {mode}')
    candidates = sorted(set(int(x) for x in candidates))
    if args.case_limit and int(args.case_limit) > 0:
        candidates = candidates[:int(args.case_limit)]
    return candidates



def stable_hash_int(text: str) -> int:
    return int(hashlib.blake2b(str(text).encode('utf-8'), digest_size=8).hexdigest(), 16)


def group_is_high_risk(group: str) -> bool:
    return str(group) in {'U2_bev_encoder', 'U3_track_det_head', 'U4_map_seg_head', 'U7_planning_head'}


def monitoring_budget_for_tensor(t: torch.Tensor, target: Dict[str, Any], strategy: str, high_budget: int, low_budget: int) -> int:
    if strategy == 'full':
        return -1
    group = target.get('uniad_group', '') if isinstance(target, dict) else ''
    budget = int(high_budget) if group_is_high_risk(group) else int(low_budget)
    if t is None or not torch.is_tensor(t):
        return budget
    n = int(t.numel())
    if budget <= 0 or budget >= n:
        return -1
    return max(1, budget)


def _stratified_indices(n: int, k: int, device, dtype=torch.long):
    if k >= n:
        return torch.arange(n, device=device, dtype=dtype)
    # Segment centers, deterministic and globally covering.
    return torch.floor((torch.arange(k, device=device, dtype=torch.float64) + 0.5) * (float(n) / float(k))).to(dtype=torch.long).clamp_(0, n - 1)


def _hash_stratified_indices(n: int, k: int, device, key: str, dtype=torch.long):
    if k >= n:
        return torch.arange(n, device=device, dtype=dtype)
    base = torch.floor(torch.arange(k, device=device, dtype=torch.float64) * (float(n) / float(k))).to(dtype=torch.long)
    seg = max(1, n // k)
    h = stable_hash_int(key)
    # Deterministic per-segment jitter without using fault metadata.
    step = (h % 104729) + 1
    off = (torch.arange(k, device=device, dtype=torch.long) * int(step) + int(h % seg)) % int(seg)
    return (base + off).clamp_(0, n - 1)


def _block_indices(n: int, k: int, device, key: str, block_size: int = 256):
    if k >= n:
        return torch.arange(n, device=device, dtype=torch.long)
    block_size = max(1, min(int(block_size), max(1, k)))
    block_budget = max(block_size, k // 2)
    num_blocks = max(1, min(64, block_budget // block_size))
    starts_base = _hash_stratified_indices(max(1, n - block_size + 1), num_blocks, device, key + ':blocks')
    offsets = torch.arange(block_size, device=device, dtype=torch.long)
    blocks = (starts_base[:, None] + offsets[None, :]).reshape(-1).clamp_(0, n - 1)
    remain = max(0, k - int(blocks.numel()))
    if remain > 0:
        pts = _hash_stratified_indices(n, remain, device, key + ':pts')
        idx = torch.cat([blocks, pts], dim=0)
    else:
        idx = blocks[:k]
    # Keep cost bounded; unique may reduce duplicate samples but also costs. For smoke it is acceptable.
    if idx.numel() > k:
        idx = idx[:k]
    return idx


def sampled_flat_tensor(x: torch.Tensor, max_values: int, strategy: str, key: str, module_class: str = '', uniad_group: str = '', block_size: int = 256) -> Tuple[torch.Tensor, Dict[str, Any]]:
    flat = x.reshape(-1)
    n = int(flat.numel())
    if n <= 0:
        return flat, {'sampling_strategy': strategy, 'sample_budget': int(max_values), 'sampled_full_tensor': 1, 'sample_numel_total': 0}
    if max_values is None or int(max_values) <= 0 or int(max_values) >= n or strategy == 'full':
        return flat, {'sampling_strategy': 'full' if strategy == 'full' or int(max_values) <= 0 else strategy, 'sample_budget': int(max_values), 'sampled_full_tensor': 1, 'sample_numel_total': n}
    k = max(1, min(int(max_values), n))
    device = flat.device
    strategy = str(strategy)
    if strategy == 'firstk':
        idx = torch.arange(k, device=device, dtype=torch.long)
    elif strategy == 'stratified':
        idx = _stratified_indices(n, k, device)
    elif strategy == 'hash_stratified':
        idx = _hash_stratified_indices(n, k, device, key)
    elif strategy == 'block':
        idx = _block_indices(n, k, device, key, block_size=block_size)
    elif strategy == 'typeaware':
        # Fault-agnostic but tensor-type aware: use mixed point+block coverage for large structured tensors.
        # The policy uses only hook metadata and tensor shape, not fault metadata.
        cls = str(module_class)
        group = str(uniad_group)
        if group in {'U2_bev_encoder', 'U4_map_seg_head'} and x.dim() >= 3:
            block_k = max(k // 2, min(k, 8192))
            point_k = max(1, k - block_k)
            idx = torch.cat([
                _block_indices(n, block_k, device, key + ':type_bev_block', block_size=block_size),
                _hash_stratified_indices(n, point_k, device, key + ':type_bev_point'),
            ], dim=0)
        elif any(tag in cls for tag in ['MultiheadAttention', 'Attention', 'Transformer', 'Decoder', 'FFN', 'Linear']):
            block_k = max(k // 3, min(k, 4096))
            point_k = max(1, k - block_k)
            idx = torch.cat([
                _hash_stratified_indices(n, point_k, device, key + ':type_query_point'),
                _block_indices(n, block_k, device, key + ':type_query_block', block_size=block_size),
            ], dim=0)
        else:
            idx = _hash_stratified_indices(n, k, device, key + ':type_default')
        if idx.numel() > k:
            idx = idx[:k]
    else:
        raise ValueError(f'Unsupported sampling_strategy: {strategy}')
    return flat.index_select(0, idx), {'sampling_strategy': strategy, 'sample_budget': k, 'sampled_full_tensor': 0, 'sample_numel_total': n}


def tensor_contract_stats(x: torch.Tensor, threshold: float, max_values: int, strategy: str, key: str, module_class: str = '', uniad_group: str = '', block_size: int = 256) -> Dict[str, Any]:
    if x is None or not torch.is_tensor(x) or x.numel() == 0:
        return {
            'finite_ratio': 1.0, 'absmax_finite': float('nan'), 'nonfinite_count': 0,
            'huge_count': 0, 'abnormal_count': 0, 'sample_numel': 0,
            'abnormal_ratio': 0.0, 'nonfinite_ratio': 0.0, 'huge_ratio': 0.0,
            'absmax_ratio': float('nan'), 'sample_numel_total': 0,
            'sampled_full_tensor': 1, 'sample_budget': int(max_values), 'sampling_strategy': strategy,
        }
    sampled, meta = sampled_flat_tensor(x, int(max_values), strategy, key, module_class=module_class, uniad_group=uniad_group, block_size=block_size)
    xf = sampled.float()
    sample_numel = int(xf.numel())
    finite = torch.isfinite(xf)
    nonfinite_count = int((~finite).sum().detach().cpu().item()) if sample_numel else 0
    finite_ratio = float(finite.float().mean().detach().cpu().item()) if sample_numel else 1.0
    if bool(finite.any().detach().cpu().item()):
        finite_vals = xf[finite]
        absmax = float(finite_vals.abs().max().detach().cpu().item())
        if math.isfinite(float(threshold)) and float(threshold) > 0:
            huge_count = int((finite_vals.abs() > float(threshold)).sum().detach().cpu().item())
        else:
            huge_count = 0
    else:
        absmax = float('nan')
        huge_count = 0
    abnormal_count = int(nonfinite_count + huge_count)
    abnormal_ratio = float(abnormal_count) / float(sample_numel) if sample_numel else 0.0
    nonfinite_ratio = float(nonfinite_count) / float(sample_numel) if sample_numel else 0.0
    huge_ratio = float(huge_count) / float(sample_numel) if sample_numel else 0.0
    absmax_ratio = (float(absmax) / float(threshold)) if (math.isfinite(float(absmax)) and math.isfinite(float(threshold)) and float(threshold) > 0) else float('nan')
    return {
        'finite_ratio': finite_ratio,
        'absmax_finite': absmax,
        'nonfinite_count': nonfinite_count,
        'huge_count': huge_count,
        'abnormal_count': abnormal_count,
        'sample_numel': sample_numel,
        'abnormal_ratio': abnormal_ratio,
        'nonfinite_ratio': nonfinite_ratio,
        'huge_ratio': huge_ratio,
        'absmax_ratio': absmax_ratio,
        **meta,
    }


def contamination_band(abnormal_ratio: Any, finite_ratio: Any) -> str:
    try:
        ar = float(abnormal_ratio)
    except Exception:
        ar = float('nan')
    try:
        fr = float(finite_ratio)
    except Exception:
        fr = float('nan')
    if math.isfinite(fr) and fr <= 0.5:
        return 'high'
    if math.isfinite(ar):
        if ar <= 0:
            return 'none'
        if ar < 0.001:
            return 'low'
        if ar < 0.1:
            return 'medium'
        return 'high'
    return 'unknown'


FEATURE_GROUPS = [
    'U1_img_backbone_neck', 'U2_bev_encoder', 'U3_track_det_head', 'U4_map_seg_head',
    'U5_motion_head', 'U6_occ_head', 'U7_planning_head', 'U9_online_track_memory', 'unknown'
]
FEATURE_ROLES = ['attention', 'transformer_layer', 'planning_head', 'ffn', 'projection_or_head', 'generic']


def feature_token(text: Any) -> str:
    s = str(text or 'unknown').strip()
    if not s:
        s = 'unknown'
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append('_')
    tok = ''.join(out)
    while '__' in tok:
        tok = tok.replace('__', '_')
    return tok.strip('_') or 'unknown'


def safe_float_value(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def safe_int_value(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def flat_update_prefixed(dst: Dict[str, Any], src: Dict[str, Any], prefix: str) -> None:
    for k, v in src.items():
        dst[prefix + str(k)] = v


def trajectory_online_features(traj: Any, prefix: str = 'feature__post_out_') -> Dict[str, Any]:
    """Online-only final-output features computed from the current repaired output.

    These features do not use clean trajectory, GT, fault metadata, repair event fields, or labels.
    They are appropriate only for a post-repair acceptance detector. For a pre-repair detector,
    train with activation-only features.
    """
    out: Dict[str, Any] = {
        prefix + 'traj_present': 0,
        prefix + 'traj_numel': 0,
        prefix + 'traj_ndim': '',
        prefix + 'traj_finite_ratio': 0.0,
        prefix + 'traj_nonfinite_count': 0,
        prefix + 'traj_absmax': '',
        prefix + 'xy_coord_absmax': '',
        prefix + 'xy_step_l2_max': '',
        prefix + 'xy_step_l2_mean': '',
        prefix + 'xy_accel_l2_max': '',
        prefix + 'xy_jerk_l2_max': '',
    }
    if traj is None:
        return out
    try:
        arr = np.asarray(traj, dtype=np.float64)
        out[prefix + 'traj_present'] = 1
        out[prefix + 'traj_numel'] = int(arr.size)
        out[prefix + 'traj_ndim'] = int(arr.ndim)
        if arr.size == 0:
            return out
        finite = np.isfinite(arr)
        out[prefix + 'traj_finite_ratio'] = float(np.mean(finite))
        out[prefix + 'traj_nonfinite_count'] = int(np.size(finite) - int(np.sum(finite)))
        if np.any(finite):
            out[prefix + 'traj_absmax'] = float(np.max(np.abs(arr[finite])))
        if arr.ndim >= 2:
            xy = arr.reshape(-1, arr.shape[-1])[..., :min(2, arr.shape[-1])]
        else:
            xy = arr.reshape(-1, 1)
        xy_finite = np.isfinite(xy)
        if xy.size and np.any(xy_finite):
            out[prefix + 'xy_coord_absmax'] = float(np.max(np.abs(xy[xy_finite])))
        if xy.shape[0] >= 2 and np.isfinite(xy).all():
            step = np.diff(xy, axis=0)
            step_norm = np.linalg.norm(step, axis=1)
            out[prefix + 'xy_step_l2_max'] = float(np.max(step_norm)) if step_norm.size else 0.0
            out[prefix + 'xy_step_l2_mean'] = float(np.mean(step_norm)) if step_norm.size else 0.0
            if step.shape[0] >= 2:
                acc = np.diff(step, axis=0)
                acc_norm = np.linalg.norm(acc, axis=1)
                out[prefix + 'xy_accel_l2_max'] = float(np.max(acc_norm)) if acc_norm.size else 0.0
                if acc.shape[0] >= 2:
                    jerk = np.diff(acc, axis=0)
                    jerk_norm = np.linalg.norm(jerk, axis=1)
                    out[prefix + 'xy_jerk_l2_max'] = float(np.max(jerk_norm)) if jerk_norm.size else 0.0
    except Exception as e:
        out[prefix + 'traj_feature_error'] = str(e)[:160]
    return out


def build_post_repair_labels(delta: Dict[str, Any], output_flags: Dict[str, Any], gate_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Offline labels for detector training/evaluation. Never feed these into detector X."""
    labels: Dict[str, Any] = {}
    out_illegal = safe_int_value(output_flags.get('output_illegal_or_missing', 0))
    out_missing = safe_int_value(output_flags.get('output_missing', 0))
    out_nonfinite = safe_int_value(output_flags.get('output_nonfinite', 0))
    out_shape = safe_int_value(output_flags.get('output_shape_invalid', 0))
    gate_reject = safe_int_value(gate_fields.get('gate_reject', 0))
    gate_accept = safe_int_value(gate_fields.get('gate_accept', 0))
    l2max = safe_float_value(delta.get('traj_delta_l2_max', float('nan')), default=float('nan'))
    l2final = safe_float_value(delta.get('traj_delta_l2_final', float('nan')), default=float('nan'))
    l2mean = safe_float_value(delta.get('traj_delta_l2_mean', float('nan')), default=float('nan'))
    harmful_0p5 = int(out_illegal or (math.isfinite(l2max) and l2max > 0.5))
    severe_1p0 = int(out_illegal or (math.isfinite(l2max) and l2max > 1.0))
    final_0p5 = int(out_illegal or (math.isfinite(l2final) and l2final > 0.5))
    final_1p0 = int(out_illegal or (math.isfinite(l2final) and l2final > 1.0))
    unsafe_0p5 = int(out_illegal or gate_reject or harmful_0p5)
    unsafe_1p0 = int(out_illegal or gate_reject or severe_1p0)
    labels.update({
        'label__repaired_output_illegal_or_missing': out_illegal,
        'label__repaired_output_missing': out_missing,
        'label__repaired_output_nonfinite': out_nonfinite,
        'label__repaired_output_shape_invalid': out_shape,
        'label__gate_reject': gate_reject,
        'label__gate_accept': gate_accept,
        'label__traj_delta_l2_available': safe_int_value(delta.get('traj_delta_available', 0)),
        'label__traj_delta_l2_mean': delta.get('traj_delta_l2_mean', ''),
        'label__traj_delta_l2_max': delta.get('traj_delta_l2_max', ''),
        'label__traj_delta_l2_final': delta.get('traj_delta_l2_final', ''),
        'label__repaired_harmful_l2max_gt_0p5_or_illegal': harmful_0p5,
        'label__repaired_severe_l2max_gt_1p0_or_illegal': severe_1p0,
        'label__repaired_final_l2_gt_0p5_or_illegal': final_0p5,
        'label__repaired_final_l2_gt_1p0_or_illegal': final_1p0,
        'label__post_repair_unsafe_0p5_or_gate': unsafe_0p5,
        'label__post_repair_unsafe_1p0_or_gate': unsafe_1p0,
        'label__post_repair_accepted_safe_0p5': int(not unsafe_0p5),
        'label__post_repair_accepted_safe_1p0': int(not unsafe_1p0),
    })
    return labels


def leakage_audit_feature_keys(feature_keys: List[str]) -> Dict[str, Any]:
    forbidden = [
        'repair_', 'gate_', 'label_', 'target_', 'fault_', 'bit_', 'clean_', 'delta_',
        'harmful', 'severe', 'applied', 'effective', 'event_id', 'module_name',
        'high_contam', 'module_aware', 'replacement', 'before', 'after',
    ]
    hits = []
    for k in sorted(set(feature_keys)):
        lk = k.lower()
        for bad in forbidden:
            if bad in lk:
                hits.append({'feature_key': k, 'matched_forbidden_token': bad})
                break
    return {'feature_key_count': len(set(feature_keys)), 'forbidden_hit_count': len(hits), 'forbidden_hits': hits[:200]}



def _input_tensor_compatible(input_tensor: Any, ref: torch.Tensor, threshold: float) -> Tuple[bool, str]:
    """Check whether a current-layer input tensor can be used as an online rollback value.

    This uses only the current forward inputs and numerical contract. It does not use clean
    reference, GT, fault metadata, or labels. The shape must match the hook output tensor so
    that set_first_tensor can safely replace the output.
    """
    if input_tensor is None or not torch.is_tensor(input_tensor):
        return False, 'no_input_tensor'
    if not torch.is_floating_point(input_tensor):
        return False, 'input_not_floating'
    if tuple(input_tensor.shape) != tuple(ref.shape):
        return False, 'input_shape_mismatch'
    if input_tensor.numel() != ref.numel():
        return False, 'input_numel_mismatch'
    try:
        xf = input_tensor.detach().float()
        finite = torch.isfinite(xf)
        if not bool(finite.all().detach().cpu().item()):
            return False, 'input_not_all_finite'
        if math.isfinite(float(threshold)) and float(threshold) > 0:
            absmax = float(xf.abs().max().detach().cpu().item()) if xf.numel() else 0.0
            if math.isfinite(absmax) and absmax > float(threshold):
                return False, 'input_exceeds_output_contract'
        return True, 'input_compatible'
    except Exception as e:
        return False, 'input_check_error:' + str(e)[:120]


def _zero_like_repair(t: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(t)



def _lower_join(*parts: Any) -> str:
    return ' '.join(str(x or '').lower() for x in parts)


def module_repair_role(group: str, module_class: str, module_name: str) -> str:
    """Coarse online role used only to choose a safer high-contamination action.

    This uses module naming and registry metadata, not fault metadata or clean/reference output.
    """
    text = _lower_join(group, module_class, module_name)
    if 'planning' in text or str(group) == 'U7_planning_head':
        return 'planning_head'
    if 'multiheadattention' in text or 'multi_head_attention' in text or 'attention' in text:
        return 'attention'
    if 'ffn' in text or 'feedforward' in text or 'feed_forward' in text:
        return 'ffn'
    if 'detrtransformerdecoderlayer' in text or 'decoderlayer' in text or 'bevformerlayer' in text or 'transformerlayers' in text:
        return 'transformer_layer'
    if 'linear' in text or 'conv' in text or 'mlp' in text:
        return 'projection_or_head'
    return 'generic'


def choose_module_aware_high_contam_action(group: str, module_class: str, module_name: str, input_ok: bool, aggressive: bool = False) -> Tuple[str, str, str]:
    """Choose a high-contamination containment action.

    Returned action is one of: input_passthrough, zero_tensor, standard_repair.
    The policy is deliberately online-only: it uses registry metadata and current tensor/input
    compatibility, but does not use clean trajectory, labels, GT, or injection metadata.
    """
    role = module_repair_role(group, module_class, module_name)
    g = str(group or '')

    # Planning output is close to the final trajectory. Avoid zeroing large planning tensors;
    # let standard hybrid repair and the trajectory gate decide validity.
    if role == 'planning_head' or g == 'U7_planning_head':
        return ('standard_repair', role, 'planning_head_no_aggressive_hidden_zero')

    # Residual/decoder style layers: if input and output shapes match and input is clean under
    # the local contract, rollback to the current layer input. If rollback is unavailable, avoid
    # zeroing the whole layer output because it may erase the residual state.
    if role == 'transformer_layer':
        if input_ok:
            return ('input_passthrough', role, 'layer_rollback_to_current_input')
        if aggressive and g not in {'U2_bev_encoder'}:
            return ('zero_tensor', role, 'aggressive_layer_zero_fallback')
        return ('standard_repair', role, 'layer_no_compatible_input_standard_fallback')

    # Attention / FFN are the dominant still-NaN failure classes. Prefer residual/query input
    # rollback when compatible. If not compatible, the non-aggressive policy uses zero only for
    # U3/U4/U5 attention/FFN as contribution containment; other groups fall back to standard.
    if role in {'attention', 'ffn'}:
        if input_ok:
            return ('input_passthrough', role, f'{role}_rollback_to_query_or_residual_input')
        if aggressive or g in {'U3_track_det_head', 'U4_map_seg_head', 'U5_motion_head'}:
            return ('zero_tensor', role, f'{role}_zero_contribution_containment')
        return ('standard_repair', role, f'{role}_no_compatible_input_standard_fallback')

    # For generic projections/heads, use rollback if safe; otherwise fall back to zero only in
    # aggressive mode. The default module-aware policy avoids broad zeroing outside the known
    # high-risk residual/attention/FFN cases.
    if input_ok:
        return ('input_passthrough', role, 'generic_rollback_to_current_input')
    if aggressive and g not in {'U7_planning_head'}:
        return ('zero_tensor', role, 'aggressive_generic_zero_fallback')
    return ('standard_repair', role, 'generic_standard_fallback')

def repair_tensor(t: torch.Tensor, threshold: float, method: str, max_values_for_median: int,
                  input_tensor: Any = None, high_contam_flag: bool = False,
                  high_contam_strategy: str = 'off', module_name: str = '',
                  module_class: str = '', uniad_group: str = '') -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fault-agnostic numerical restoration.

    Detection uses only current tensor finite/range contract. The method does not inspect fault type,
    bit position, or injection metadata. It intentionally avoids direct torch.nan_to_num repair.
    """
    info: Dict[str, Any] = {
        'repair_method': method,
        'threshold_absmax': float(threshold),
        'repair_triggered': 0,
        'repair_modified': 0,
        'repair_reason': 'ok_no_violation',
        'abnormal_count_sample': 0,
        'nonfinite_count_sample': 0,
        'huge_count_sample': 0,
        'replacement_value': '',
        'high_contam_flag': int(bool(high_contam_flag)),
        'high_contam_strategy': str(high_contam_strategy),
        'high_contam_action': '',
        'high_contam_input_status': '',
        'module_aware_role': '',
        'module_aware_policy_reason': '',
    }
    if t is None or not torch.is_tensor(t) or not torch.is_floating_point(t) or t.numel() <= 0:
        info['repair_reason'] = 'not_repairable_tensor'
        return t, info
    if not math.isfinite(float(threshold)) or float(threshold) <= 0:
        info['repair_reason'] = 'invalid_threshold'
        return t, info

    with torch.no_grad():
        # Full-tensor mask is used for the actual repair. This first smoke prioritizes correctness
        # over overhead; later versions can restrict to sampled/sub-block repair.
        xf = t.float()
        finite = torch.isfinite(xf)
        huge = finite & (xf.abs() > float(threshold))
        abnormal = (~finite) | huge
        abnormal_any = bool(abnormal.any().detach().cpu().item())
        if not abnormal_any:
            return t, info
        nonfinite_count = int((~finite).sum().detach().cpu().item())
        huge_count = int(huge.sum().detach().cpu().item())
        abnormal_count = int(abnormal.sum().detach().cpu().item())
        info.update({
            'repair_triggered': 1,
            'abnormal_count_sample': abnormal_count,
            'nonfinite_count_sample': nonfinite_count,
            'huge_count_sample': huge_count,
        })

        # v3.5.8 high-contamination handling. When a tensor is already massively
        # contaminated, local median/clip can produce a legal-looking but semantically
        # weak tensor. The new module-aware policy uses registry group/class/name to choose
        # a residual-preserving or contribution-containment action for U3/U4/U5 attention,
        # FFN, and decoder-style modules.
        if bool(high_contam_flag) and str(high_contam_strategy) not in {'', 'off', 'median'} and method != 'detect_only':
            strat = str(high_contam_strategy)
            input_ok, input_reason = _input_tensor_compatible(input_tensor, t, threshold)
            info['high_contam_input_status'] = input_reason

            action = ''
            role = ''
            policy_reason = ''
            if strat in {'module_aware', 'module_aware_aggressive'}:
                action, role, policy_reason = choose_module_aware_high_contam_action(
                    uniad_group, module_class, module_name, input_ok,
                    aggressive=(strat == 'module_aware_aggressive'),
                )
                info['module_aware_role'] = role
                info['module_aware_policy_reason'] = policy_reason
            elif strat in {'input_passthrough', 'input_then_zero'} and input_ok:
                action, role, policy_reason = 'input_passthrough', module_repair_role(uniad_group, module_class, module_name), 'legacy_input_passthrough'
                info['module_aware_role'] = role
                info['module_aware_policy_reason'] = policy_reason
            elif strat in {'zero', 'input_then_zero'}:
                action, role, policy_reason = 'zero_tensor', module_repair_role(uniad_group, module_class, module_name), 'legacy_zero_containment'
                info['module_aware_role'] = role
                info['module_aware_policy_reason'] = policy_reason
            else:
                action, role, policy_reason = 'standard_repair', module_repair_role(uniad_group, module_class, module_name), 'fallback_to_standard_repair'
                info['module_aware_role'] = role
                info['module_aware_policy_reason'] = policy_reason

            if action == 'input_passthrough' and input_ok:
                out = input_tensor.detach().clone().to(device=t.device, dtype=t.dtype)
                info['repair_modified'] = 1
                info['repair_reason'] = 'high_contamination_module_aware_input_passthrough'
                info['high_contam_action'] = 'input_passthrough'
                return out, info
            if action == 'zero_tensor':
                out = _zero_like_repair(t)
                info['repair_modified'] = 1
                info['repair_reason'] = 'high_contamination_module_aware_zero_tensor_containment'
                info['high_contam_action'] = 'zero_tensor'
                return out, info
            # action == standard_repair: continue to ordinary hybrid/median/clip repair below.
            info['high_contam_action'] = 'module_aware_standard_repair_fallback' if strat.startswith('module_aware') else 'fallback_to_standard_repair'

        if method == 'detect_only':
            info['repair_reason'] = 'violation_detected_no_modification'
            return t, info

        out = t.clone()
        outf = out.float()

        if method == 'clip':
            # Clip only finite extremes. Nonfinite values are not silently converted.
            if huge_count > 0:
                clamped = torch.clamp(outf, min=-float(threshold), max=float(threshold)).to(dtype=out.dtype)
                out = torch.where(huge, clamped, out)
                info['repair_modified'] = 1
                info['repair_reason'] = 'finite_extremes_clipped_nonfinite_not_modified' if nonfinite_count else 'finite_extremes_clipped'
            else:
                info['repair_reason'] = 'nonfinite_detected_clip_cannot_restore'
            return out, info

        # Build robust replacement from finite in-contract values from current tensor context.
        in_bound = finite & (outf.abs() <= float(threshold))
        if bool(in_bound.any().detach().cpu().item()):
            vals = outf[in_bound]
            if max_values_for_median and int(max_values_for_median) > 0 and vals.numel() > int(max_values_for_median):
                vals = vals[:int(max_values_for_median)]
            # median is more robust than mean for isolated spikes.
            repl = vals.median()
            repl_val = float(repl.detach().cpu().item())
        else:
            # If the entire tensor is corrupted, use zero-centered contract fallback but still mark low confidence.
            repl = torch.tensor(0.0, device=out.device, dtype=torch.float32)
            repl_val = 0.0
            info['repair_low_confidence'] = 1
        repl_cast = repl.to(dtype=out.dtype)
        info['replacement_value'] = repl_val

        if method == 'median_impute':
            out = torch.where(abnormal, repl_cast.expand_as(out), out)
            info['repair_modified'] = 1
            info['repair_reason'] = 'abnormal_values_imputed_with_tensor_median'
            return out, info

        if method == 'hybrid':
            # Finite extremes are saturated to calibrated range; nonfinite values are imputed from tensor context.
            if huge_count > 0:
                clamped = torch.clamp(outf, min=-float(threshold), max=float(threshold)).to(dtype=out.dtype)
                out = torch.where(huge, clamped, out)
            if nonfinite_count > 0:
                out = torch.where(~finite, repl_cast.expand_as(out), out)
            info['repair_modified'] = 1
            info['repair_reason'] = 'finite_extremes_clipped_nonfinite_imputed_with_tensor_median'
            return out, info

        raise ValueError(f'Unsupported repair method: {method}')


class FaultRepairRuntime:
    def __init__(self, registry_rows: List[Dict[str, str]], thresholds: Dict[str, Dict[str, Any]], method: str, groups_filter=None, stat_max_values: int = 4096, median_max_values: int = 4096, sampling_strategy: str = 'firstk', high_risk_sample_values: int = 16384, low_risk_sample_values: int = 4096, block_size: int = 256, escalate_full_on_trigger: bool = True, repair_policy: str = 'first_only', max_repair_count_per_sample: int = 1, high_contam_strategy: str = 'off', high_contam_ratio: float = 0.5, enable_acceptance_gate: bool = False, gate_coordinate_absmax: float = 200.0, gate_step_l2_max: float = 20.0, gate_accel_l2_max: float = 40.0, gate_jerk_l2_max: float = 80.0, extended_feature_max_values: int = 4096):
        self.registry_rows = list(registry_rows)
        self.target_by_module = {r['module_name']: r for r in self.registry_rows}
        self.thresholds = dict(thresholds)
        self.method = str(method)
        self.groups_filter = groups_filter
        self.stat_max_values = int(stat_max_values)
        self.median_max_values = int(median_max_values)
        self.sampling_strategy = str(sampling_strategy)
        self.high_risk_sample_values = int(high_risk_sample_values)
        self.low_risk_sample_values = int(low_risk_sample_values)
        self.block_size = int(block_size)
        self.escalate_full_on_trigger = bool(escalate_full_on_trigger)
        self.repair_policy = str(repair_policy)
        self.max_repair_count_per_sample = int(max_repair_count_per_sample)
        self.high_contam_strategy = str(high_contam_strategy)
        self.high_contam_ratio = float(high_contam_ratio)
        self.enable_acceptance_gate = bool(enable_acceptance_gate)
        self.gate_coordinate_absmax = float(gate_coordinate_absmax)
        self.gate_step_l2_max = float(gate_step_l2_max)
        self.gate_accel_l2_max = float(gate_accel_l2_max)
        self.gate_jerk_l2_max = float(gate_jerk_l2_max)
        self.extended_feature_max_values = int(extended_feature_max_values)
        self.current_sample_i = -1
        self.current_sample_id = ''
        self.pending = None
        self.exact_infos: List[Dict[str, Any]] = []
        self.base_seed = 0
        self.consumed_event_ids = set()
        self.fault_infos: List[Dict[str, Any]] = []
        self.repair_events: List[Dict[str, Any]] = []
        self.call_counts = Counter()
        self.trace_order = 0
        self.first_detect_event = None
        self.first_repair_done = False
        self.repair_modified_count_so_far = 0
        self.repair_modified_count_by_group = Counter()
        self.repair_modified_count_by_module = Counter()
        self.final_trace_order = 0
        self.detector_features: Dict[str, Any] = {}
        self._detector_mean_sums: Dict[str, float] = defaultdict(float)
        self._detector_mean_counts: Dict[str, int] = defaultdict(int)
        self._detector_first_violation_recorded = False

    def reset_sample(self, sample_i: int, sample_id: str, pending, exact_infos: List[Dict[str, Any]], base_seed: int):
        self.current_sample_i = int(sample_i)
        self.current_sample_id = str(sample_id)
        self.pending = pending
        self.exact_infos = list(exact_infos or [])
        self.base_seed = int(base_seed)
        self.consumed_event_ids = set()
        self.fault_infos = []
        self.repair_events = []
        self.call_counts = Counter()
        self.trace_order = 0
        self.first_detect_event = None
        self.first_repair_done = False
        self.repair_modified_count_so_far = 0
        self.repair_modified_count_by_group = Counter()
        self.repair_modified_count_by_module = Counter()
        self.final_trace_order = 0
        self.detector_features = {}
        self._detector_mean_sums = defaultdict(float)
        self._detector_mean_counts = defaultdict(int)
        self._detector_first_violation_recorded = False

    def exact_info_for_spec(self, spec):
        if not self.exact_infos:
            return None
        for info in self.exact_infos:
            try:
                if int(info.get('event_id', -999)) == int(spec.event_id):
                    return info
            except Exception:
                pass
        return self.exact_infos[0]


    def _feature_inc(self, key: str, value: int = 1):
        self.detector_features[key] = int(self.detector_features.get(key, 0) or 0) + int(value)

    def _feature_max(self, key: str, value: Any):
        try:
            v = float(value)
        except Exception:
            return
        if not math.isfinite(v):
            return
        old = self.detector_features.get(key, None)
        if old is None or old == '':
            self.detector_features[key] = v
        else:
            try:
                self.detector_features[key] = max(float(old), v)
            except Exception:
                self.detector_features[key] = v

    def _feature_min(self, key: str, value: Any):
        try:
            v = float(value)
        except Exception:
            return
        if not math.isfinite(v):
            return
        old = self.detector_features.get(key, None)
        if old is None or old == '':
            self.detector_features[key] = v
        else:
            try:
                self.detector_features[key] = min(float(old), v)
            except Exception:
                self.detector_features[key] = v

    def _feature_mean_acc(self, key: str, value: Any):
        try:
            v = float(value)
        except Exception:
            return
        if not math.isfinite(v):
            return
        self._detector_mean_sums[key] += v
        self._detector_mean_counts[key] += 1

    def record_detector_hook_feature(self, target: Dict[str, Any], module_name: str, st: Dict[str, Any], full_st: Dict[str, Any], violation: bool, raw_st: Optional[Dict[str, Any]] = None):
        """Pre-repair detector branch.

        This method records only online numerical observations from the current tensor BEFORE
        any repair output is applied. It intentionally does not record repair actions, repair
        counts, fault metadata, clean trajectory, GT, or labels.
        """
        group = str(target.get('uniad_group', '') or 'unknown')
        role = module_repair_role(group, target.get('module_class', ''), module_name)
        # Use fixed grouping for new hook-level features, while keeping legacy group summaries unchanged.
        fixed_group = fixed_infer_group(module_name, target.get('module_class', ''), group)
        fixed_role = fixed_infer_role(module_name, target.get('module_class', ''), role)
        if raw_st:
            record_hook_scalar_features(self.detector_features, module_name, target.get('module_class', ''), fixed_group, fixed_role, raw_st, self.call_counts[module_name])
        gtok = feature_token(group)
        rtok = feature_token(role)
        self._feature_inc('feature__act_checked_hook_count', 1)
        self._feature_inc(f'feature__act_group__{gtok}__checked_hook_count', 1)
        self._feature_inc(f'feature__act_role__{rtok}__checked_hook_count', 1)
        # Sampled pre-repair values. These are contract-monitor observations, not repair outputs.
        nf = safe_int_value(st.get('nonfinite_count', 0))
        huge = safe_int_value(st.get('huge_count', 0))
        abnormal = safe_int_value(st.get('abnormal_count', 0))
        sample_numel = safe_int_value(st.get('sample_numel', 0))
        self._feature_inc('feature__act_nonfinite_count_sample_sum', nf)
        self._feature_inc('feature__act_huge_count_sample_sum', huge)
        self._feature_inc('feature__act_abnormal_count_sample_sum', abnormal)
        self._feature_inc('feature__act_sample_numel_checked_sum', sample_numel)
        for prefix in ['feature__act', f'feature__act_group__{gtok}', f'feature__act_role__{rtok}']:
            self._feature_max(prefix + '__max_absmax_ratio_sample', st.get('absmax_ratio'))
            self._feature_max(prefix + '__max_abnormal_ratio_sample', st.get('abnormal_ratio'))
            self._feature_max(prefix + '__max_nonfinite_ratio_sample', st.get('nonfinite_ratio'))
            self._feature_max(prefix + '__max_huge_ratio_sample', st.get('huge_ratio'))
            self._feature_min(prefix + '__min_finite_ratio_sample', st.get('finite_ratio'))
            self._feature_mean_acc(prefix + '__mean_abnormal_ratio_sample', st.get('abnormal_ratio'))
            self._feature_mean_acc(prefix + '__mean_nonfinite_ratio_sample', st.get('nonfinite_ratio'))
            self._feature_mean_acc(prefix + '__mean_huge_ratio_sample', st.get('huge_ratio'))
        if full_st:
            for prefix in ['feature__act', f'feature__act_group__{gtok}', f'feature__act_role__{rtok}']:
                self._feature_max(prefix + '__max_absmax_ratio_full_on_trigger', full_st.get('absmax_ratio'))
                self._feature_max(prefix + '__max_abnormal_ratio_full_on_trigger', full_st.get('abnormal_ratio'))
                self._feature_min(prefix + '__min_finite_ratio_full_on_trigger', full_st.get('finite_ratio'))
        if violation:
            self._feature_inc('feature__act_violation_event_count', 1)
            self._feature_inc(f'feature__act_group__{gtok}__violation_event_count', 1)
            self._feature_inc(f'feature__act_role__{rtok}__violation_event_count', 1)
            if not self._detector_first_violation_recorded:
                self._detector_first_violation_recorded = True
                self.detector_features['feature__act_first_violation_trace_order'] = int(self.trace_order)
                self.detector_features['feature__act_first_violation_absmax_ratio_sample'] = st.get('absmax_ratio', '')
                self.detector_features['feature__act_first_violation_abnormal_ratio_sample'] = st.get('abnormal_ratio', '')
                self.detector_features['feature__act_first_violation_nonfinite_ratio_sample'] = st.get('nonfinite_ratio', '')
                self.detector_features['feature__act_first_violation_huge_ratio_sample'] = st.get('huge_ratio', '')
                self.detector_features[f'feature__act_first_violation_group__{gtok}'] = 1
                self.detector_features[f'feature__act_first_violation_role__{rtok}'] = 1

    def finish_detector_features(self) -> Dict[str, Any]:
        out = dict(self.detector_features)
        out['feature__act_final_trace_order'] = int(self.trace_order)
        out['feature__act_has_violation'] = int(int(out.get('feature__act_violation_event_count', 0) or 0) > 0)
        for k, s in self._detector_mean_sums.items():
            c = int(self._detector_mean_counts.get(k, 0) or 0)
            if c > 0:
                out[k] = float(s) / float(c)
        # Stabilize schema for common groups/roles with zero-valued counters.
        for g in FEATURE_GROUPS:
            gt = feature_token(g)
            out.setdefault(f'feature__act_group__{gt}__checked_hook_count', 0)
            out.setdefault(f'feature__act_group__{gt}__violation_event_count', 0)
        for r in FEATURE_ROLES:
            rt = feature_token(r)
            out.setdefault(f'feature__act_role__{rt}__checked_hook_count', 0)
            out.setdefault(f'feature__act_role__{rt}__violation_event_count', 0)
        return out

    def hook_for(self, module_name: str):
        target = self.target_by_module[module_name]
        threshold_rec = self.thresholds.get(module_name)
        method = self.method

        def _hook(module, inputs, output):
            self.call_counts[module_name] += 1
            self.trace_order += 1
            current_output = output

            # Reproduce the same injected activation fault. Exact fault metadata is used only here;
            # detection/restoration below remains fault-agnostic.
            spec = self.pending
            if spec is not None and spec.module_name == module_name and int(spec.event_id) not in self.consumed_event_ids:
                self.consumed_event_ids.add(int(spec.event_id))
                before_t = helpers.first_tensor(current_output)
                rng = random.Random(helpers.sample_seed(self.base_seed, self.current_sample_id, int(spec.event_id)))
                exact_info = self.exact_info_for_spec(spec)
                new_t, info = helpers.mutate_activation_clone_exact_or_random(before_t, spec, rng, exact_info)
                info['call_count_at_injection'] = int(self.call_counts[module_name])
                info['fault_trace_order'] = int(self.trace_order)
                info['fault_module_name'] = module_name
                info['fault_uniad_group'] = target.get('uniad_group', '')
                info['fault_module_class'] = target.get('module_class', '')
                replaced, ok = helpers.set_first_tensor(current_output, new_t)
                info['replace_output_ok'] = int(ok)
                if not ok:
                    info['applied'] = 0
                    info['effective'] = 0
                    info['reason'] = info.get('reason', 'replace_output_failed')
                self.fault_infos.append(info)
                current_output = replaced if ok else current_output

            # Fault-agnostic numerical contract check and optional repair.
            group = target.get('uniad_group', '')
            if self.groups_filter is not None and group not in self.groups_filter:
                return current_output
            if threshold_rec is None:
                return current_output
            t = helpers.first_tensor(current_output)
            if t is None or not torch.is_tensor(t) or not torch.is_floating_point(t):
                return current_output

            threshold = float(threshold_rec['threshold_absmax'])
            budget = monitoring_budget_for_tensor(t, target, self.sampling_strategy, self.high_risk_sample_values, self.low_risk_sample_values)
            stat_key = f'{self.current_sample_i}:{module_name}:{self.trace_order}:{self.sampling_strategy}'
            st = tensor_contract_stats(
                t, threshold, budget, self.sampling_strategy, stat_key,
                module_class=target.get('module_class', ''), uniad_group=group, block_size=self.block_size,
            )
            raw_st = extended_tensor_scan_stats(t, self.extended_feature_max_values)
            violation = (int(st.get('nonfinite_count', 0) or 0) > 0) or (int(st.get('huge_count', 0) or 0) > 0)
            if not violation:
                self.record_detector_hook_feature(target, module_name, st, {}, False, raw_st)
                return current_output

            full_st = {}
            if self.escalate_full_on_trigger and not int(st.get('sampled_full_tensor', 0) or 0):
                full_st = tensor_contract_stats(
                    t, threshold, -1, 'full', stat_key + ':full_escalation',
                    module_class=target.get('module_class', ''), uniad_group=group, block_size=self.block_size,
                )

            self.record_detector_hook_feature(target, module_name, st, full_st, bool(violation), raw_st)

            should_repair = True
            no_modify_reason = ''
            if self.method == 'detect_only':
                should_repair = False
                no_modify_reason = 'violation_detected_no_modification'
            elif self.repair_policy == 'first_only' and self.first_repair_done:
                should_repair = False
                no_modify_reason = 'violation_detected_after_first_repair_no_modification'
            elif self.repair_policy == 'budgeted_multi' and self.repair_modified_count_so_far >= int(self.max_repair_count_per_sample):
                should_repair = False
                no_modify_reason = 'violation_detected_repair_budget_exhausted'

            contam_ratio_for_policy = full_st.get('abnormal_ratio', st.get('abnormal_ratio')) if full_st else st.get('abnormal_ratio')
            finite_ratio_for_policy = full_st.get('finite_ratio', st.get('finite_ratio')) if full_st else st.get('finite_ratio')
            try:
                contam_ratio_float = float(contam_ratio_for_policy)
            except Exception:
                contam_ratio_float = float('nan')
            try:
                finite_ratio_float = float(finite_ratio_for_policy)
            except Exception:
                finite_ratio_float = float('nan')
            high_contam_flag = (math.isfinite(contam_ratio_float) and contam_ratio_float >= float(self.high_contam_ratio)) or (math.isfinite(finite_ratio_float) and finite_ratio_float <= 0.5)

            if should_repair:
                input_t = helpers.first_tensor(inputs)
                new_t, repair_info = repair_tensor(
                    t, threshold, method, self.median_max_values,
                    input_tensor=input_t,
                    high_contam_flag=high_contam_flag,
                    high_contam_strategy=self.high_contam_strategy,
                    module_name=module_name,
                    module_class=target.get('module_class', ''),
                    uniad_group=group,
                )
                replaced, ok = helpers.set_first_tensor(current_output, new_t)
                if bool(repair_info.get('repair_modified', 0)):
                    self.first_repair_done = True
                    self.repair_modified_count_so_far += 1
                    self.repair_modified_count_by_group[group] += 1
                    self.repair_modified_count_by_module[module_name] += 1
                repair_info['repair_idx'] = int(self.repair_modified_count_so_far) if bool(repair_info.get('repair_modified', 0)) else ''
            else:
                repair_info = {
                    'repair_method': method,
                    'threshold_absmax': threshold,
                    'repair_triggered': 1,
                    'repair_modified': 0,
                    'repair_reason': no_modify_reason,
                    'abnormal_count_sample': int(st.get('abnormal_count', 0) or 0),
                    'nonfinite_count_sample': int(st.get('nonfinite_count', 0) or 0),
                    'huge_count_sample': int(st.get('huge_count', 0) or 0),
                    'replacement_value': '',
                    'repair_idx': '',
                    'high_contam_flag': int(bool(high_contam_flag)),
                    'high_contam_strategy': str(self.high_contam_strategy),
                    'high_contam_action': '',
                    'high_contam_input_status': '',
                    'module_aware_role': module_repair_role(group, target.get('module_class', ''), module_name),
                    'module_aware_policy_reason': '',
                }
                replaced, ok = current_output, True

            event_base = {
                'sample_i': int(self.current_sample_i),
                'sample_id': self.current_sample_id,
                'trace_order': int(self.trace_order),
                'module_name': module_name,
                'target_id': target.get('target_id', ''),
                'module_class': target.get('module_class', ''),
                'uniad_group': group,
                'call_count': int(self.call_counts[module_name]),
                'sampling_strategy': self.sampling_strategy,
                'monitor_sample_budget': int(budget),
                'repair_policy': self.repair_policy,
                'max_repair_count_per_sample': int(self.max_repair_count_per_sample),
                'repair_modified_count_so_far': int(self.repair_modified_count_so_far),
                'high_contam_policy_ratio': float(self.high_contam_ratio),
                'high_contam_policy_flag': int(bool(high_contam_flag)),
                'pre_repair_finite_ratio_sample': st.get('finite_ratio'),
                'pre_repair_absmax_finite_sample': st.get('absmax_finite'),
                'pre_repair_nonfinite_count_sample': st.get('nonfinite_count'),
                'pre_repair_sample_numel': st.get('sample_numel'),
                'sample_numel_total': st.get('sample_numel_total'),
                'sampled_full_tensor': st.get('sampled_full_tensor'),
                'nonfinite_count_sample': st.get('nonfinite_count'),
                'huge_count_sample': st.get('huge_count'),
                'abnormal_count_sample': st.get('abnormal_count'),
                'nonfinite_ratio_sample': st.get('nonfinite_ratio'),
                'huge_ratio_sample': st.get('huge_ratio'),
                'abnormal_ratio_sample': st.get('abnormal_ratio'),
                'absmax_ratio_sample': st.get('absmax_ratio'),
                'contamination_band_sample': contamination_band(st.get('abnormal_ratio'), st.get('finite_ratio')),
                'full_check_on_trigger': int(bool(full_st)),
                'replace_output_ok': int(ok),
            }
            if full_st:
                event_base.update({
                    'finite_ratio_full': full_st.get('finite_ratio'),
                    'absmax_finite_full': full_st.get('absmax_finite'),
                    'nonfinite_count_full': full_st.get('nonfinite_count'),
                    'huge_count_full': full_st.get('huge_count'),
                    'abnormal_count_full': full_st.get('abnormal_count'),
                    'nonfinite_ratio_full': full_st.get('nonfinite_ratio'),
                    'huge_ratio_full': full_st.get('huge_ratio'),
                    'abnormal_ratio_full': full_st.get('abnormal_ratio'),
                    'absmax_ratio_full': full_st.get('absmax_ratio'),
                    'contamination_band_full': contamination_band(full_st.get('abnormal_ratio'), full_st.get('finite_ratio')),
                })
            if should_repair and bool(repair_info.get('repair_modified', 0)):
                try:
                    post_t = helpers.first_tensor(replaced if ok else current_output)
                    post_st = tensor_contract_stats(
                        post_t, threshold, -1, 'full', stat_key + ':post_repair_full',
                        module_class=target.get('module_class', ''), uniad_group=group, block_size=self.block_size,
                    )
                    event_base.update({
                        'post_repair_finite_ratio_full': post_st.get('finite_ratio'),
                        'post_repair_absmax_finite_full': post_st.get('absmax_finite'),
                        'post_repair_nonfinite_count_full': post_st.get('nonfinite_count'),
                        'post_repair_huge_count_full': post_st.get('huge_count'),
                        'post_repair_abnormal_count_full': post_st.get('abnormal_count'),
                        'post_repair_abnormal_ratio_full': post_st.get('abnormal_ratio'),
                        'post_repair_absmax_ratio_full': post_st.get('absmax_ratio'),
                        'post_repair_contamination_band_full': contamination_band(post_st.get('abnormal_ratio'), post_st.get('finite_ratio')),
                    })
                except Exception as e:
                    event_base['post_repair_stats_error'] = str(e)[:200]
            repair_info.update(event_base)
            if self.first_detect_event is None:
                repair_info['is_first_detection'] = 1
                self.first_detect_event = dict(repair_info)
            else:
                repair_info['is_first_detection'] = 0
            self.repair_events.append(repair_info)
            return replaced if ok else current_output
        return _hook

    def finish_sample(self):
        self.final_trace_order = int(self.trace_order)
        return list(self.fault_infos), list(self.repair_events), dict(self.call_counts), int(self.final_trace_order), self.finish_detector_features()


def register_repair_hooks(model, runtime: FaultRepairRuntime):
    module_map = dict(model.named_modules())
    handles = []
    missing = []
    registered = []
    for r in runtime.registry_rows:
        name = r['module_name']
        mod = module_map.get(name)
        if mod is None:
            missing.append(r)
            continue
        handles.append(mod.register_forward_hook(runtime.hook_for(name)))
        registered.append(r)
    return handles, {'registered_count': len(registered), 'missing_count': len(missing), 'missing': missing[:20]}


def summarize_repairs(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    c = Counter()
    first = events[0] if events else {}
    groups = Counter()
    methods = Counter()
    for e in events:
        c['repair_trigger_count'] += int(e.get('repair_triggered', 0) or 0)
        c['repair_modified_count'] += int(e.get('repair_modified', 0) or 0)
        if int(e.get('nonfinite_count_sample', 0) or 0) > 0:
            c['repair_nonfinite_event_count'] += 1
        if int(e.get('huge_count_sample', 0) or 0) > 0:
            c['repair_huge_event_count'] += 1
        groups[e.get('uniad_group', '')] += 1
        methods[e.get('repair_method', '')] += 1
    return {
        'repair_event_count': len(events),
        'repair_trigger_count': int(c['repair_trigger_count']),
        'repair_modified_count': int(c['repair_modified_count']),
        'repair_nonfinite_event_count': int(c['repair_nonfinite_event_count']),
        'repair_huge_event_count': int(c['repair_huge_event_count']),
        'first_repair_module_name': first.get('module_name', ''),
        'first_repair_module_class': first.get('module_class', ''),
        'first_repair_uniad_group': first.get('uniad_group', ''),
        'first_repair_reason': first.get('repair_reason', ''),
        'repair_groups_top': ';'.join(f'{k}:{v}' for k, v in groups.most_common(8) if k),
    }



def first_detection_fields(events: List[Dict[str, Any]], fault_infos: List[Dict[str, Any]], final_trace_order: int) -> Dict[str, Any]:
    first = None
    for e in events:
        if int(e.get('is_first_detection', 0) or 0) == 1:
            first = e
            break
    if first is None and events:
        first = events[0]
    fault = fault_infos[0] if fault_infos else {}
    out: Dict[str, Any] = {
        'final_trace_order': int(final_trace_order),
        'true_fault_trace_order': fault.get('fault_trace_order', ''),
        'true_fault_module_name': fault.get('fault_module_name', ''),
        'true_fault_module_class': fault.get('fault_module_class', ''),
        'true_fault_uniad_group': fault.get('fault_uniad_group', ''),
        'first_detect_trace_order': '',
        'first_detect_module_name': '',
        'first_detect_module_class': '',
        'first_detect_uniad_group': '',
        'detection_distance': '',
        'normalized_detection_distance': '',
        'first_detect_finite_ratio_sample': '',
        'first_detect_abnormal_ratio_sample': '',
        'first_detect_nonfinite_ratio_sample': '',
        'first_detect_huge_ratio_sample': '',
        'first_detect_absmax_ratio_sample': '',
        'first_detect_contamination_band_sample': '',
        'first_detect_finite_ratio_full': '',
        'first_detect_abnormal_ratio_full': '',
        'first_detect_contamination_band_full': '',
    }
    if first:
        out.update({
            'first_detect_trace_order': first.get('trace_order', ''),
            'first_detect_module_name': first.get('module_name', ''),
            'first_detect_module_class': first.get('module_class', ''),
            'first_detect_uniad_group': first.get('uniad_group', ''),
            'first_detect_finite_ratio_sample': first.get('pre_repair_finite_ratio_sample', ''),
            'first_detect_abnormal_ratio_sample': first.get('abnormal_ratio_sample', ''),
            'first_detect_nonfinite_ratio_sample': first.get('nonfinite_ratio_sample', ''),
            'first_detect_huge_ratio_sample': first.get('huge_ratio_sample', ''),
            'first_detect_absmax_ratio_sample': first.get('absmax_ratio_sample', ''),
            'first_detect_contamination_band_sample': first.get('contamination_band_sample', ''),
            'first_detect_finite_ratio_full': first.get('finite_ratio_full', ''),
            'first_detect_abnormal_ratio_full': first.get('abnormal_ratio_full', ''),
            'first_detect_contamination_band_full': first.get('contamination_band_full', ''),
        })
    try:
        ft = int(out.get('true_fault_trace_order'))
        dt = int(out.get('first_detect_trace_order'))
        out['detection_distance'] = int(dt - ft)
        denom = int(final_trace_order) - int(ft)
        out['normalized_detection_distance'] = float(dt - ft) / float(denom) if denom > 0 else ''
    except Exception:
        pass
    return out



def trajectory_acceptance_gate(traj: Any, output_flags: Dict[str, Any], coord_absmax: float = 200.0,
                               step_l2_max: float = 20.0, accel_l2_max: float = 40.0,
                               jerk_l2_max: float = 80.0) -> Dict[str, Any]:
    """Online-style final output gate using only the current repaired trajectory.

    This does not compare to clean reference or GT. It is a conservative validity gate to
    avoid counting finite-but-obviously-spiky trajectories as accepted repair outputs.
    """
    out: Dict[str, Any] = {
        'gate_available': 0,
        'gate_accept': 0,
        'gate_reject': 0,
        'gate_reason': '',
        'gate_coord_absmax': '',
        'gate_step_l2_max': '',
        'gate_accel_l2_max': '',
        'gate_jerk_l2_max': '',
        'gate_coord_threshold': float(coord_absmax),
        'gate_step_l2_threshold': float(step_l2_max),
        'gate_accel_l2_threshold': float(accel_l2_max),
        'gate_jerk_l2_threshold': float(jerk_l2_max),
    }
    if int(output_flags.get('output_missing', 0) or 0) or int(output_flags.get('output_shape_invalid', 0) or 0):
        out.update({'gate_available': 1, 'gate_reject': 1, 'gate_reason': 'missing_or_shape_invalid'})
        return out
    if traj is None:
        out.update({'gate_available': 1, 'gate_reject': 1, 'gate_reason': 'traj_none'})
        return out
    try:
        arr = np.asarray(traj, dtype=np.float64)
        out['gate_available'] = 1
        if arr.size == 0:
            out.update({'gate_reject': 1, 'gate_reason': 'traj_empty'})
            return out
        if not np.isfinite(arr).all():
            out.update({'gate_reject': 1, 'gate_reason': 'traj_nonfinite'})
            return out
        if arr.ndim >= 2:
            xy = arr.reshape(-1, arr.shape[-1])[..., :min(2, arr.shape[-1])]
        else:
            xy = arr.reshape(-1, 1)
        coord = float(np.max(np.abs(xy))) if xy.size else 0.0
        out['gate_coord_absmax'] = coord
        if math.isfinite(coord_absmax) and coord > float(coord_absmax):
            out.update({'gate_reject': 1, 'gate_reason': 'coord_absmax_exceeded'})
            return out
        if xy.shape[0] >= 2:
            step = np.diff(xy, axis=0)
            step_norm = np.linalg.norm(step, axis=1)
            step_max = float(np.max(step_norm)) if step_norm.size else 0.0
        else:
            step = np.zeros((0, xy.shape[-1] if xy.ndim == 2 else 1), dtype=np.float64)
            step_max = 0.0
        out['gate_step_l2_max'] = step_max
        if math.isfinite(step_l2_max) and step_max > float(step_l2_max):
            out.update({'gate_reject': 1, 'gate_reason': 'step_l2_exceeded'})
            return out
        if step.shape[0] >= 2:
            acc = np.diff(step, axis=0)
            acc_norm = np.linalg.norm(acc, axis=1)
            acc_max = float(np.max(acc_norm)) if acc_norm.size else 0.0
        else:
            acc = np.zeros((0, xy.shape[-1] if xy.ndim == 2 else 1), dtype=np.float64)
            acc_max = 0.0
        out['gate_accel_l2_max'] = acc_max
        if math.isfinite(accel_l2_max) and acc_max > float(accel_l2_max):
            out.update({'gate_reject': 1, 'gate_reason': 'accel_l2_exceeded'})
            return out
        if acc.shape[0] >= 2:
            jerk = np.diff(acc, axis=0)
            jerk_norm = np.linalg.norm(jerk, axis=1)
            jerk_max = float(np.max(jerk_norm)) if jerk_norm.size else 0.0
        else:
            jerk_max = 0.0
        out['gate_jerk_l2_max'] = jerk_max
        if math.isfinite(jerk_l2_max) and jerk_max > float(jerk_l2_max):
            out.update({'gate_reject': 1, 'gate_reason': 'jerk_l2_exceeded'})
            return out
        out.update({'gate_accept': 1, 'gate_reject': 0, 'gate_reason': 'accepted'})
        return out
    except Exception as e:
        out.update({'gate_available': 1, 'gate_reject': 1, 'gate_reason': 'gate_exception:' + str(e)[:160]})
        return out

def harmful_flags(delta: Dict[str, Any], output_flags: Dict[str, Any]) -> Dict[str, int]:
    # User-confirmed labels for evaluation only. They are not used by repair.
    out_nonfinite = int(output_flags.get('output_nonfinite', 0) or 0)
    out_illegal = int(output_flags.get('output_illegal_or_missing', 0) or 0)
    try:
        l2max = float(delta.get('traj_delta_l2_max', 'nan'))
    except Exception:
        l2max = float('nan')
    harmful = int(out_illegal or out_nonfinite or (math.isfinite(l2max) and l2max > 0.5))
    severe = int(out_illegal or out_nonfinite or (math.isfinite(l2max) and l2max > 1.0))
    return {'harmful_l2_only_or_illegal': harmful, 'severe_l2_only_or_illegal': severe}



# ===== v4.3.1 extended activation feature helpers =====
def sanitize_hook_name(s: str) -> str:
    out=[]
    for ch in str(s):
        out.append(ch.lower() if ch.isalnum() else '_')
    text=''.join(out)
    while '__' in text:
        text=text.replace('__','_')
    return text.strip('_')[:180] or 'unknown'

def fixed_infer_group(module_name: str, module_class: str = '', fallback: str = '') -> str:
    """v4.3.1 group inference with motion/memory priority fixes.

    Scan v4.3.0 intentionally was conservative. For the material logger we fix
    common misclassification: motion_head.* must be U5 even if it contains 'bev',
    and memory_bank.* must be U9.
    """
    n=str(module_name or '').lower(); c=str(module_class or '').lower()
    if n.startswith('motion_head') or '.motion_head' in n or 'motionformer' in n:
        return 'U5_motion_head'
    if n.startswith('memory_bank') or '.memory_bank' in n or 'temporal' in n or 'memory' in n:
        return 'U9_online_track_memory'
    if 'planning' in n or 'plan' in n or 'sdc' in n:
        return 'U7_planning_head'
    if 'occ' in n or 'occupancy' in n:
        return 'U6_occ_head'
    if 'map' in n or 'seg' in n:
        return 'U4_map_seg_head'
    if 'track' in n or 'det' in n or 'bbox' in n:
        return 'U3_track_det_head'
    if 'bev' in n or (('encoder' in n) and 'pts_bbox_head' in n):
        return 'U2_bev_encoder'
    if 'img_backbone' in n or 'img_neck' in n or 'backbone' in n or 'neck' in n:
        return 'U1_img_backbone_neck'
    if fallback:
        return str(fallback)
    if 'transformer' in n or 'decoder' in n or 'encoder' in c:
        return 'U8_transformer_generic'
    return 'U0_other'

def fixed_infer_role(module_name: str, module_class: str = '', fallback: str = '') -> str:
    n=str(module_name or '').lower(); c=str(module_class or '').lower()
    if 'memory' in n or 'temporal' in n:
        return 'temporal_memory'
    if 'planning' in n or 'sdc' in n:
        return 'planning_head'
    if 'attention' in n or 'attn' in n or 'multiheadattention' in c:
        return 'attention'
    if 'ffn' in n or 'feedforward' in n:
        return 'ffn'
    if 'decoder' in n:
        return 'decoder'
    if 'encoder' in n:
        return 'encoder'
    if 'norm' in c or 'norm' in n:
        return 'normalization'
    if 'linear' in c or 'conv' in c:
        return 'projection_or_conv'
    if fallback:
        return str(fallback)
    return 'generic'

def extended_tensor_scan_stats(t: Any, max_values: int = 4096) -> Dict[str, Any]:
    """Read-only scalar statistics for one online activation tensor.

    Shape may vary; output schema does not. All reductions are on a bounded CPU sample
    to avoid CUDA indexing assertions seen in broad scans.
    """
    if not torch.is_tensor(t):
        return {'tensor_available':0, 'finite_ratio':'', 'nonfinite_count_sample':'', 'absmax_sample':'', 'mean_abs_sample':'', 'std_sample':'', 'q90_abs_sample':'', 'q95_abs_sample':'', 'q99_abs_sample':'', 'zero_ratio_sample':'', 'large_1e3_ratio_sample':'', 'large_1e4_ratio_sample':'', 'numel':0}
    out={'tensor_available':1, 'numel': int(t.numel()), 'ndim': int(t.ndim), 'dtype': str(t.dtype)}
    try:
        with torch.no_grad():
            n=int(t.numel())
            if n<=0:
                out.update({'finite_ratio':1.0,'nonfinite_count_sample':0,'absmax_sample':0.0,'mean_abs_sample':0.0,'std_sample':0.0,'q90_abs_sample':0.0,'q95_abs_sample':0.0,'q99_abs_sample':0.0,'zero_ratio_sample':1.0,'large_1e3_ratio_sample':0.0,'large_1e4_ratio_sample':0.0})
                return out
            if not torch.is_floating_point(t):
                out['summary_error']='non_floating_tensor_skipped'
                return out
            flat=t.detach().reshape(-1)
            m=min(max(1,int(max_values)), n)
            if n>m:
                if m<=1:
                    idx=torch.zeros((1,), device=flat.device, dtype=torch.long)
                else:
                    step=max(1,(n-1)//(m-1))
                    idx=torch.arange(0, step*m, step, device=flat.device, dtype=torch.long)[:m]
                    if idx.numel()>0:
                        idx[-1]=n-1
                    idx=torch.clamp(idx,0,n-1)
                sample=flat.index_select(0,idx)
            else:
                sample=flat
            sf=sample.detach().float().cpu()
            finite=torch.isfinite(sf)
            out['finite_ratio']=float(finite.float().mean().item()) if sf.numel() else 1.0
            out['nonfinite_count_sample']=int((~finite).sum().item()) if sf.numel() else 0
            if bool(finite.any().item()):
                vals=sf[finite]
                av=vals.abs()
                out['absmax_sample']=float(av.max().item())
                out['mean_abs_sample']=float(av.mean().item())
                out['std_sample']=float(vals.std(unbiased=False).item()) if vals.numel()>1 else 0.0
                av_np=av.numpy()
                out['q90_abs_sample']=float(np.quantile(av_np,0.90))
                out['q95_abs_sample']=float(np.quantile(av_np,0.95))
                out['q99_abs_sample']=float(np.quantile(av_np,0.99))
                out['zero_ratio_sample']=float((av==0).float().mean().item())
                out['large_1e3_ratio_sample']=float((av>1e3).float().mean().item())
                out['large_1e4_ratio_sample']=float((av>1e4).float().mean().item())
            else:
                out.update({'absmax_sample':float('nan'),'mean_abs_sample':float('nan'),'std_sample':float('nan'),'q90_abs_sample':float('nan'),'q95_abs_sample':float('nan'),'q99_abs_sample':float('nan'),'zero_ratio_sample':float('nan'),'large_1e3_ratio_sample':float('nan'),'large_1e4_ratio_sample':float('nan')})
    except Exception as e:
        out['summary_error']=f'{type(e).__name__}:{e}'[:200]
    return out

def record_hook_scalar_features(dst: Dict[str, Any], module_name: str, module_class: str, group: str, role: str, st: Dict[str, Any], call_count: int = 1) -> None:
    sid=sanitize_hook_name(module_name)
    g=sanitize_hook_name(group)
    r=sanitize_hook_name(role)
    base=f'feature__act_hook__{sid}'
    # fixed scalar schema independent of original tensor shape
    dst[f'{base}__available']=1
    dst[f'{base}__called']=1
    try:
        dst[f'{base}__call_count']=max(int(dst.get(f'{base}__call_count',0) or 0), int(call_count))
    except Exception:
        dst[f'{base}__call_count']=int(call_count)
    mapping={
        'finite_ratio':'finite_ratio',
        'nonfinite_count_sample':'nonfinite_count_sample',
        'absmax_sample':'absmax_sample',
        'mean_abs_sample':'mean_abs_sample',
        'std_sample':'std_sample',
        'q90_abs_sample':'q90_abs_sample',
        'q95_abs_sample':'q95_abs_sample',
        'q99_abs_sample':'q99_abs_sample',
        'zero_ratio_sample':'zero_ratio_sample',
        'large_1e3_ratio_sample':'large_1e3_ratio_sample',
        'large_1e4_ratio_sample':'large_1e4_ratio_sample',
    }
    for src, name in mapping.items():
        v=st.get(src, '')
        try:
            fv=float(v)
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        key=f'{base}__{name}'
        # For repeated calls, max aggregation is conservative for risk indicators.
        if key in dst:
            try:
                dst[key]=max(float(dst[key]), fv)
            except Exception:
                dst[key]=fv
        else:
            dst[key]=fv
    # source markers for generated consistency candidates
    try:
        av=float(st.get('absmax_sample','nan'))
        if math.isfinite(av):
            dst[f'feature__act_group_source__{g}__from__{sid}__absmax_sample']=av
            dst[f'feature__act_role_source__{r}__from__{sid}__absmax_sample']=av
    except Exception:
        pass

def load_feature_contract(path: Path) -> Dict[str, Any]:
    if path and Path(path).exists():
        return json.load(open(path, 'r', encoding='utf-8'))
    return {'selected_sources': []}

def fault_spec_from_event(ev: Dict[str, Any]):
    return helpers.FaultEventSpec(
        sample_i=int(ev.get('sample_i')), sample_id=str(ev.get('sample_id','')), event_id=int(ev.get('event_id')),
        target_id=str(ev.get('target_id','')), module_name=str(ev.get('module_name','')), module_class=str(ev.get('module_class','')),
        uniad_group=str(ev.get('uniad_group','')), target_level=str(ev.get('target_level','')),
        bit_count=int(ev.get('bit_count',1)), bit_op=str(ev.get('bit_op','flip'))
    )

def load_fault_plan_json(path: Path) -> Dict[int, Any]:
    data=json.load(open(path,'r',encoding='utf-8'))
    events=data.get('events', data if isinstance(data, list) else [])
    plan={}
    for ev in events:
        spec=fault_spec_from_event(ev)
        plan[int(spec.sample_i)]=spec
    return plan

class ExtendedActivationFeatureRecorder:
    def __init__(self, source_rows: List[Dict[str, Any]], max_values: int = 4096):
        self.source_rows=list(source_rows or [])
        self.rows_by_name={r['module_name']: r for r in self.source_rows}
        self.max_values=int(max_values)
        self.sample_i=-1
        self.sample_id=''
        self.call_counts=Counter()
        self.features={}
        self.errors=[]
    def reset_sample(self, sample_i:int, sample_id:str):
        self.sample_i=int(sample_i); self.sample_id=str(sample_id); self.call_counts=Counter(); self.features={}; self.errors=[]
    def hook_for(self, module_name: str):
        def _hook(module, inputs, output):
            self.call_counts[module_name]+=1
            r=self.rows_by_name.get(module_name,{})
            try:
                t=helpers.first_tensor(output)
                st=extended_tensor_scan_stats(t, self.max_values)
                group=fixed_infer_group(module_name, r.get('module_class', type(module).__name__), r.get('fixed_group') or r.get('inferred_group',''))
                role=fixed_infer_role(module_name, r.get('module_class', type(module).__name__), r.get('fixed_role') or r.get('inferred_role',''))
                record_hook_scalar_features(self.features, module_name, r.get('module_class', type(module).__name__), group, role, st, self.call_counts[module_name])
            except Exception as e:
                self.errors.append({'sample_i':int(self.sample_i),'module_name':module_name,'error':f'{type(e).__name__}:{e}'[:200]})
            return output
        return _hook
    def finish_sample(self):
        return dict(self.features), list(self.errors)

def register_extended_feature_hooks(model, recorder: ExtendedActivationFeatureRecorder, exclude_modules=None):
    exclude_modules=set(exclude_modules or [])
    module_map=dict(model.named_modules())
    handles=[]; missing=[]; registered=[]
    for r in recorder.source_rows:
        name=r.get('module_name','')
        if not name or name in exclude_modules:
            continue
        mod=module_map.get(name)
        if mod is None:
            missing.append(r); continue
        handles.append(mod.register_forward_hook(recorder.hook_for(name)))
        registered.append(r)
    return handles, {'registered_count':len(registered),'missing_count':len(missing),'missing':missing[:20]}


def maybe_get_baseline_l2(row: Dict[str, Any], field: str):
    d = row.get('delta', {}) if isinstance(row, dict) else {}
    if isinstance(d, dict):
        try:
            v = d.get(field, '')
            return float(v) if v != '' else None
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibration-dir', type=Path, default=DEFAULT_CAL_DIR)
    ap.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument('--out-root', type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument('--seed', type=int, default=958101)
    ap.add_argument('--method', type=str, default='hybrid', choices=['detect_only', 'clip', 'median_impute', 'hybrid'])
    ap.add_argument('--margin', type=float, default=10.0)
    ap.add_argument('--selected-mode', type=str, default='all_prefix', choices=['all_prefix', 'fault_cases'])
    ap.add_argument('--case-limit', type=int, default=6019)
    ap.add_argument('--case-ids', type=str, default='', help='Optional comma-separated sample indices. If set, selected-mode is ignored and a prefix up to max(case_ids) is replayed to preserve UniAD state.')
    ap.add_argument('--positive-ratio-min', type=float, default=0.80)
    ap.add_argument('--positive-ratio-max', type=float, default=0.85)
    ap.add_argument('--group-weights', type=str, default='U1_img_backbone_neck:10,U2_bev_encoder:20,U3_track_det_head:15,U4_map_seg_head:15,U5_motion_head:15,U6_occ_head:15,U7_planning_head:10,U9_online_track_memory:0')
    ap.add_argument('--bit-count-weights', type=str, default='1:1,2:1,3:1')
    ap.add_argument('--bit-op-weights', type=str, default='flip:1,stuck0:1,stuck1:1')
    ap.add_argument('--guard-groups', type=str, default='all')
    ap.add_argument('--stat-max-values', type=int, default=4096, help='Legacy budget. For v4.0.1, use high/low risk budgets unless sampling_strategy=firstk.')
    ap.add_argument('--median-max-values', type=int, default=4096)
    ap.add_argument('--sampling-strategy', type=str, default='typeaware', choices=['firstk', 'stratified', 'hash_stratified', 'block', 'typeaware', 'full'])
    ap.add_argument('--high-risk-sample-values', type=int, default=16384)
    ap.add_argument('--low-risk-sample-values', type=int, default=4096)
    ap.add_argument('--block-size', type=int, default=256)
    ap.add_argument('--repair-policy', type=str, default='first_only', choices=['first_only', 'all', 'budgeted_multi'])
    ap.add_argument('--max-repair-count-per-sample', type=int, default=8, help='Only used by repair_policy=budgeted_multi.')
    ap.add_argument('--high-contam-strategy', type=str, default='off', choices=['off', 'median', 'input_passthrough', 'zero', 'input_then_zero', 'module_aware', 'module_aware_aggressive'], help='Module-aware strategy for massively contaminated tensors. module_aware chooses residual-preserving or contribution-containment actions from registry group/class/name.')
    ap.add_argument('--high-contam-ratio', type=float, default=0.5, help='Abnormal-ratio threshold for high-contamination strategy.')
    ap.add_argument('--enable-acceptance-gate', action='store_true', help='Record online-style repaired trajectory validity gate fields.')
    ap.add_argument('--gate-coordinate-absmax', type=float, default=200.0)
    ap.add_argument('--gate-step-l2-max', type=float, default=20.0)
    ap.add_argument('--gate-accel-l2-max', type=float, default=40.0)
    ap.add_argument('--gate-jerk-l2-max', type=float, default=80.0)
    ap.add_argument('--write-output-detector-features', action='store_true', default=True, help='Write post-repair final-output online features to a separate feature branch. They do not include repair events or labels.')
    ap.add_argument('--feature-contract', type=Path, default=Path('/root/autodl-tmp/UniAD2/outputs/feature_contract_v4_3_1/latest/feature_contract_v4_3_1.json'), help='v4.3.1 feature contract JSON generated from v4.3.0 scan.')
    ap.add_argument('--disable-extended-hook-features', action='store_true', help='Disable v4.3.1 extra hook-level feature observer; keeps only v4.0.1 features plus registry hook-level additions.')
    ap.add_argument('--extended-feature-max-values', type=int, default=4096, help='Max sampled values per extended hook tensor, reduced on CPU for fixed scalar stats.')
    ap.add_argument('--fault-plan-json', type=Path, default=None, help='Optional existing generated_fault_plan_v4_3_1.json to reuse exactly.')
    ap.add_argument('--reuse-plan-dir', type=Path, default=None, help='Optional existing v4.0.1 seed output directory containing generated_fault_plan_v4_3_1.json.')
    ap.add_argument('--disable-escalate-full-on-trigger', action='store_true')
    ap.add_argument('--disable-online-state-clone', action='store_true')
    ap.add_argument('--progress-every', type=int, default=25)
    args = ap.parse_args()

    thresholds = load_thresholds(args.calibration_dir, args.margin)
    registry = helpers.load_registry(args.registry)

    # v4.0.1 is the direct mode: build the dataset/models once, generate the
    # seed-controlled fault plan in this run, then inject+repair+label in the
    # same sequential pass. No v3.5.1 fault-only output directory is required.
    dataset, clean_model, fault_model = helpers.build_dataset_and_models()
    dataset_len = len(dataset)

    if args.case_ids.strip():
        selected_cases = parse_case_ids(args.case_ids)
        if not selected_cases:
            raise RuntimeError('case_ids was provided but no valid indices were parsed.')
        if max(selected_cases) >= dataset_len:
            raise ValueError(f'case_ids max={max(selected_cases)} exceeds dataset_len={dataset_len}')
        n_for_plan = max(selected_cases) + 1
        effective_mode = 'case_ids'
    else:
        n_for_plan = int(args.case_limit) if int(args.case_limit) > 0 else dataset_len
        n_for_plan = min(n_for_plan, dataset_len)
        if n_for_plan <= 0:
            raise RuntimeError(f'Invalid n_for_plan={n_for_plan}')
        effective_mode = args.selected_mode

    sample_ids_for_plan = [helpers.sample_id_from_dataset(dataset, i) for i in range(n_for_plan)]
    group_weights = helpers.parse_weight_map(args.group_weights, key_cast=str, val_cast=float)
    bit_count_weights = helpers.parse_weight_map(args.bit_count_weights, key_cast=int, val_cast=float)
    bit_op_weights = helpers.parse_weight_map(args.bit_op_weights, key_cast=str, val_cast=float)
    original_plan = helpers.make_fault_plan(
        sample_ids_for_plan, n_for_plan, registry, int(args.seed),
        float(args.positive_ratio_min), float(args.positive_ratio_max),
        group_weights, bit_count_weights, bit_op_weights,
    )
    reused_fault_plan_path = ''
    plan_in = args.fault_plan_json
    if plan_in is None and args.reuse_plan_dir is not None:
        plan_in = Path(args.reuse_plan_dir) / 'generated_fault_plan_v4_3_1.json'
    if plan_in is not None and Path(plan_in).exists():
        original_plan = load_fault_plan_json(Path(plan_in))
        reused_fault_plan_path = str(plan_in)
        print(f'[v4.3.1] Reusing existing fault plan: {reused_fault_plan_path} events={len(original_plan)}', flush=True)
    original_infos: Dict[int, List[Dict[str, Any]]] = {}
    baseline_rows: Dict[int, Dict[str, Any]] = {}

    if not args.case_ids.strip():
        if args.selected_mode == 'all_prefix':
            selected_cases = list(range(n_for_plan))
        elif args.selected_mode == 'fault_cases':
            selected_cases = sorted(original_plan.keys())
        else:
            raise ValueError(f'Unsupported selected_mode for direct v4.0.1: {args.selected_mode}')
    if not selected_cases:
        raise RuntimeError('No selected cases found.')
    max_i = max(selected_cases)
    selected_set = set(selected_cases)

    group_tag = str(args.guard_groups).replace(',', '_').replace('*', 'all')
    strat_tag = f'{args.sampling_strategy}_hi{args.high_risk_sample_values}_lo{args.low_risk_sample_values}_rp{args.repair_policy}'
    if args.repair_policy == 'budgeted_multi':
        strat_tag += f'_K{int(args.max_repair_count_per_sample)}'
    if str(args.high_contam_strategy) not in {'', 'off'}:
        hc_ratio_tag = str(args.high_contam_ratio).replace('.', 'p')
        strat_tag += f'_hc{str(args.high_contam_strategy)}_r{hc_ratio_tag}'
    if args.enable_acceptance_gate:
        strat_tag += '_gate'
    out_dir = args.out_root / f'{effective_mode}_{len(selected_cases)}_seed{args.seed}_{args.method}_m{str(args.margin).replace(".", "p")}_{group_tag}_{strat_tag}_extv431'
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / 'debug_repair_events_v4_3_1.jsonl'
    rows_path = out_dir / 'debug_case_summaries_v4_3_1.jsonl'
    csv_path = out_dir / 'debug_case_summaries_v4_3_1.csv'
    activation_features_path = out_dir / 'detector_activation_features_v4_3_1.jsonl'
    output_features_path = out_dir / 'detector_output_features_v4_3_1.jsonl'
    labels_path = out_dir / 'repair_outcome_labels_v4_3_1.jsonl'
    debug_meta_path = out_dir / 'debug_metadata_v4_3_1.jsonl'
    train_activation_csv = out_dir / 'detector_train_activation_only_v4_3_1.csv'
    train_activation_output_csv = out_dir / 'detector_train_activation_plus_output_v4_3_1.csv'
    plan_path = out_dir / 'generated_fault_plan_v4_3_1.json'
    for p in [events_path, rows_path, csv_path, activation_features_path, output_features_path, labels_path, debug_meta_path, train_activation_csv, train_activation_output_csv]:
        if p.exists():
            p.unlink()

    plan_json = {
        'version': VERSION,
        'mode': 'direct_generate_fault_then_repair_with_extended_feature_logging',
        'seed': int(args.seed),
        'dataset_len': int(dataset_len),
        'sample_count_for_plan': int(n_for_plan),
        'positive_count': int(len(original_plan)),
        'positive_ratio': float(len(original_plan) / max(n_for_plan, 1)),
        'positive_ratio_min': float(args.positive_ratio_min),
        'positive_ratio_max': float(args.positive_ratio_max),
        'group_weights': group_weights,
        'bit_count_weights': bit_count_weights,
        'bit_op_weights': bit_op_weights,
        'events': [helpers.asdict(v) for _, v in sorted(original_plan.items())],
        'notes': [
            'This plan is generated or reused by v4.3.1. It is saved for reproducibility/debug only.',
            'Fault metadata from this plan must not be used as detector features.',
        ],
    }
    write_json(plan_path, plan_json)

    write_json(out_dir / 'selected_cases_v4_3_1.json', {
        'version': VERSION,
        'selected_mode': effective_mode,
        'case_limit': args.case_limit,
        'case_ids': selected_cases,
        'max_replay_sample_i': max_i,
        'method': args.method,
        'margin': args.margin,
        'guard_groups': args.guard_groups,
        'sampling_strategy': args.sampling_strategy,
        'high_risk_sample_values': args.high_risk_sample_values,
        'low_risk_sample_values': args.low_risk_sample_values,
        'repair_policy': args.repair_policy,
        'max_repair_count_per_sample': int(args.max_repair_count_per_sample),
        'high_contam_strategy': args.high_contam_strategy,
        'high_contam_ratio': float(args.high_contam_ratio),
        'enable_acceptance_gate': int(bool(args.enable_acceptance_gate)),
        'generated_fault_plan': str(plan_path),
        'reused_fault_plan_path': reused_fault_plan_path,
        'feature_contract': str(args.feature_contract),
        'notes': [
            'v4.0.1 directly generates the fault plan and applies repair in the same inference.',
            'Repair uses clean-calibrated numerical contracts and current tensor context only.',
            'Detector activation features are collected before repair at each hook and do not include repair actions or labels.',
        ],
    })

    groups_filter = parse_group_filter(args.guard_groups)
    runtime = FaultRepairRuntime(
        registry, thresholds, args.method,
        groups_filter=groups_filter,
        stat_max_values=args.stat_max_values,
        median_max_values=args.median_max_values,
        sampling_strategy=args.sampling_strategy,
        high_risk_sample_values=args.high_risk_sample_values,
        low_risk_sample_values=args.low_risk_sample_values,
        block_size=args.block_size,
        escalate_full_on_trigger=(not args.disable_escalate_full_on_trigger),
        repair_policy=args.repair_policy,
        max_repair_count_per_sample=args.max_repair_count_per_sample,
        high_contam_strategy=args.high_contam_strategy,
        high_contam_ratio=args.high_contam_ratio,
        enable_acceptance_gate=args.enable_acceptance_gate,
        gate_coordinate_absmax=args.gate_coordinate_absmax,
        gate_step_l2_max=args.gate_step_l2_max,
        gate_accel_l2_max=args.gate_accel_l2_max,
        gate_jerk_l2_max=args.gate_jerk_l2_max,
        extended_feature_max_values=args.extended_feature_max_values,
    )
    feature_contract = load_feature_contract(args.feature_contract)
    selected_sources = feature_contract.get('selected_sources', []) if isinstance(feature_contract, dict) else []
    extra_recorder = ExtendedActivationFeatureRecorder(selected_sources, max_values=args.extended_feature_max_values)
    handles, hook_status = register_repair_hooks(fault_model, runtime)
    if hook_status['registered_count'] <= 0:
        raise RuntimeError('No repair hooks registered on fault_model.')
    extra_hook_status = {'registered_count': 0, 'missing_count': 0, 'missing': []}
    extra_handles = []
    if (not args.disable_extended_hook_features) and selected_sources:
        extra_handles, extra_hook_status = register_extended_feature_hooks(fault_model, extra_recorder, exclude_modules=set(runtime.target_by_module.keys()))
        handles.extend(extra_handles)
    print(f'[v4.3.1] repair_hooks={hook_status["registered_count"]} extra_feature_hooks={extra_hook_status.get("registered_count",0)} missing_extra={extra_hook_status.get("missing_count",0)}', flush=True)

    counters = Counter()
    case_rows: List[Dict[str, Any]] = []
    train_activation_rows: List[Dict[str, Any]] = []
    train_activation_output_rows: List[Dict[str, Any]] = []
    feature_key_list: List[str] = []
    start_all = time.time()

    with torch.no_grad():
        for i in tqdm(range(max_i + 1), desc=f'UniAD2 repair {args.method}', dynamic_ncols=True):
            sample_id = helpers.sample_id_from_dataset(dataset, i)
            selected = i in selected_set
            sample_start = time.time()

            # Match prior exact replay: copy clean pre-sample online state to fault model, run fault path,
            # then advance clean path after fault path.
            state_copy_info = {'disabled': 1}
            if not args.disable_online_state_clone:
                state_copy_info = helpers.copy_online_state(clean_model, fault_model)

            spec = original_plan.get(i)
            exact_infos = original_infos.get(i, [])
            runtime.reset_sample(i, sample_id, spec, exact_infos, int(args.seed))
            extra_recorder.reset_sample(i, sample_id)

            data_fault = helpers.scatter_one(dataset[i])
            try:
                fault_result = fault_model(return_loss=False, rescale=True, **data_fault)
                fault_error = None
            except Exception:
                fault_result = None
                fault_error = traceback.format_exc()
            fault_infos, repair_events, call_counts, final_trace_order, detector_activation_features = runtime.finish_sample()
            extra_feature_values, extra_feature_errors = extra_recorder.finish_sample()
            if extra_feature_values:
                # extra hook observer records online activation scalar stats only; no repair/fault/label metadata.
                merged = dict(extra_feature_values)
                merged.update(detector_activation_features)
                detector_activation_features = merged

            # Advance clean model after fault path, consistent with v3.5.2b exact replay.
            data_clean = helpers.scatter_one(dataset[i])
            try:
                clean_result = clean_model(return_loss=False, rescale=True, **data_clean)
                clean_error = None
            except Exception:
                clean_result = None
                clean_error = traceback.format_exc()

            if selected:
                for ev in repair_events:
                    append_jsonl(events_path, ev)

                clean_traj, clean_traj_path = helpers.extract_planning_traj(clean_result) if clean_result is not None else (None, '')
                repaired_traj, repaired_traj_path = helpers.extract_planning_traj(fault_result) if fault_result is not None else (None, '')
                delta = helpers.traj_l2_delta(clean_traj, repaired_traj)
                clean_health = helpers.traj_health(clean_traj, clean_error, 'clean')
                repaired_health = helpers.traj_health(repaired_traj, fault_error, 'fault')
                output_flags = helpers.combine_output_illegal(clean_health, repaired_health)
                gate_fields = trajectory_acceptance_gate(
                    repaired_traj, output_flags,
                    coord_absmax=args.gate_coordinate_absmax,
                    step_l2_max=args.gate_step_l2_max,
                    accel_l2_max=args.gate_accel_l2_max,
                    jerk_l2_max=args.gate_jerk_l2_max,
                ) if args.enable_acceptance_gate else {}
                repair_summary = summarize_repairs(repair_events)
                first_det_summary = first_detection_fields(repair_events, fault_infos, final_trace_order)
                # Direct v4.0.1 has no separate fault-only baseline row.
                # Labels are built from clean-vs-repaired output in this same run.
                base_fields = {}
                repair_hflags = harmful_flags(delta, output_flags)
                detector_output_features = trajectory_online_features(repaired_traj, prefix='feature__post_out_') if args.write_output_detector_features else {}
                labels = build_post_repair_labels(delta, output_flags, gate_fields)

                # Strict branch separation:
                #   detector_activation_features: pre-repair hook observations only.
                #   detector_output_features: final output observations only, no repair event/gate/label fields.
                #   labels: offline post-repair outcome labels from clean-vs-repair and online-style gate.
                #   debug metadata: fault/repair internals for analysis only.
                activation_row = {'sample_i': int(i), 'sample_id': sample_id, **detector_activation_features}
                output_row = {'sample_i': int(i), 'sample_id': sample_id, **detector_output_features}
                label_row = {'sample_i': int(i), 'sample_id': sample_id, **labels}
                debug_row = {
                    'sample_i': int(i), 'sample_id': sample_id,
                    'debug__target_present': int(i in original_plan),
                    'debug__target_module': spec.module_name if spec else '',
                    'debug__target_module_class': spec.module_class if spec else '',
                    'debug__target_uniad_group': spec.uniad_group if spec else '',
                    'debug__target_level': spec.target_level if spec else '',
                    'debug__event_id': int(spec.event_id) if spec else '',
                    'debug__bit_count': int(spec.bit_count) if spec else '',
                    'debug__bit_op': spec.bit_op if spec else '',
                    'debug__fault_applied_count': int(sum(int(x.get('applied', 0) or 0) for x in fault_infos)),
                    'debug__fault_effective_count': int(sum(int(x.get('effective', 0) or 0) for x in fault_infos)),
                    'debug__exact_replay_used_count': int(sum(int(x.get('exact_replay_used', 0) or 0) for x in fault_infos)),
                    'debug__extra_feature_error_count': int(len(extra_feature_errors)),
                }
                debug_row.update({f'debug__{k}': v for k, v in summarize_repairs(repair_events).items()})
                debug_row.update({f'debug__{k}': v for k, v in first_detection_fields(repair_events, fault_infos, final_trace_order).items()})

                append_jsonl(activation_features_path, activation_row)
                append_jsonl(output_features_path, output_row)
                append_jsonl(labels_path, label_row)
                append_jsonl(debug_meta_path, debug_row)

                train_a = dict(activation_row)
                train_a.update(label_row)
                train_ao = dict(train_a)
                train_ao.update(output_row)
                train_activation_rows.append(train_a)
                train_activation_output_rows.append(train_ao)
                feature_key_list.extend([k for k in activation_row.keys() if k.startswith('feature__')])
                feature_key_list.extend([k for k in output_row.keys() if k.startswith('feature__')])

                row = {
                    'version': VERSION,
                    'sample_i': int(i),
                    'sample_id': sample_id,
                    'elapsed_sec': time.time() - sample_start,
                    'method': args.method,
                    'margin': float(args.margin),
                    'guard_groups': args.guard_groups,
                    'selected_mode': args.selected_mode,
                    'sampling_strategy': args.sampling_strategy,
                    'high_risk_sample_values': int(args.high_risk_sample_values),
                    'low_risk_sample_values': int(args.low_risk_sample_values),
                    'repair_policy': args.repair_policy,
                    'max_repair_count_per_sample': int(args.max_repair_count_per_sample),
                    'high_contam_strategy': args.high_contam_strategy,
                    'high_contam_ratio': float(args.high_contam_ratio),
                    'enable_acceptance_gate': int(bool(args.enable_acceptance_gate)),
                    'state_copy_copied_count': state_copy_info.get('copied_count', ''),
                    'clean_forward_error': clean_error,
                    'repaired_forward_error': fault_error,
                    'clean_traj_path': clean_traj_path,
                    'repaired_traj_path': repaired_traj_path,
                    'target_present': int(i in original_plan),
                    'target_module': spec.module_name if spec else '',
                    'target_module_class': spec.module_class if spec else '',
                    'target_uniad_group': spec.uniad_group if spec else '',
                    'target_level': spec.target_level if spec else '',
                    'event_id': int(spec.event_id) if spec else '',
                    'bit_count': int(spec.bit_count) if spec else '',
                    'bit_op': spec.bit_op if spec else '',
                    'fault_applied_count': int(sum(int(x.get('applied', 0) or 0) for x in fault_infos)),
                    'fault_effective_count': int(sum(int(x.get('effective', 0) or 0) for x in fault_infos)),
                    'exact_replay_used_count': int(sum(int(x.get('exact_replay_used', 0) or 0) for x in fault_infos)),
                    'repaired_output_illegal_or_missing': int(output_flags.get('output_illegal_or_missing', 0) or 0),
                    'repaired_output_missing': int(output_flags.get('output_missing', 0) or 0),
                    'repaired_output_nonfinite': int(output_flags.get('output_nonfinite', 0) or 0),
                    'repaired_output_shape_invalid': int(output_flags.get('output_shape_invalid', 0) or 0),
                    'repaired_traj_delta_available': delta.get('traj_delta_available', ''),
                    'repaired_traj_delta_reason': delta.get('traj_delta_reason', ''),
                    'repaired_traj_delta_l2_mean': delta.get('traj_delta_l2_mean', ''),
                    'repaired_traj_delta_l2_max': delta.get('traj_delta_l2_max', ''),
                    'repaired_traj_delta_l2_final': delta.get('traj_delta_l2_final', ''),
                    **repair_summary,
                    **first_det_summary,
                    **repair_hflags,
                    **gate_fields,
                    **base_fields,
                }
                row['direct_fault_repair_mode'] = 1
                row['fault_only_baseline_available'] = 0

                append_jsonl(rows_path, row)
                case_rows.append(row)

                counters['selected_cases_done'] += 1
                counters['repaired_output_nonfinite'] += int(row.get('repaired_output_nonfinite', 0) or 0)
                counters['repair_event_count'] += int(row.get('repair_event_count', 0) or 0)
                counters['repair_modified_count'] += int(row.get('repair_modified_count', 0) or 0)
                counters['gate_accept_count'] += int(row.get('gate_accept', 0) or 0)
                counters['gate_reject_count'] += int(row.get('gate_reject', 0) or 0)
                counters['fault_applied_count'] += int(row.get('fault_applied_count', 0) or 0)
                counters['fault_effective_count'] += int(row.get('fault_effective_count', 0) or 0)
                counters['extra_feature_error_count'] += int(len(extra_feature_errors))

            counters['prefix_samples_processed'] += 1
            if args.progress_every and (i + 1) % int(args.progress_every) == 0:
                print(f'[{VERSION}] progress prefix {i+1}/{max_i+1} selected_done={counters["selected_cases_done"]}/{len(selected_cases)} method={args.method}', flush=True)
            if (i + 1) % 25 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if train_activation_rows:
        train_fields = sorted(set().union(*(r.keys() for r in train_activation_rows)))
        write_csv(train_activation_csv, train_activation_rows, train_fields)
    if train_activation_output_rows:
        train_fields2 = sorted(set().union(*(r.keys() for r in train_activation_output_rows)))
        write_csv(train_activation_output_csv, train_activation_output_rows, train_fields2)
    leak_audit = leakage_audit_feature_keys(feature_key_list)
    write_json(out_dir / 'leakage_audit_v4_3_1.json', leak_audit)

    fieldnames = sorted({k for r in case_rows for k in r.keys()})
    if case_rows:
        write_csv(csv_path, case_rows, fieldnames)

    summary = {
        'version': VERSION,
        'out_dir': str(out_dir),
        'calibration_dir': str(args.calibration_dir),
        'seed': int(args.seed),
        'method': args.method,
        'margin': float(args.margin),
        'guard_groups': args.guard_groups,
        'selected_mode': args.selected_mode,
        'sampling_strategy': args.sampling_strategy,
        'high_risk_sample_values': int(args.high_risk_sample_values),
        'low_risk_sample_values': int(args.low_risk_sample_values),
        'repair_policy': args.repair_policy,
        'max_repair_count_per_sample': int(args.max_repair_count_per_sample),
        'high_contam_strategy': args.high_contam_strategy,
        'high_contam_ratio': float(args.high_contam_ratio),
        'enable_acceptance_gate': int(bool(args.enable_acceptance_gate)),
        'gate_coordinate_absmax': float(args.gate_coordinate_absmax),
        'gate_step_l2_max': float(args.gate_step_l2_max),
        'gate_accel_l2_max': float(args.gate_accel_l2_max),
        'gate_jerk_l2_max': float(args.gate_jerk_l2_max),
        'escalate_full_on_trigger': int(not args.disable_escalate_full_on_trigger),
        'case_count_requested': len(selected_cases),
        'dataset_len': int(dataset_len),
        'sample_count_for_plan': int(n_for_plan),
        'generated_fault_positive_count': int(len(original_plan)),
        'generated_fault_positive_ratio': float(len(original_plan) / max(n_for_plan, 1)),
        'case_ids': selected_cases,
        'max_replay_sample_i': max_i,
        'elapsed_sec': time.time() - start_all,
        'counters': dict(counters),
        'hook_status': hook_status,
        'extra_hook_status': extra_hook_status,
        'feature_contract': str(args.feature_contract),
        'reused_fault_plan_path': reused_fault_plan_path,
        'extended_feature_max_values': int(args.extended_feature_max_values),
        'threshold_count': len(thresholds),
        'leakage_audit': leak_audit,
        'files': {
            'selected_cases': str(out_dir / 'selected_cases_v4_3_1.json'),
            'generated_fault_plan': str(plan_path),
        'reused_fault_plan_path': reused_fault_plan_path,
        'feature_contract': str(args.feature_contract),
            'repair_events': str(events_path),
            'case_summaries_jsonl': str(rows_path),
            'case_summaries_csv': str(csv_path),
            'detector_activation_features': str(activation_features_path),
            'detector_output_features': str(output_features_path),
            'repair_outcome_labels': str(labels_path),
            'debug_metadata': str(debug_meta_path),
            'detector_train_activation_only_csv': str(train_activation_csv),
            'detector_train_activation_plus_output_csv': str(train_activation_output_csv),
            'leakage_audit': str(out_dir / 'leakage_audit_v4_3_1.json'),
            'summary': str(out_dir / 'fault_repair_extended_feature_dataset_summary_v4_3_1.json'),
        },
        'notes': [
            'v4.3.1 reuses or generates the fault plan; fault metadata is saved only for reproducibility/debug.',
            'The repair decision does not use fault metadata, bit position, clean reference, GT, or harmful labels.',
            'v4.3.1 integrates direct fault generation/reuse, repair, extended hook-level online feature logging, post-repair labels, and leakage-audited train tables in one inference.',
        ],
    }
    write_json(out_dir / 'fault_repair_extended_feature_dataset_summary_v4_3_1.json', summary)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
