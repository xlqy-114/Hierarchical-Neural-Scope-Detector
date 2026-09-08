#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm import tqdm
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

REPO = Path('/root/autodl-tmp/UniAD2/code/UniAD')
CFG_PATH = REPO / 'projects/configs/stage2_e2e/base_e2e.py'
CKPT_PATH = REPO / 'ckpts/uniad_base_e2e.pth'
DEFAULT_REGISTRY = Path('/root/autodl-tmp/UniAD2/outputs/uniad2_hook_registry_v3_5_1/uniad2_hook_registry_v3_5_1.csv')
DEFAULT_OUT_ROOT = Path('/root/autodl-tmp/UniAD2/outputs/fault_inference_v3_5_1')
VERSION = 'uniad2_nan_replay_trace_v3_5_2b_1_exact'

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

@dataclass
class FaultEventSpec:
    sample_i: int
    sample_id: str
    event_id: int
    target_id: str
    module_name: str
    module_class: str
    uniad_group: str
    target_level: str
    bit_count: int
    bit_op: str


def json_default(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return str(x)


def write_jsonl(path: Path, rows):
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=json_default) + '\n')


def append_jsonl(path: Path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, default=json_default) + '\n')


def sanitize_dataset_cfg(cfg_node):
    if isinstance(cfg_node, list):
        for item in cfg_node:
            sanitize_dataset_cfg(item)
        return cfg_node
    if not isinstance(cfg_node, dict):
        return cfg_node
    for k in ['samples_per_gpu', 'workers_per_gpu', 'persistent_workers', 'shuffle', 'drop_last']:
        if k in cfg_node:
            cfg_node.pop(k, None)
    if 'dataset' in cfg_node:
        sanitize_dataset_cfg(cfg_node['dataset'])
    if 'datasets' in cfg_node:
        sanitize_dataset_cfg(cfg_node['datasets'])
    return cfg_node


def ensure_map_links():
    nus = Path('/root/autodl-tmp/UniAD2/data/nuscenes')
    maps = nus / 'maps'
    maps.mkdir(parents=True, exist_ok=True)
    for name in ['expansion', 'basemap', 'prediction']:
        src = nus / name
        dst = maps / name
        if src.exists() and not dst.exists():
            os.symlink(f'../{name}', dst)
    required = [
        maps / 'expansion/boston-seaport.json',
        maps / 'expansion/singapore-hollandvillage.json',
        maps / 'expansion/singapore-onenorth.json',
        maps / 'expansion/singapore-queenstown.json',
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing map expansion files:\n' + '\n'.join(missing))


def load_registry(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open('r', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'Empty registry: {path}')
    return rows


def weighted_choice(rng: random.Random, items: List[Any], weights: List[float]):
    total = float(sum(weights))
    if total <= 0:
        return rng.choice(items)
    x = rng.random() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += float(w)
        if x <= acc:
            return item
    return items[-1]


def parse_weight_map(text: str, key_cast=str, val_cast=float):
    out = {}
    for part in str(text or '').split(','):
        part = part.strip()
        if not part:
            continue
        if ':' not in part:
            raise ValueError(f'Invalid weighted map item: {part}')
        k, v = part.split(':', 1)
        out[key_cast(k.strip())] = val_cast(v.strip())
    return out


def sample_id_from_dataset(dataset, idx: int) -> str:
    for attr in ['data_infos', 'infos']:
        infos = getattr(dataset, attr, None)
        if isinstance(infos, list) and idx < len(infos):
            info = infos[idx]
            if isinstance(info, dict):
                for k in ['token', 'sample_token', 'sample_idx', 'scene_token']:
                    if k in info:
                        return str(info[k])
    return f'idx_{idx:06d}'


def dtype_exp_info_torch(dtype: torch.dtype) -> Tuple[int, int]:
    if dtype == torch.float16:
        return (10, 5)
    if dtype == torch.bfloat16:
        return (7, 8)
    if dtype == torch.float32:
        return (23, 8)
    if dtype == torch.float64:
        return (52, 11)
    raise TypeError(f'Unsupported torch dtype for bit fault: {dtype}')


def apply_uint_mask_signed(x: int, bits_total: int) -> int:
    sign_limit = 1 << (bits_total - 1)
    full = 1 << bits_total
    return x - full if x >= sign_limit else x


def enumerate_exponent_spans(lsb_start: int, exp_bits: int, width: int) -> List[List[int]]:
    return [list(range(start, start + width)) for start in range(lsb_start, lsb_start + exp_bits - width + 1)]


def valid_spans_for_uint(uint: int, mantissa_bits: int, exp_bits: int, width: int, bit_op: str) -> List[List[int]]:
    spans = enumerate_exponent_spans(mantissa_bits, exp_bits, width)
    valid = []
    for span in spans:
        changed = False
        for bit in span:
            cur = (uint >> bit) & 1
            if bit_op == 'flip':
                changed = True
                break
            if bit_op == 'stuck0' and cur == 1:
                changed = True
                break
            if bit_op == 'stuck1' and cur == 0:
                changed = True
                break
        if changed:
            valid.append(span)
    return valid


def scalar_uint_info_torch(flat_tensor: torch.Tensor, flat_idx: int) -> Tuple[int, int, int, int]:
    if flat_tensor.dtype == torch.float16:
        bits_total = 16
        intval = int(flat_tensor.view(torch.int16)[flat_idx].item())
    elif flat_tensor.dtype == torch.bfloat16:
        bits_total = 16
        intval = int(flat_tensor.view(torch.int16)[flat_idx].item())
    elif flat_tensor.dtype == torch.float32:
        bits_total = 32
        intval = int(flat_tensor.view(torch.int32)[flat_idx].item())
    elif flat_tensor.dtype == torch.float64:
        bits_total = 64
        intval = int(flat_tensor.view(torch.int64)[flat_idx].item())
    else:
        raise TypeError(f'Unsupported dtype {flat_tensor.dtype}')
    mantissa_bits, exp_bits = dtype_exp_info_torch(flat_tensor.dtype)
    uint = intval & ((1 << bits_total) - 1)
    return uint, bits_total, mantissa_bits, exp_bits


def mutate_one_scalar_tensor_exponent_bits_(flat_tensor: torch.Tensor, flat_idx: int, targets: List[int], bit_op: str):
    orig = flat_tensor[flat_idx].item()
    uint, bits_total, _, _ = scalar_uint_info_torch(flat_tensor, flat_idx)
    for bit in targets:
        mask = 1 << bit
        if bit_op == 'flip':
            uint ^= mask
        elif bit_op == 'stuck0':
            uint &= ~mask
        elif bit_op == 'stuck1':
            uint |= mask
        else:
            raise ValueError(f'Unsupported bit_op {bit_op}')
    new_int = apply_uint_mask_signed(uint, bits_total)
    if bits_total == 16:
        flat_tensor.view(torch.int16)[flat_idx] = new_int
    elif bits_total == 32:
        flat_tensor.view(torch.int32)[flat_idx] = new_int
    else:
        flat_tensor.view(torch.int64)[flat_idx] = new_int
    new_val = flat_tensor[flat_idx].item()
    return {
        'flat_idx': int(flat_idx),
        'bit_positions_lsb': [int(b) for b in targets],
        'before': safe_float(orig),
        'after': safe_float(new_val),
    }


def safe_float(x, default=None):
    try:
        y = float(x)
        if math.isnan(y):
            return 'nan'
        if math.isinf(y):
            return 'inf' if y > 0 else '-inf'
        return y
    except Exception:
        return default


def effective_value_changed(before, after) -> bool:
    try:
        b = float(before)
        a = float(after)
        if not (math.isfinite(b) and math.isfinite(a)):
            return str(before) != str(after)
        return abs(a - b) > 0
    except Exception:
        return str(before) != str(after)


def first_tensor(obj: Any):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            t = first_tensor(item)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for item in obj.values():
            t = first_tensor(item)
            if t is not None:
                return t
    return None


def set_first_tensor(obj: Any, new_tensor: torch.Tensor):
    if torch.is_tensor(obj):
        return new_tensor, True
    if isinstance(obj, tuple):
        items = list(obj)
        for i, item in enumerate(items):
            new_item, ok = set_first_tensor(item, new_tensor)
            if ok:
                items[i] = new_item
                return tuple(items), True
        return obj, False
    if isinstance(obj, list):
        items = list(obj)
        for i, item in enumerate(items):
            new_item, ok = set_first_tensor(item, new_tensor)
            if ok:
                items[i] = new_item
                return items, True
        return obj, False
    if isinstance(obj, dict):
        d = dict(obj)
        for k, item in obj.items():
            new_item, ok = set_first_tensor(item, new_tensor)
            if ok:
                d[k] = new_item
                return d, True
        return obj, False
    return obj, False


def tensor_stats(t: Any, prefix: str, max_values: int = 2048) -> Dict[str, float]:
    out = {}
    if t is None or not torch.is_tensor(t):
        out[f'{prefix}__available'] = 0.0
        return out
    out[f'{prefix}__available'] = 1.0
    out[f'{prefix}__numel'] = float(t.numel())
    out[f'{prefix}__ndim'] = float(t.ndim)
    with torch.no_grad():
        try:
            x = t.detach()
            if x.numel() == 0:
                out[f'{prefix}__finite_ratio'] = 1.0
                return out
            flat = x.reshape(-1)
            sample = flat[:min(int(max_values), int(flat.numel()))].float()
            finite = torch.isfinite(sample)
            out[f'{prefix}__finite_ratio'] = float(finite.float().mean().detach().cpu().item())
            if bool(finite.any().item()):
                vals = sample[finite]
                out[f'{prefix}__mean'] = float(vals.mean().detach().cpu().item())
                out[f'{prefix}__std'] = float(vals.std(unbiased=False).detach().cpu().item()) if vals.numel() > 1 else 0.0
                out[f'{prefix}__absmax'] = float(vals.abs().max().detach().cpu().item())
                out[f'{prefix}__norm2_sample'] = float(torch.linalg.vector_norm(vals).detach().cpu().item())
            else:
                out[f'{prefix}__mean'] = math.nan
                out[f'{prefix}__std'] = math.nan
                out[f'{prefix}__absmax'] = math.nan
                out[f'{prefix}__norm2_sample'] = math.nan
        except Exception:
            out[f'{prefix}__summary_error'] = 1.0
    return out


def np_stats(arr: Any, prefix: str) -> Dict[str, float]:
    out = {}
    try:
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        out[f'{prefix}__available'] = 1.0
        out[f'{prefix}__numel'] = float(a.size)
        if a.size == 0:
            out[f'{prefix}__finite_ratio'] = 1.0
            return out
        finite = np.isfinite(a)
        out[f'{prefix}__finite_ratio'] = float(finite.mean())
        if finite.any():
            vals = a[finite]
            out[f'{prefix}__mean'] = float(vals.mean())
            out[f'{prefix}__std'] = float(vals.std())
            out[f'{prefix}__absmax'] = float(np.abs(vals).max())
            out[f'{prefix}__norm2'] = float(np.linalg.norm(vals))
    except Exception:
        out[f'{prefix}__available'] = 0.0
        out[f'{prefix}__summary_error'] = 1.0
    return out


def mutate_activation_clone(t: torch.Tensor, spec: FaultEventSpec, rng: random.Random, max_attempts: int = 200):
    info = {
        'event_id': int(spec.event_id),
        'target_id': spec.target_id,
        'module_name': spec.module_name,
        'module_class': spec.module_class,
        'uniad_group': spec.uniad_group,
        'target_level': spec.target_level,
        'bit_count': int(spec.bit_count),
        'bit_op': spec.bit_op,
        'field': 'activation_exponent',
        'applied': 0,
        'effective': 0,
    }
    if t is None or not torch.is_tensor(t):
        info['reason'] = 'no_tensor_output'
        return t, info
    if not torch.is_floating_point(t):
        info['reason'] = f'not_floating_dtype:{t.dtype}'
        return t, info
    if t.numel() <= 0:
        info['reason'] = 'empty_tensor'
        return t, info

    out = t.clone()
    flat = out.reshape(-1)
    orig_flat = t.reshape(-1)
    numel = int(flat.numel())
    tried = set()
    tries = 0
    while tries < max_attempts and len(tried) < numel:
        flat_idx = int(rng.randrange(numel))
        if flat_idx in tried:
            continue
        tried.add(flat_idx)
        tries += 1
        try:
            val = flat[flat_idx]
            if not bool(torch.isfinite(val.detach().float()).item()):
                continue
            uint, _, mantissa_bits, exp_bits = scalar_uint_info_torch(flat, flat_idx)
            spans = valid_spans_for_uint(uint, mantissa_bits, exp_bits, int(spec.bit_count), str(spec.bit_op))
        except Exception as e:
            info.update({'reason': f'span_error:{type(e).__name__}:{e}', 'attempts': tries})
            return t, info
        rng.shuffle(spans)
        for span in spans:
            before = safe_float(flat[flat_idx].detach().float().cpu().item())
            try:
                fi = mutate_one_scalar_tensor_exponent_bits_(flat, flat_idx, span, str(spec.bit_op))
            except Exception:
                continue
            after = safe_float(flat[flat_idx].detach().float().cpu().item())
            if effective_value_changed(before, after):
                info.update(fi)
                info.update({
                    'applied': 1,
                    'effective': 1,
                    'reason': 'ok',
                    'attempts': tries,
                    'dtype': str(t.dtype),
                    'shape': list(t.shape),
                    'replace_output_ok': 1,
                })
                return out, info
            flat[flat_idx] = orig_flat[flat_idx]
    info.update({'reason': 'no_effective_span_found', 'attempts': tries, 'dtype': str(t.dtype), 'shape': list(t.shape)})
    return t, info


class UniADHookRuntime:
    def __init__(self, registry_rows: List[Dict[str, str]], max_feature_hooks: int = 256):
        self.registry_rows = list(registry_rows)
        self.target_by_module = {r['module_name']: r for r in self.registry_rows}
        self.current_sample_id = ''
        self.pending: Optional[FaultEventSpec] = None
        self.base_seed = 0
        self.consumed_event_ids = set()
        self.fault_infos: List[Dict[str, Any]] = []
        self.feature_row: Dict[str, float] = {}
        self.feature_hook_count = 0
        self.max_feature_hooks = int(max_feature_hooks)
        self.call_counts = Counter()

    def reset_sample(self, sample_id: str, pending: Optional[FaultEventSpec], base_seed: int):
        self.current_sample_id = str(sample_id)
        self.pending = pending
        self.base_seed = int(base_seed)
        self.consumed_event_ids = set()
        self.fault_infos = []
        self.feature_row = {}
        self.feature_hook_count = 0
        self.call_counts = Counter()

    def hook_for(self, module_name: str):
        target = self.target_by_module[module_name]
        target_id = target['target_id']
        group = target['uniad_group']
        level = target['target_level']
        short = target_id

        def _hook(module, inputs, output):
            self.call_counts[module_name] += 1
            # Online feature collection from this same fault inference path.
            if self.feature_hook_count < self.max_feature_hooks:
                t = first_tensor(output)
                stats = tensor_stats(t, f'hook__{short}')
                self.feature_row.update(stats)
                self.feature_row[f'hook__{short}__called'] = 1.0
                self.feature_row[f'hook__{short}__call_count'] = float(self.call_counts[module_name])
                self.feature_hook_count += 1

            spec = self.pending
            if spec is None:
                return output
            if spec.module_name != module_name:
                return output
            if int(spec.event_id) in self.consumed_event_ids:
                return output
            self.consumed_event_ids.add(int(spec.event_id))
            before_t = first_tensor(output)
            rng = random.Random(sample_seed(self.base_seed, self.current_sample_id, int(spec.event_id)))
            new_t, info = mutate_activation_clone(before_t, spec, rng)
            info['call_count_at_injection'] = int(self.call_counts[module_name])
            replaced, ok = set_first_tensor(output, new_t)
            info['replace_output_ok'] = int(ok)
            if not ok:
                info['applied'] = 0
                info['effective'] = 0
                info['reason'] = info.get('reason', 'replace_output_failed')
            self.fault_infos.append(info)
            return replaced if ok else output
        return _hook

    def finish_sample(self):
        return list(self.fault_infos), dict(self.feature_row)


def sample_seed(base_seed: int, sample_id: str, event_id: int) -> int:
    text = f'{int(base_seed)}::{sample_id}::{int(event_id)}'
    # deterministic, stable across Python runs
    h = 2166136261
    for ch in text.encode('utf-8'):
        h ^= int(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def clone_obj(x):
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: clone_obj(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clone_obj(v) for v in x]
    if isinstance(x, tuple):
        return tuple(clone_obj(v) for v in x)
    try:
        return copy.deepcopy(x)
    except Exception:
        return x


STATE_ATTR_ALLOW = {
    'prev_frame_info', 'test_track_instances', 'track_instances', 'track_base',
    'prev_bev', 'prev_pos', 'prev_angle', 'scene_token', 'frame_idx'
}


def copy_online_state(src_model, dst_model):
    copied = []
    errors = []
    for (src_name, src_mod), (dst_name, dst_mod) in zip(src_model.named_modules(), dst_model.named_modules()):
        if src_name != dst_name:
            errors.append({'reason': 'module_name_mismatch', 'src': src_name, 'dst': dst_name})
            break
        for attr in STATE_ATTR_ALLOW:
            if hasattr(src_mod, attr):
                try:
                    setattr(dst_mod, attr, clone_obj(getattr(src_mod, attr)))
                    copied.append(f'{src_name}.{attr}' if src_name else attr)
                except Exception as e:
                    errors.append({'module': src_name, 'attr': attr, 'error': repr(e)})
    return {'copied_count': len(copied), 'copied_attrs': copied[:100], 'errors': errors[:20]}


def maybe_reset_uniad_state(model):
    # Keep minimal reset only at run start. Sequential clean model then controls online state.
    if hasattr(model, 'prev_frame_info'):
        try:
            model.prev_frame_info = {
                'prev_bev': None,
                'scene_token': None,
                'prev_pos': 0,
                'prev_angle': 0,
            }
        except Exception:
            pass


def scatter_one(data):
    batch = collate([data], samples_per_gpu=1)
    if torch.cuda.is_available():
        batch = scatter(batch, [0])[0]
    return batch


def extract_nested(obj, path: List[Any]):
    cur = obj
    for p in path:
        if isinstance(p, int):
            if not isinstance(cur, (list, tuple)) or len(cur) <= p:
                return None
            cur = cur[p]
        else:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
    return cur


def to_numpy_array(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return None


def extract_planning_traj(result):
    # UniAD result is list[dict]. Field observed in v3.4.1 scan.
    candidates = [
        [0, 'planning', 'result_planning', 'sdc_traj'],
        [0, 'planning', 'result_planning', 'sdc_traj_all'],
        [0, 'traj'],
        ['planning', 'result_planning', 'sdc_traj'],
    ]
    for path in candidates:
        x = extract_nested(result, path)
        arr = to_numpy_array(x)
        if arr is None:
            continue
        return arr, '.'.join(str(p) for p in path)
    return None, ''


def extract_output_features(result) -> Dict[str, float]:
    row: Dict[str, float] = {}
    if isinstance(result, list) and result:
        r0 = result[0]
    elif isinstance(result, dict):
        r0 = result
    else:
        row['out__result_available'] = 0.0
        return row
    row['out__result_available'] = 1.0

    # Planning trajectory features.
    traj, traj_path = extract_planning_traj(result)
    row['out__planning_traj_available'] = 1.0 if traj is not None else 0.0
    if traj is not None:
        row.update(np_stats(traj, 'out__planning_traj'))
        row['out__planning_traj_path_code'] = float(len(traj_path))

    # Observed top-level UniAD outputs.
    for key in ['scores_3d', 'track_scores', 'sdc_scores_3d', 'sdc_track_scores', 'scores_3d_det', 'traj_scores', 'traj_scores_0', 'traj_scores_1']:
        if isinstance(r0, dict) and key in r0:
            row.update(np_stats(to_numpy_array(r0.get(key)), f'out__{key}'))
        else:
            row[f'out__{key}__available'] = 0.0

    occ = r0.get('occ') if isinstance(r0, dict) else None
    if isinstance(occ, dict):
        for key in ['seg_out', 'ins_seg_out']:
            row.update(np_stats(to_numpy_array(occ.get(key)), f'out__occ_{key}'))
    else:
        row['out__occ_available'] = 0.0

    planning = r0.get('planning') if isinstance(r0, dict) else None
    if isinstance(planning, dict):
        rp = planning.get('result_planning')
        if isinstance(rp, dict):
            for key in ['sdc_traj', 'sdc_traj_all']:
                row.update(np_stats(to_numpy_array(rp.get(key)), f'out__planning_{key}'))
    return row


def traj_l2_delta(clean, fault):
    ca = to_numpy_array(clean)
    fa = to_numpy_array(fault)
    if ca is None or fa is None:
        return {'traj_delta_available': 0, 'traj_delta_reason': 'missing'}
    try:
        c = np.asarray(ca, dtype=np.float64)
        f = np.asarray(fa, dtype=np.float64)
        if c.shape != f.shape:
            # Try flattening leading singleton dimensions.
            c2 = c.reshape(-1, c.shape[-1]) if c.ndim >= 2 else c.reshape(-1, 1)
            f2 = f.reshape(-1, f.shape[-1]) if f.ndim >= 2 else f.reshape(-1, 1)
        else:
            c2 = c.reshape(-1, c.shape[-1]) if c.ndim >= 2 else c.reshape(-1, 1)
            f2 = f.reshape(-1, f.shape[-1]) if f.ndim >= 2 else f.reshape(-1, 1)
        n = min(len(c2), len(f2))
        if n <= 0:
            return {'traj_delta_available': 0, 'traj_delta_reason': 'empty'}
        c2 = c2[:n]
        f2 = f2[:n]
        if c2.shape[1] >= 2 and f2.shape[1] >= 2:
            d = np.linalg.norm(f2[:, :2] - c2[:, :2], axis=1)
        else:
            d = np.abs(f2.reshape(-1)[:n] - c2.reshape(-1)[:n])
        finite = d[np.isfinite(d)]
        if finite.size == 0:
            return {'traj_delta_available': 0, 'traj_delta_reason': 'nonfinite'}
        return {
            'traj_delta_available': 1,
            'traj_delta_l2_mean': float(finite.mean()),
            'traj_delta_l2_max': float(finite.max()),
            'traj_delta_l2_final': float(finite[-1]),
            'traj_delta_steps': int(finite.size),
        }
    except Exception as e:
        return {'traj_delta_available': 0, 'traj_delta_reason': f'error:{type(e).__name__}:{e}'}



def traj_health(traj, forward_error=None, clean_or_fault: str = 'traj') -> Dict[str, Any]:
    """Classify trajectory availability and numerical validity for label-ready raw ingredients.

    This function intentionally does not use GT or clean-vs-fault delta. It only checks whether
    the trajectory output exists, has a usable shape, and contains finite numeric values.
    """
    info: Dict[str, Any] = {
        f'{clean_or_fault}_traj_missing': 0,
        f'{clean_or_fault}_traj_nonfinite': 0,
        f'{clean_or_fault}_traj_shape_invalid': 0,
        f'{clean_or_fault}_forward_error': 1 if forward_error is not None else 0,
        f'{clean_or_fault}_traj_illegal': 0,
        f'{clean_or_fault}_traj_illegal_reason': 'ok',
    }
    if forward_error is not None:
        info[f'{clean_or_fault}_traj_illegal'] = 1
        info[f'{clean_or_fault}_traj_illegal_reason'] = 'forward_error'
        return info

    arr = to_numpy_array(traj)
    if arr is None:
        info[f'{clean_or_fault}_traj_missing'] = 1
        info[f'{clean_or_fault}_traj_illegal'] = 1
        info[f'{clean_or_fault}_traj_illegal_reason'] = 'missing'
        return info

    try:
        a = np.asarray(arr)
        info[f'{clean_or_fault}_traj_shape'] = list(a.shape)
        info[f'{clean_or_fault}_traj_numel'] = int(a.size)
        if a.size == 0:
            info[f'{clean_or_fault}_traj_shape_invalid'] = 1
            info[f'{clean_or_fault}_traj_illegal'] = 1
            info[f'{clean_or_fault}_traj_illegal_reason'] = 'empty'
            return info
        if a.ndim < 2:
            info[f'{clean_or_fault}_traj_shape_invalid'] = 1
            info[f'{clean_or_fault}_traj_illegal'] = 1
            info[f'{clean_or_fault}_traj_illegal_reason'] = 'ndim_lt_2'
            return info
        if a.shape[-1] < 2:
            info[f'{clean_or_fault}_traj_shape_invalid'] = 1
            info[f'{clean_or_fault}_traj_illegal'] = 1
            info[f'{clean_or_fault}_traj_illegal_reason'] = 'last_dim_lt_2'
            return info
        finite = np.isfinite(a.astype(np.float64, copy=False))
        info[f'{clean_or_fault}_traj_finite_ratio'] = float(finite.mean())
        if not bool(finite.all()):
            info[f'{clean_or_fault}_traj_nonfinite'] = 1
            info[f'{clean_or_fault}_traj_illegal'] = 1
            info[f'{clean_or_fault}_traj_illegal_reason'] = 'nonfinite'
            return info
        return info
    except Exception as e:
        info[f'{clean_or_fault}_traj_shape_invalid'] = 1
        info[f'{clean_or_fault}_traj_illegal'] = 1
        info[f'{clean_or_fault}_traj_illegal_reason'] = f'health_error:{type(e).__name__}:{e}'
        return info


def combine_output_illegal(clean_health: Dict[str, Any], fault_health: Dict[str, Any]) -> Dict[str, int]:
    clean_illegal = int(clean_health.get('clean_traj_illegal', 0) or 0)
    fault_illegal = int(fault_health.get('fault_traj_illegal', 0) or 0)
    clean_missing = int(clean_health.get('clean_traj_missing', 0) or 0)
    fault_missing = int(fault_health.get('fault_traj_missing', 0) or 0)
    clean_nonfinite = int(clean_health.get('clean_traj_nonfinite', 0) or 0)
    fault_nonfinite = int(fault_health.get('fault_traj_nonfinite', 0) or 0)
    clean_shape_invalid = int(clean_health.get('clean_traj_shape_invalid', 0) or 0)
    fault_shape_invalid = int(fault_health.get('fault_traj_shape_invalid', 0) or 0)
    return {
        'output_illegal_or_missing': int(clean_illegal or fault_illegal),
        'output_missing': int(clean_missing or fault_missing),
        'output_nonfinite': int(clean_nonfinite or fault_nonfinite),
        'output_shape_invalid': int(clean_shape_invalid or fault_shape_invalid),
        'clean_output_illegal': clean_illegal,
        'fault_output_illegal': fault_illegal,
    }


def build_dataset_and_models():
    ensure_map_links()
    import projects.mmdet3d_plugin  # noqa: F401
    cfg = Config.fromfile(str(CFG_PATH))
    cfg.gpu_ids = [0]
    cfg.data.workers_per_gpu = 0
    cfg.data.val.test_mode = True
    cfg.data.val = sanitize_dataset_cfg(cfg.data.val)
    dataset = build_dataset(cfg.data.val)

    clean_model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fault_model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    for m in [clean_model, fault_model]:
        m.CLASSES = getattr(dataset, 'CLASSES', None)
        m.PALETTE = getattr(dataset, 'PALETTE', None)
        load_checkpoint(m, str(CKPT_PATH), map_location='cpu', strict=False)
        if torch.cuda.is_available():
            m.cuda()
        m.eval()
        maybe_reset_uniad_state(m)
    return dataset, clean_model, fault_model


def make_fault_plan(sample_ids: List[str], sample_count: int, registry: List[Dict[str, str]], seed: int, positive_ratio_min: float, positive_ratio_max: float, group_weights: Dict[str, float], bit_count_weights: Dict[int, float], bit_op_weights: Dict[str, float]):
    rng = random.Random(int(seed))
    n = int(sample_count)
    positive_min = int(math.ceil(n * float(positive_ratio_min)))
    positive_max = int(math.floor(n * float(positive_ratio_max)))
    positive_max = max(positive_min, positive_max)
    positive_count = min(n, rng.randint(positive_min, positive_max))
    positive_indices = set(rng.sample(list(range(n)), positive_count))

    by_group = defaultdict(list)
    for r in registry:
        by_group[r['uniad_group']].append(r)
    available_groups = [g for g in by_group.keys() if group_weights.get(g, 0) > 0]
    if not available_groups:
        available_groups = list(by_group.keys())
    group_w = [float(group_weights.get(g, 1.0)) for g in available_groups]

    bit_counts = list(bit_count_weights.keys())
    bit_count_w = [float(bit_count_weights[k]) for k in bit_counts]
    bit_ops = list(bit_op_weights.keys())
    bit_op_w = [float(bit_op_weights[k]) for k in bit_ops]

    plan: Dict[int, FaultEventSpec] = {}
    event_id = 0
    for i in range(n):
        if i not in positive_indices:
            continue
        g = weighted_choice(rng, available_groups, group_w)
        target = rng.choice(by_group[g])
        bit_count = int(weighted_choice(rng, bit_counts, bit_count_w))
        bit_op = str(weighted_choice(rng, bit_ops, bit_op_w))
        plan[i] = FaultEventSpec(
            sample_i=i,
            sample_id=sample_ids[i],
            event_id=event_id,
            target_id=target['target_id'],
            module_name=target['module_name'],
            module_class=target['module_class'],
            uniad_group=target['uniad_group'],
            target_level=target['target_level'],
            bit_count=bit_count,
            bit_op=bit_op,
        )
        event_id += 1
    return plan


def register_hooks(model, runtime: UniADHookRuntime):
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



# =============================
# v3.5.2b replay trace extension
# =============================
VERSION = 'uniad2_nan_replay_trace_v3_5_2b_1_exact'
DEFAULT_V351_DIR = Path('/root/autodl-tmp/UniAD2/outputs/fault_inference_v3_5_1/full6019_seed958101')
DEFAULT_V352A_DIR = Path('/root/autodl-tmp/UniAD2/outputs/nan_origin_offline_v3_5_2a/full6019_seed958101')
DEFAULT_TRACE_OUT_ROOT = Path('/root/autodl-tmp/UniAD2/outputs/nan_replay_trace_v3_5_2b')


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


def write_json_safe(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2, allow_nan=False)


def append_jsonl_safe(path: Path, row: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + '\n')


def write_csv_safe(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: json_safe(r.get(k, '')) for k in fieldnames})


def as_numeric_float_or_none(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ['', 'nan', 'inf', '+inf', '-inf', 'infinity', '+infinity', '-infinity']:
            return None
        try:
            x = float(s)
            if math.isnan(x) or math.isinf(x):
                return None
            return x
        except Exception:
            return None
    return None


def value_is_nonfinite(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ['nan', 'inf', '+inf', '-inf', 'infinity', '+infinity', '-infinity']
    if isinstance(v, (float, np.floating)):
        return bool(math.isnan(float(v)) or math.isinf(float(v)))
    return False


def read_jsonl_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_original_fault_plan(plan_path: Path) -> Dict[int, FaultEventSpec]:
    if not plan_path.exists():
        raise FileNotFoundError(f'Missing original v3.5.1 fault plan: {plan_path}')
    obj = json.loads(plan_path.read_text(encoding='utf-8'))
    events = obj.get('events', [])
    out: Dict[int, FaultEventSpec] = {}
    for e in events:
        spec = FaultEventSpec(
            sample_i=int(e['sample_i']),
            sample_id=str(e['sample_id']),
            event_id=int(e['event_id']),
            target_id=str(e['target_id']),
            module_name=str(e['module_name']),
            module_class=str(e['module_class']),
            uniad_group=str(e['uniad_group']),
            target_level=str(e['target_level']),
            bit_count=int(e['bit_count']),
            bit_op=str(e['bit_op']),
        )
        out[int(spec.sample_i)] = spec
    return out


def load_original_raw_labels(raw_path: Path):
    labels = {}
    for row in read_jsonl_rows(raw_path):
        sid = int(row.get('sample_i'))
        labels[sid] = {
            'original_output_illegal_or_missing': int(row.get('output_illegal_or_missing', 0) or 0),
            'original_output_nonfinite': int(row.get('output_nonfinite', 0) or 0),
            'original_output_missing': int(row.get('output_missing', 0) or 0),
            'original_output_shape_invalid': int(row.get('output_shape_invalid', 0) or 0),
            'original_fault_applied_count': int(row.get('fault_applied_count', 0) or 0),
            'original_fault_effective_count': int(row.get('fault_effective_count', 0) or 0),
            'original_clean_traj_path': row.get('clean_traj_path', ''),
            'original_fault_traj_path': row.get('fault_traj_path', ''),
            'original_fault_health': row.get('fault_health', {}),
            'original_delta': row.get('delta', {}),
        }
    return labels




def load_original_fault_info_by_sample(meta_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """Load the exact fault_infos recorded by v3.5.1.

    This is critical for replay: v3.5.1 sampled a concrete flat_idx and exponent-bit span
    during the first run. v3.5.2b.1 reuses that exact flat_idx/span instead of drawing a
    new random location, so the replay targets the same scalar mutation.
    """
    out: Dict[int, List[Dict[str, Any]]] = {}
    for row in read_jsonl_rows(meta_path):
        try:
            sid = int(row.get('sample_i'))
        except Exception:
            continue
        infos = row.get('fault_infos', [])
        if isinstance(infos, list):
            out[sid] = infos
        else:
            out[sid] = []
    return out


def mutate_activation_clone_exact_or_random(t: torch.Tensor, spec: FaultEventSpec, rng: random.Random, exact_info: Optional[Dict[str, Any]] = None):
    """Replay the exact v3.5.1 scalar mutation when possible.

    If exact flat_idx and bit_positions_lsb are available, they are used directly. If not,
    this falls back to the original random mutation function, but records that the replay was
    not exact.
    """
    if exact_info is None:
        new_t, info = mutate_activation_clone(t, spec, rng)
        info['exact_replay_used'] = 0
        info['exact_replay_reason'] = 'missing_exact_info_fallback_random'
        return new_t, info

    info = {
        'event_id': int(spec.event_id),
        'target_id': spec.target_id,
        'module_name': spec.module_name,
        'module_class': spec.module_class,
        'uniad_group': spec.uniad_group,
        'target_level': spec.target_level,
        'bit_count': int(spec.bit_count),
        'bit_op': spec.bit_op,
        'field': 'activation_exponent',
        'applied': 0,
        'effective': 0,
        'exact_replay_used': 1,
        'original_flat_idx': exact_info.get('flat_idx', ''),
        'original_bit_positions_lsb': exact_info.get('bit_positions_lsb', ''),
        'original_before': exact_info.get('before', ''),
        'original_after': exact_info.get('after', ''),
    }

    if t is None or not torch.is_tensor(t):
        info['reason'] = 'no_tensor_output'
        return t, info
    if not torch.is_floating_point(t):
        info['reason'] = f'not_floating_dtype:{t.dtype}'
        info['dtype'] = str(t.dtype)
        info['shape'] = list(t.shape)
        return t, info
    if t.numel() <= 0:
        info['reason'] = 'empty_tensor'
        info['dtype'] = str(t.dtype)
        info['shape'] = list(t.shape)
        return t, info

    try:
        flat_idx = int(exact_info.get('flat_idx'))
    except Exception as e:
        new_t, fallback_info = mutate_activation_clone(t, spec, rng)
        fallback_info['exact_replay_used'] = 0
        fallback_info['exact_replay_reason'] = f'invalid_exact_flat_idx:{type(e).__name__}:{e}'
        return new_t, fallback_info

    out = t.clone()
    flat = out.reshape(-1)
    if flat_idx < 0 or flat_idx >= int(flat.numel()):
        info.update({
            'reason': 'exact_flat_idx_out_of_range',
            'dtype': str(t.dtype),
            'shape': list(t.shape),
            'numel': int(flat.numel()),
            'flat_idx': int(flat_idx),
        })
        return t, info

    span = exact_info.get('bit_positions_lsb')
    if isinstance(span, str):
        try:
            span = json.loads(span)
        except Exception:
            span = []
    if not isinstance(span, list) or not span:
        new_t, fallback_info = mutate_activation_clone(t, spec, rng)
        fallback_info['exact_replay_used'] = 0
        fallback_info['exact_replay_reason'] = 'missing_or_invalid_exact_bit_positions_fallback_random'
        return new_t, fallback_info
    span = [int(x) for x in span]

    try:
        before = safe_float(flat[flat_idx].detach().float().cpu().item())
        fi = mutate_one_scalar_tensor_exponent_bits_(flat, flat_idx, span, str(spec.bit_op))
        after = safe_float(flat[flat_idx].detach().float().cpu().item())
        original_before = as_numeric_float_or_none(exact_info.get('before'))
        original_after = as_numeric_float_or_none(exact_info.get('after'))
        before_abs_diff = '' if original_before is None or as_numeric_float_or_none(before) is None else abs(float(as_numeric_float_or_none(before)) - float(original_before))
        after_abs_diff = '' if original_after is None or as_numeric_float_or_none(after) is None else abs(float(as_numeric_float_or_none(after)) - float(original_after))
        info.update(fi)
        info.update({
            'applied': 1,
            'effective': int(effective_value_changed(before, after)),
            'reason': 'ok_exact_replay',
            'attempts': 1,
            'dtype': str(t.dtype),
            'shape': list(t.shape),
            'replace_output_ok': 1,
            'before': before,
            'after': after,
            'before_abs_diff_from_original': before_abs_diff,
            'after_abs_diff_from_original': after_abs_diff,
            'before_matches_original_approx': int(before_abs_diff != '' and float(before_abs_diff) <= 1e-5),
            'after_matches_original_approx': int((value_is_nonfinite(after) and value_is_nonfinite(exact_info.get('after'))) or (after_abs_diff != '' and float(after_abs_diff) <= 1e-5)),
            'flat_idx': int(flat_idx),
            'bit_positions_lsb': span,
        })
        return out, info
    except Exception as e:
        info.update({
            'reason': f'exact_replay_error:{type(e).__name__}:{e}',
            'dtype': str(t.dtype),
            'shape': list(t.shape),
            'flat_idx': int(flat_idx),
            'bit_positions_lsb': span,
        })
        return t, info

def load_nan_case_ids(raw_path: Path, manifest_path: Optional[Path], case_limit: int, explicit_case_ids: str):
    ids: List[int] = []
    if explicit_case_ids.strip():
        for x in explicit_case_ids.split(','):
            x = x.strip()
            if x:
                ids.append(int(x))
    elif manifest_path is not None and manifest_path.exists():
        with manifest_path.open('r', encoding='utf-8', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                if 'case_id' in row:
                    ids.append(int(row['case_id']))
                elif 'sample_i' in row:
                    ids.append(int(row['sample_i']))
    else:
        for row in read_jsonl_rows(raw_path):
            if int(row.get('output_nonfinite', 0) or 0):
                ids.append(int(row['sample_i']))
    ids = sorted(set(ids))
    if case_limit and case_limit > 0:
        ids = ids[:int(case_limit)]
    if not ids:
        raise RuntimeError('No NaN/nonfinite case ids selected.')
    return ids


def tensor_trace_stats(t: Any, max_trace_elements: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'tensor_available': 0,
        'summary_error': '',
    }
    if t is None or not torch.is_tensor(t):
        return out
    out['tensor_available'] = 1
    out['dtype'] = str(t.dtype)
    out['shape'] = list(t.shape)
    out['numel'] = int(t.numel())
    out['ndim'] = int(t.ndim)
    try:
        with torch.no_grad():
            x = t.detach()
            if x.numel() == 0:
                out.update({
                    'checked_elements': 0,
                    'finite_count': 0,
                    'nonfinite_count': 0,
                    'nan_count': 0,
                    'posinf_count': 0,
                    'neginf_count': 0,
                    'finite_ratio': 1.0,
                    'finite_all': 1,
                    'absmax_finite': 0.0,
                })
                return out
            flat = x.reshape(-1)
            if int(max_trace_elements) > 0 and flat.numel() > int(max_trace_elements):
                flat = flat[:int(max_trace_elements)]
                out['trace_truncated'] = 1
            else:
                out['trace_truncated'] = 0
            xf = flat.float()
            finite = torch.isfinite(xf)
            nan_mask = torch.isnan(xf)
            posinf_mask = torch.isposinf(xf)
            neginf_mask = torch.isneginf(xf)
            checked = int(xf.numel())
            finite_count = int(finite.sum().detach().cpu().item())
            nan_count = int(nan_mask.sum().detach().cpu().item())
            posinf_count = int(posinf_mask.sum().detach().cpu().item())
            neginf_count = int(neginf_mask.sum().detach().cpu().item())
            nonfinite_count = checked - finite_count
            out['checked_elements'] = checked
            out['finite_count'] = finite_count
            out['nonfinite_count'] = nonfinite_count
            out['nan_count'] = nan_count
            out['posinf_count'] = posinf_count
            out['neginf_count'] = neginf_count
            out['finite_ratio'] = float(finite_count / max(checked, 1))
            out['finite_all'] = int(nonfinite_count == 0)
            if finite_count > 0:
                vals = xf[finite]
                absmax = float(vals.abs().max().detach().cpu().item())
                out['absmax_finite'] = safe_float(absmax)
                out['mean_finite'] = safe_float(vals.mean().detach().cpu().item())
                out['std_finite'] = safe_float(vals.std(unbiased=False).detach().cpu().item()) if vals.numel() > 1 else 0.0
                out['norm2_finite_sample'] = safe_float(torch.linalg.vector_norm(vals).detach().cpu().item())
            else:
                out['absmax_finite'] = 'nan'
                out['mean_finite'] = 'nan'
                out['std_finite'] = 'nan'
                out['norm2_finite_sample'] = 'nan'
    except Exception as e:
        out['summary_error'] = f'{type(e).__name__}:{e}'
    return out


def extract_planning_field_health(result) -> Dict[str, Any]:
    fields = {
        'planning_result_sdc_traj': [0, 'planning', 'result_planning', 'sdc_traj'],
        'planning_result_sdc_traj_all': [0, 'planning', 'result_planning', 'sdc_traj_all'],
        'top_traj': [0, 'traj'],
    }
    out = {}
    for name, path in fields.items():
        arr = to_numpy_array(extract_nested(result, path))
        out[f'{name}_path'] = '.'.join(str(p) for p in path)
        if arr is None:
            out[f'{name}_available'] = 0
            continue
        out[f'{name}_available'] = 1
        try:
            a = np.asarray(arr)
            out[f'{name}_shape'] = list(a.shape)
            out[f'{name}_numel'] = int(a.size)
            if a.size == 0:
                out[f'{name}_finite_ratio'] = 1.0
                out[f'{name}_nonfinite_count'] = 0
                out[f'{name}_absmax_finite'] = 0.0
            else:
                af = a.astype(np.float64, copy=False)
                finite = np.isfinite(af)
                out[f'{name}_finite_ratio'] = float(finite.mean())
                out[f'{name}_nonfinite_count'] = int((~finite).sum())
                if finite.any():
                    out[f'{name}_absmax_finite'] = safe_float(np.abs(af[finite]).max())
                else:
                    out[f'{name}_absmax_finite'] = 'nan'
        except Exception as e:
            out[f'{name}_error'] = f'{type(e).__name__}:{e}'
    return out


class ReplayTraceRuntime:
    def __init__(self, registry_rows: List[Dict[str, str]], max_trace_elements: int = 0, huge_threshold: float = 1e20, trace_max_events_per_sample: int = 20000, exact_fault_info_by_sample: Optional[Dict[int, List[Dict[str, Any]]]] = None):
        self.registry_rows = list(registry_rows)
        self.target_by_module = {r['module_name']: r for r in self.registry_rows}
        self.current_sample_i = -1
        self.current_sample_id = ''
        self.pending: Optional[FaultEventSpec] = None
        self.base_seed = 0
        self.trace_enabled = False
        self.max_trace_elements = int(max_trace_elements)
        self.huge_threshold = float(huge_threshold)
        self.trace_max_events_per_sample = int(trace_max_events_per_sample)
        self.call_counts = Counter()
        self.trace_events: List[Dict[str, Any]] = []
        self.fault_infos: List[Dict[str, Any]] = []
        self.consumed_event_ids = set()
        self.trace_order = 0
        self.exact_fault_info_by_sample = exact_fault_info_by_sample or {}

    def reset_sample(self, sample_i: int, sample_id: str, pending: Optional[FaultEventSpec], base_seed: int, trace_enabled: bool):
        self.current_sample_i = int(sample_i)
        self.current_sample_id = str(sample_id)
        self.pending = pending
        self.base_seed = int(base_seed)
        self.trace_enabled = bool(trace_enabled)
        self.call_counts = Counter()
        self.trace_events = []
        self.fault_infos = []
        self.consumed_event_ids = set()
        self.trace_order = 0

    def _record_trace(self, module_name: str, phase: str, output: Any, injection_info: Optional[Dict[str, Any]] = None):
        if not self.trace_enabled:
            return
        if self.trace_max_events_per_sample > 0 and len(self.trace_events) >= self.trace_max_events_per_sample:
            return
        r = self.target_by_module.get(module_name, {})
        t = first_tensor(output)
        stats = tensor_trace_stats(t, max_trace_elements=self.max_trace_elements)
        self.trace_order += 1
        absmax = as_numeric_float_or_none(stats.get('absmax_finite'))
        event = {
            'sample_i': int(self.current_sample_i),
            'sample_id': self.current_sample_id,
            'trace_order': int(self.trace_order),
            'phase': phase,
            'module_name': module_name,
            'target_id': r.get('target_id', ''),
            'module_class': r.get('module_class', ''),
            'uniad_group': r.get('uniad_group', ''),
            'target_level': r.get('target_level', ''),
            'call_count': int(self.call_counts[module_name]),
            'is_fault_module': int(self.pending is not None and self.pending.module_name == module_name),
            'is_post_mutation': int(phase == 'post_mutation'),
            'huge_threshold': float(self.huge_threshold),
            'huge_finite': int(absmax is not None and absmax >= self.huge_threshold),
            **stats,
        }
        if injection_info:
            for k, v in injection_info.items():
                event[f'injection_{k}'] = v
        self.trace_events.append(event)

    def hook_for(self, module_name: str):
        def _hook(module, inputs, output):
            self.call_counts[module_name] += 1
            self._record_trace(module_name, 'pre_mutation_or_forward_output', output)

            spec = self.pending
            if spec is None:
                return output
            if spec.module_name != module_name:
                return output
            if int(spec.event_id) in self.consumed_event_ids:
                return output
            self.consumed_event_ids.add(int(spec.event_id))

            before_t = first_tensor(output)
            rng = random.Random(sample_seed(self.base_seed, self.current_sample_id, int(spec.event_id)))
            exact_infos = self.exact_fault_info_by_sample.get(int(self.current_sample_i), [])
            exact_info = exact_infos[0] if exact_infos else None
            new_t, info = mutate_activation_clone_exact_or_random(before_t, spec, rng, exact_info=exact_info)
            info['call_count_at_injection'] = int(self.call_counts[module_name])
            replaced, ok = set_first_tensor(output, new_t)
            info['replace_output_ok'] = int(ok)
            if not ok:
                info['applied'] = 0
                info['effective'] = 0
                info['reason'] = info.get('reason', 'replace_output_failed')
            self.fault_infos.append(info)
            self._record_trace(module_name, 'post_mutation', replaced if ok else output, info)
            return replaced if ok else output
        return _hook

    def finish_sample(self):
        return list(self.fault_infos), list(self.trace_events)


def register_replay_trace_hooks(model, runtime: ReplayTraceRuntime):
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


def summarize_trace_events(events: List[Dict[str, Any]], huge_threshold: float):
    first_nonfinite = None
    first_huge = None
    post_mutation = None
    for e in events:
        if e.get('phase') == 'post_mutation' and post_mutation is None:
            post_mutation = e
        nonfinite_count = int(e.get('nonfinite_count', 0) or 0)
        if first_nonfinite is None and nonfinite_count > 0:
            first_nonfinite = e
        if first_huge is None:
            absmax = as_numeric_float_or_none(e.get('absmax_finite'))
            if absmax is not None and absmax >= float(huge_threshold):
                first_huge = e
    return first_nonfinite, first_huge, post_mutation


def event_brief(prefix: str, event: Optional[Dict[str, Any]]):
    if not event:
        return {
            f'{prefix}_found': 0,
            f'{prefix}_trace_order': '',
            f'{prefix}_phase': '',
            f'{prefix}_module_name': '',
            f'{prefix}_target_id': '',
            f'{prefix}_module_class': '',
            f'{prefix}_uniad_group': '',
            f'{prefix}_finite_ratio': '',
            f'{prefix}_nonfinite_count': '',
            f'{prefix}_nan_count': '',
            f'{prefix}_posinf_count': '',
            f'{prefix}_neginf_count': '',
            f'{prefix}_absmax_finite': '',
        }
    return {
        f'{prefix}_found': 1,
        f'{prefix}_trace_order': event.get('trace_order', ''),
        f'{prefix}_phase': event.get('phase', ''),
        f'{prefix}_module_name': event.get('module_name', ''),
        f'{prefix}_target_id': event.get('target_id', ''),
        f'{prefix}_module_class': event.get('module_class', ''),
        f'{prefix}_uniad_group': event.get('uniad_group', ''),
        f'{prefix}_finite_ratio': event.get('finite_ratio', ''),
        f'{prefix}_nonfinite_count': event.get('nonfinite_count', ''),
        f'{prefix}_nan_count': event.get('nan_count', ''),
        f'{prefix}_posinf_count': event.get('posinf_count', ''),
        f'{prefix}_neginf_count': event.get('neginf_count', ''),
        f'{prefix}_absmax_finite': event.get('absmax_finite', ''),
    }


def classify_case(fault_infos: List[Dict[str, Any]], first_nonfinite: Optional[Dict[str, Any]], first_huge: Optional[Dict[str, Any]], output_flags: Dict[str, int]):
    info = fault_infos[0] if fault_infos else {}
    after = info.get('after')
    after_x = as_numeric_float_or_none(after)
    after_nonfinite = value_is_nonfinite(after)
    after_huge = after_x is not None and abs(after_x) >= 1e20
    if after_nonfinite:
        return 'A_immediate_nonfinite_injection'
    if first_nonfinite is not None and first_nonfinite.get('phase') == 'post_mutation':
        return 'A_immediate_nonfinite_injection'
    if after_huge or first_huge is not None:
        if first_nonfinite is not None:
            return 'B_huge_finite_then_downstream_nonfinite'
        if int(output_flags.get('output_nonfinite', 0) or 0):
            return 'B_huge_finite_then_output_nonfinite'
        return 'B_huge_finite_no_reproduced_nonfinite'
    if first_nonfinite is not None:
        return 'C_downstream_nonfinite_without_recorded_huge'
    if int(output_flags.get('output_nonfinite', 0) or 0):
        return 'D_output_nonfinite_only_not_seen_in_hooks'
    return 'E_not_reproduced_or_no_nonfinite'


def write_group_count_csv(path: Path, rows: List[Dict[str, Any]], keys: List[str]):
    counter = Counter()
    for r in rows:
        counter[tuple(str(r.get(k, '')) for k in keys)] += 1
    out_rows = []
    for vals, cnt in counter.most_common():
        row = {k: vals[i] for i, k in enumerate(keys)}
        row['count'] = cnt
        out_rows.append(row)
    write_csv_safe(path, out_rows, keys + ['count'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry-csv', type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument('--v351-dir', type=Path, default=DEFAULT_V351_DIR)
    ap.add_argument('--v352a-dir', type=Path, default=DEFAULT_V352A_DIR)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=958101)
    ap.add_argument('--case-limit', type=int, default=12, help='0 means all selected NaN cases')
    ap.add_argument('--case-ids', type=str, default='', help='Comma-separated sample_i list; overrides manifest/raw selection.')
    ap.add_argument('--max-trace-elements', type=int, default=0, help='0 means full tensor finite check; >0 checks first N flattened elements.')
    ap.add_argument('--huge-threshold', type=float, default=1e20)
    ap.add_argument('--trace-max-events-per-sample', type=int, default=20000)
    ap.add_argument('--disable-online-state-clone', action='store_true')
    ap.add_argument('--no-replay-prefix-faults', action='store_true', help='If set, run fault_model only on selected cases. Default replays fault_model on every prefix sample to match v3.5.1 execution.')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[{VERSION}] out_dir={args.out_dir}', flush=True)
    print(f'[{VERSION}] v351_dir={args.v351_dir}', flush=True)
    print(f'[{VERSION}] case_limit={args.case_limit} seed={args.seed}', flush=True)

    raw_path = args.v351_dir / 'uniad2_clean_fault_raw_ingredients_v3_5_1.jsonl'
    meta_path = args.v351_dir / 'uniad2_fault_metadata_v3_5_1.jsonl'
    plan_path = args.v351_dir / 'uniad2_fault_plan_v3_5_1.json'
    manifest_path = args.v352a_dir / 'nan_case_manifest_v3_5_2a.csv'
    for p in [raw_path, meta_path, plan_path, args.registry_csv]:
        if not p.exists():
            raise FileNotFoundError(f'Missing required input: {p}')

    registry = load_registry(args.registry_csv)
    original_plan = load_original_fault_plan(plan_path)
    original_labels = load_original_raw_labels(raw_path)
    original_fault_info_by_sample = load_original_fault_info_by_sample(meta_path)
    case_ids = load_nan_case_ids(raw_path, manifest_path, int(args.case_limit), args.case_ids)
    selected_set = set(case_ids)
    max_i = max(case_ids)

    missing_plan = [i for i in case_ids if i not in original_plan]
    if missing_plan:
        raise RuntimeError(f'Selected NaN cases missing from original fault plan: {missing_plan[:20]}')

    selected_json = {
        'version': VERSION,
        'case_count': len(case_ids),
        'case_ids': case_ids,
        'max_replay_sample_i': max_i,
        'mode': 'exact_prefix_replay_with_original_fault_infos',
        'notes': [
            'The clean model is advanced through every sample from 0 to max(case_ids).',
            'By default, the fault model is executed for every prefix sample, matching v3.5.1; only selected NaN cases have trace saved.',
            'Fault specs are loaded from the original v3.5.1 full6019 fault plan, not regenerated.',
            'Exact flat_idx and bit_positions_lsb are loaded from v3.5.1 metadata fault_infos when available.'
        ]
    }
    write_json_safe(args.out_dir / 'selected_nan_cases_v3_5_2b.json', selected_json)

    dataset, clean_model, fault_model = build_dataset_and_models()
    if max_i >= len(dataset):
        raise RuntimeError(f'max selected sample_i {max_i} exceeds dataset length {len(dataset)}')

    runtime = ReplayTraceRuntime(
        registry,
        max_trace_elements=int(args.max_trace_elements),
        huge_threshold=float(args.huge_threshold),
        trace_max_events_per_sample=int(args.trace_max_events_per_sample),
        exact_fault_info_by_sample=original_fault_info_by_sample,
    )
    handles, hook_status = register_replay_trace_hooks(fault_model, runtime)
    if hook_status['registered_count'] <= 0:
        raise RuntimeError('No hooks registered on fault_model')

    trace_path = args.out_dir / 'uniad2_nan_replay_trace_events_v3_5_2b.jsonl'
    case_jsonl_path = args.out_dir / 'uniad2_nan_replay_case_summaries_v3_5_2b.jsonl'
    for p in [trace_path, case_jsonl_path]:
        if p.exists():
            p.unlink()

    counters = Counter()
    case_rows: List[Dict[str, Any]] = []
    group_counter = Counter()
    start_all = time.time()

    with torch.no_grad():
        for i in tqdm(range(max_i + 1), desc='UniAD2 NaN replay trace v3.5.2b', dynamic_ncols=True):
            sample_id = sample_id_from_dataset(dataset, i)
            selected = i in selected_set
            sample_start = time.time()

            fault_result = None
            fault_error = None
            fault_infos: List[Dict[str, Any]] = []
            trace_events: List[Dict[str, Any]] = []
            state_copy_info = {'disabled': 1, 'not_selected': int(not selected)}

            # Match v3.5.1 execution more closely: copy clean pre-sample online state,
            # run fault_model for every prefix sample, and inject original planned faults.
            # Trace is saved only for selected NaN cases.
            run_fault_prefix = (not args.no_replay_prefix_faults) or selected
            if run_fault_prefix:
                spec = original_plan.get(i)
                if (not selected) and args.no_replay_prefix_faults:
                    spec = None
                if not args.disable_online_state_clone:
                    state_copy_info = copy_online_state(clean_model, fault_model)
                data_fault = scatter_one(dataset[i])
                runtime.reset_sample(i, sample_id, spec, int(args.seed), trace_enabled=selected)
                try:
                    fault_result = fault_model(return_loss=False, rescale=True, **data_fault)
                    fault_error = None
                except Exception:
                    fault_result = None
                    fault_error = traceback.format_exc()
                fault_infos, trace_events = runtime.finish_sample()
                if selected:
                    for ev in trace_events:
                        append_jsonl_safe(trace_path, ev)

            # Always run clean path after fault path to advance clean online state exactly as v3.5.1.
            data_clean = scatter_one(dataset[i])
            try:
                clean_result = clean_model(return_loss=False, rescale=True, **data_clean)
                clean_error = None
            except Exception:
                clean_result = None
                clean_error = traceback.format_exc()

            if selected:
                clean_traj, clean_traj_path = extract_planning_traj(clean_result) if clean_result is not None else (None, '')
                fault_traj, fault_traj_path = extract_planning_traj(fault_result) if fault_result is not None else (None, '')
                delta = traj_l2_delta(clean_traj, fault_traj)
                clean_health = traj_health(clean_traj, clean_error, 'clean')
                fault_health = traj_health(fault_traj, fault_error, 'fault')
                output_flags = combine_output_illegal(clean_health, fault_health)
                planning_field_health = extract_planning_field_health(fault_result) if fault_result is not None else {}

                first_nonfinite, first_huge, post_mutation = summarize_trace_events(trace_events, float(args.huge_threshold))
                info0 = fault_infos[0] if fault_infos else {}
                failure_class = classify_case(fault_infos, first_nonfinite, first_huge, output_flags)
                original = original_labels.get(i, {})
                reproduced_nonfinite = int(output_flags.get('output_nonfinite', 0) or 0)
                original_nonfinite = int(original.get('original_output_nonfinite', 0) or 0)

                spec = original_plan[i]
                row = {
                    'version': VERSION,
                    'sample_i': i,
                    'sample_id': sample_id,
                    'elapsed_sec': time.time() - sample_start,
                    'fault_forward_error': fault_error,
                    'clean_forward_error': clean_error,
                    'state_copy_copied_count': state_copy_info.get('copied_count', ''),
                    'target_module_call_count': int(runtime.call_counts.get(original_plan[i].module_name, 0)) if i in original_plan else 0,
                    'target_id': spec.target_id,
                    'target_module': spec.module_name,
                    'target_module_class': spec.module_class,
                    'target_uniad_group': spec.uniad_group,
                    'target_level': spec.target_level,
                    'event_id': int(spec.event_id),
                    'bit_count': int(spec.bit_count),
                    'bit_op': spec.bit_op,
                    'fault_applied_count': int(sum(int(x.get('applied', 0) or 0) for x in fault_infos)) if fault_infos else 0,
                    'fault_effective_count': int(sum(int(x.get('effective', 0) or 0) for x in fault_infos)) if fault_infos else 0,
                    'injection_flat_idx': info0.get('flat_idx', ''),
                    'injection_bit_positions_lsb': info0.get('bit_positions_lsb', ''),
                    'injection_before': info0.get('before', ''),
                    'injection_after': info0.get('after', ''),
                    'injection_exact_replay_used': info0.get('exact_replay_used', ''),
                    'injection_original_before': info0.get('original_before', ''),
                    'injection_original_after': info0.get('original_after', ''),
                    'injection_before_abs_diff_from_original': info0.get('before_abs_diff_from_original', ''),
                    'injection_after_abs_diff_from_original': info0.get('after_abs_diff_from_original', ''),
                    'injection_after_is_nonfinite': int(value_is_nonfinite(info0.get('after'))),
                    'injection_after_abs': abs(as_numeric_float_or_none(info0.get('after'))) if as_numeric_float_or_none(info0.get('after')) is not None else '',
                    'injection_reason': info0.get('reason', ''),
                    'injection_dtype': info0.get('dtype', ''),
                    'injection_shape': info0.get('shape', ''),
                    'trace_event_count': len(trace_events),
                    'failure_class': failure_class,
                    'reproduced_output_illegal_or_missing': int(output_flags.get('output_illegal_or_missing', 0) or 0),
                    'reproduced_output_nonfinite': reproduced_nonfinite,
                    'reproduced_output_missing': int(output_flags.get('output_missing', 0) or 0),
                    'reproduced_output_shape_invalid': int(output_flags.get('output_shape_invalid', 0) or 0),
                    'reproduced_matches_original_nonfinite': int(reproduced_nonfinite == original_nonfinite),
                    'clean_traj_path': clean_traj_path,
                    'fault_traj_path': fault_traj_path,
                    'delta': delta,
                    'clean_health': clean_health,
                    'fault_health': fault_health,
                    'planning_field_health': planning_field_health,
                    **event_brief('post_mutation', post_mutation),
                    **event_brief('first_nonfinite', first_nonfinite),
                    **event_brief('first_huge', first_huge),
                    **original,
                }
                append_jsonl_safe(case_jsonl_path, row)
                case_rows.append(row)
                counters['selected_cases_done'] += 1
                counters['reproduced_nonfinite'] += reproduced_nonfinite
                counters['matches_original_nonfinite'] += int(reproduced_nonfinite == original_nonfinite)
                counters[f'failure_class::{failure_class}'] += 1
                group_counter[spec.uniad_group] += 1

                print(
                    f'[{VERSION}] traced sample_i={i} class={failure_class} '
                    f'reprod_nonfinite={reproduced_nonfinite} first_nonfinite={row.get("first_nonfinite_module_name")}',
                    flush=True,
                )

            counters['prefix_samples_processed'] += 1
            if (i + 1) % 25 == 0 or i == max_i:
                print(f'[{VERSION}] progress prefix {i+1}/{max_i+1} selected_done={counters["selected_cases_done"]}/{len(case_ids)}', flush=True)

    for h in handles:
        h.remove()

    case_csv_path = args.out_dir / 'uniad2_nan_replay_case_summaries_v3_5_2b.csv'
    if case_rows:
        fieldnames = []
        for r in case_rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        write_csv_safe(case_csv_path, case_rows, fieldnames)

    write_group_count_csv(args.out_dir / 'group_by_failure_class_v3_5_2b.csv', case_rows, ['failure_class'])
    write_group_count_csv(args.out_dir / 'group_by_target_group_and_failure_class_v3_5_2b.csv', case_rows, ['target_uniad_group', 'failure_class'])
    write_group_count_csv(args.out_dir / 'group_by_first_nonfinite_module_v3_5_2b.csv', case_rows, ['first_nonfinite_uniad_group', 'first_nonfinite_module_class', 'first_nonfinite_module_name'])
    write_group_count_csv(args.out_dir / 'group_by_target_module_v3_5_2b.csv', case_rows, ['target_uniad_group', 'target_module_class', 'target_module'])
    write_group_count_csv(args.out_dir / 'group_by_bit_op_and_failure_class_v3_5_2b.csv', case_rows, ['bit_op', 'failure_class'])

    summary = {
        'version': VERSION,
        'out_dir': str(args.out_dir),
        'v351_dir': str(args.v351_dir),
        'case_count_requested': len(case_ids),
        'case_ids': case_ids,
        'max_replay_sample_i': max_i,
        'seed': int(args.seed),
        'elapsed_sec': time.time() - start_all,
        'counters': dict(counters),
        'target_group_counts': dict(group_counter),
        'hook_status': hook_status,
        'files': {
            'selected_cases': str(args.out_dir / 'selected_nan_cases_v3_5_2b.json'),
            'trace_events': str(trace_path),
            'case_summaries_jsonl': str(case_jsonl_path),
            'case_summaries_csv': str(case_csv_path),
            'group_by_failure_class': str(args.out_dir / 'group_by_failure_class_v3_5_2b.csv'),
            'group_by_first_nonfinite_module': str(args.out_dir / 'group_by_first_nonfinite_module_v3_5_2b.csv'),
        },
        'interpretation_notes': [
            'post_mutation identifies whether the injected scalar/tensor immediately became nonfinite or huge.',
            'first_nonfinite is the first traced hook event whose output contains at least one nonfinite value.',
            'The clean model is advanced through every prefix sample to preserve UniAD online state before each selected replay case.',
            'If reproduced_matches_original_nonfinite is 0 for a case, interpret its trace cautiously and inspect online state or nondeterminism.',
        ],
    }
    write_json_safe(args.out_dir / 'uniad2_nan_replay_summary_v3_5_2b.json', summary)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
