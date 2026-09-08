#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def metrics(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    return {
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'recall': tp / (tp + fn) if tp + fn else None,
        'far': fp / (fp + tn) if fp + tn else None,
        'precision': tp / (tp + fp) if tp + fp else None,
    }


def normalize_sample_token(x: str) -> str:
    x = str(x)
    return x[:-11] if x.endswith('_trajectory') else x


def v48_normalize_trajectory_object(obj: Any) -> Optional[List[Tuple[float, float]]]:
    try:
        arr = np.asarray(obj, dtype=np.float64)
        if arr.size < 2:
            return None
        arr = arr.reshape(-1, 2)
        if arr.shape[0] < 1:
            return None
        arr = arr[:6]
        if arr.shape[0] < 6:
            arr = np.concatenate([arr] + [arr[-1:].copy()] * (6 - arr.shape[0]), axis=0)
        return [(float(x), float(y)) for x, y in arr[:6]]
    except Exception:
        return None


def robust_retrieve_traj_with_meta(answer: Any) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    if isinstance(answer, (list, tuple)):
        raw_text = ' '.join(str(x) for x in answer)
    else:
        raw_text = str(answer)

    numeric_tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_text)
    float_tokens = [t for t in numeric_tokens if ('.' in t or 'e' in t.lower())]
    marker_tokens = ['trajectory', 'traj', 'waypoint', 'points', 'planning', 'plan']
    repeated_char_max = 0
    if raw_text:
        cur = 1
        for i in range(1, len(raw_text)):
            if raw_text[i] == raw_text[i - 1]:
                cur += 1
            else:
                repeated_char_max = max(repeated_char_max, cur)
                cur = 1
        repeated_char_max = max(repeated_char_max, cur)

    meta: Dict[str, Any] = {
        'answer_type': type(answer).__name__,
        'answer_char_len': len(raw_text),
        'answer_token_like_len': len(raw_text.split()),
        'answer_line_count': max(1, raw_text.count('\n') + 1) if raw_text else 0,
        'answer_digit_count': len(re.findall(r'\d', raw_text)),
        'answer_minus_count': raw_text.count('-'),
        'answer_decimal_count': raw_text.count('.'),
        'answer_comma_count': raw_text.count(','),
        'answer_colon_count': raw_text.count(':'),
        'answer_semicolon_count': raw_text.count(';'),
        'answer_lparen_count': raw_text.count('('),
        'answer_lbracket_count': raw_text.count('['),
        'answer_numeric_token_count': len(numeric_tokens),
        'answer_float_token_count': len(float_tokens),
        'answer_unique_numeric_token_count': len(set(numeric_tokens)),
        'answer_repeated_char_max_run': repeated_char_max,
        'answer_is_empty': int(len(raw_text.strip()) == 0),
        'answer_is_maxlen_like': int(len(raw_text) >= 512),
        'answer_has_alpha': int(bool(re.search(r'[A-Za-z]', raw_text))),
        'answer_has_chinese': int(bool(re.search(r'[\u4e00-\u9fff]', raw_text))),
        'answer_has_bracket': int(any(ch in raw_text for ch in '[]()<>')),
        'answer_has_trajectory_marker': int(any(tok in raw_text.lower() for tok in marker_tokens)),
        'answer_has_explanatory_text': int(bool(re.search(r'[A-Za-z]{4,}', raw_text))),
    }

    text = re.sub(r'[a-zA-Z]', '', raw_text)
    text = re.sub(r'[\u4e00-\u9fff]', '', text)
    text = re.sub(r'(\d+)\.\.+', r'\1.', text)
    text = re.sub(r'[<>]', '', text)
    text = re.sub(r'(\d+\.\d+)\s+(\d+\.\d+)(?=\s*[,\)])', r'\1, \2', text)
    text = re.sub(r'(\d+\.\d+)-(\d+\.\d+)', r'\1, \2', text)
    text = re.sub(r'(\(\s*-?\d+\.\d+,\s*-?\d+\.\d+),\s*-?\d+\.\d+(\s*\))', r'\1\2', text)
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'(\d+\.\d+)\.(\d+)', r'\1\2', text)
    text = re.sub(r'(\d+)\.\.(\d+)', r'\1.\2', text)
    text = re.sub(r'(\d+)\.\-(\d+)', r'\1.\2', text)

    coord_pairs = re.findall(r'[\[\(]([-\deE.+]+),\s*([-\deE.+]+)[\]\)]', text)
    meta.update({
        'raw_coord_pair_count': len(coord_pairs),
        'parse_success': int(len(coord_pairs) > 0),
        'parse_padded': int(0 < len(coord_pairs) < 6),
        'parse_truncated': int(len(coord_pairs) > 6),
        'parse_fallback_zero': int(len(coord_pairs) == 0),
        'parse_fail_no_numeric_tokens': int(len(coord_pairs) == 0 and len(numeric_tokens) == 0),
        'parse_fail_numbers_no_pairs': int(len(coord_pairs) == 0 and len(numeric_tokens) > 0),
        'parse_fail_brackets_no_pairs': int(len(coord_pairs) == 0 and meta.get('answer_has_bracket', 0) == 1),
        'parse_fail_no_brackets': int(len(coord_pairs) == 0 and meta.get('answer_has_bracket', 0) == 0),
    })
    try:
        coords = [(float(x), float(y)) for x, y in coord_pairs]
        meta['parse_exception'] = 0
    except Exception:
        coords = []
        meta['parse_success'] = 0
        meta['parse_fallback_zero'] = 1
        meta['parse_exception'] = 1
    if not coords:
        traj = [(0.0, 0.0)] * 6
    else:
        if len(coords) < 6:
            coords.extend([coords[-1]] * (6 - len(coords)))
        traj = coords[:6]
    arr = np.asarray(traj, dtype=np.float64).reshape(-1, 2)
    meta['valid_waypoint_count'] = int(np.isfinite(arr).all(axis=1).sum()) if arr.size else 0
    meta['parsed_all_zero'] = int(float(np.max(np.abs(np.nan_to_num(arr, nan=0.0)))) < 1e-9) if arr.size else 1
    meta['parsed_final_disp'] = float(np.linalg.norm(np.nan_to_num(arr[-1], nan=0.0))) if arr.size else 0.0
    if len(arr) >= 2:
        step = np.linalg.norm(np.diff(np.nan_to_num(arr, nan=0.0), axis=0), axis=1)
        meta['parsed_path_length'] = float(np.sum(step))
        meta['parsed_step_nonzero_count'] = int((step > 1e-6).sum())
        meta['parsed_repeated_waypoint_count'] = int((step <= 1e-9).sum())
    else:
        meta['parsed_path_length'] = 0.0
        meta['parsed_step_nonzero_count'] = 0
        meta['parsed_repeated_waypoint_count'] = 0
    return traj, meta


def v48_plan_conv_forced_traj(row: Dict[str, Any]) -> Tuple[Optional[List[Tuple[float, float]]], Dict[str, Any]]:
    if 'v48_e1_forced_traj' not in row:
        return None, {'v48_e1_forced_traj_used': 0}
    forced = v48_normalize_trajectory_object(row.get('v48_e1_forced_traj'))
    if not forced:
        return None, {'v48_e1_forced_traj_used': 0, 'v48_e1_forced_traj_invalid': 1}
    meta = {
        'v48_e1_forced_traj_used': 1,
        'v48_e1_forced_traj_invalid': 0,
        'v48_e1_fault_meta': row.get('v48_e1_fault_meta', {}),
    }
    return forced, meta


def build_pred_files_from_plan_conv(result_dir: Path) -> Dict[str, Any]:
    plan_conv_path = result_dir / 'plan_conv.json'
    if not plan_conv_path.is_file():
        raise FileNotFoundError(f'Missing {plan_conv_path}')
    pred: Dict[str, Any] = {}
    pred_multi: Dict[str, Any] = {}
    parse_rows: List[Dict[str, Any]] = []
    invalid: List[str] = []
    total = 0
    with plan_conv_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            sid = str(row.get('id', ''))
            ans = row.get('answer', '')
            forced, forced_meta = v48_plan_conv_forced_traj(row)
            if forced is not None:
                traj = forced
                _, parse_meta = robust_retrieve_traj_with_meta(ans)
                parse_meta.update(forced_meta)
            else:
                traj, parse_meta = robust_retrieve_traj_with_meta(ans)
            if traj == [(0.0, 0.0)] * 6:
                invalid.append(sid)
            pred[sid] = traj
            pred_multi[sid] = [traj]
            parse_rows.append({'sample_id': normalize_sample_token(sid), 'raw_sample_id': sid, 'e_parse_features': parse_meta})
    (result_dir / 'pred_trajs_dict.json').write_text(json.dumps(pred, ensure_ascii=False), encoding='utf-8')
    (result_dir / 'pred_trajs_multi_modal_dict.json').write_text(json.dumps(pred_multi, ensure_ascii=False), encoding='utf-8')
    with (result_dir / 'e_parse_metadata_v26.jsonl').open('w', encoding='utf-8') as f:
        for row in parse_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return {
        'count': total,
        'invalid_parse_count': len(invalid),
        'invalid_parse_examples': invalid[:10],
        'pred_path': str(result_dir / 'pred_trajs_dict.json'),
        'pred_multi_path': str(result_dir / 'pred_trajs_multi_modal_dict.json'),
        'e_parse_meta_path': str(result_dir / 'e_parse_metadata_v26.jsonl'),
    }


def iter_plan_conv_tokens(path: Path) -> List[str]:
    out: List[str] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = normalize_sample_token(str(row.get('id', '')))
            if sid:
                out.append(sid)
    return out


def pred_lookup(pred: Dict[str, Any], token: str):
    for k in (token + '_trajectory', token):
        if k in pred:
            return pred[k]
    return None


def compute_subset_metrics_standalone(repo: Path, result_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    # Deliberately imports only the lightweight official planning metric helper.
    # OpenDriveVLA historically exposes ``eval_share`` from <repo>/drivevla,
    # so reproduce that exact import path without importing the VLA inference/model stack.
    drivevla_root = repo / 'drivevla'
    metric_py = drivevla_root / 'eval_share' / 'metric.py'
    evaluation_py = drivevla_root / 'eval_share' / 'evaluation.py'
    if not metric_py.is_file() or not evaluation_py.is_file():
        raise FileNotFoundError(
            'Expected local eval_share package under '
            f'{drivevla_root}; missing metric.py/evaluation.py'
        )
    for local_path in (repo, drivevla_root):
        local_s = str(local_path)
        if local_s not in sys.path:
            sys.path.insert(0, local_s)

    from eval_share.metric import PlanningMetric
    import eval_share.evaluation as evaluation_mod

    gt_dir = Path(evaluation_mod.__file__).resolve().parent / 'gt'
    required = [gt_dir / 'planing_gt_segmentation_val', gt_dir / 'gt_traj.pkl', gt_dir / 'gt_traj_mask.pkl']
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f'Missing official planning GT material: {p}')

    with (result_dir / 'pred_trajs_dict.json').open('r', encoding='utf-8') as f:
        pred_dict = json.load(f)
    tokens = iter_plan_conv_tokens(result_dir / 'plan_conv.json')

    with (gt_dir / 'planing_gt_segmentation_val').open('rb') as f:
        gt_occ_map = pickle.load(f)
    for token in list(gt_occ_map.keys()):
        if not isinstance(gt_occ_map[token], torch.Tensor):
            gt_occ_map[token] = torch.tensor(gt_occ_map[token])
        gt_occ_map[token] = torch.flip(gt_occ_map[token], [-1])
    with (gt_dir / 'gt_traj.pkl').open('rb') as f:
        gt_trajs = pickle.load(f)
    with (gt_dir / 'gt_traj_mask.pkl').open('rb') as f:
        gt_masks = pickle.load(f)

    metric = PlanningMetric(6).to(torch.device('cpu'))
    rows: List[Dict[str, Any]] = []
    missing_pred: List[str] = []
    missing_gt: List[str] = []

    for token in tokens:
        pred = pred_lookup(pred_dict, token)
        if pred is None:
            missing_pred.append(token)
            continue
        if token not in gt_trajs or token not in gt_masks or token not in gt_occ_map:
            missing_gt.append(token)
            continue
        gt_traj = torch.tensor(gt_trajs[token]).to(torch.device('cpu'))
        gt_mask = torch.tensor(gt_masks[token]).to(torch.device('cpu'))
        occ = gt_occ_map[token].to(torch.device('cpu'))
        out = torch.tensor(pred).reshape(gt_mask.shape).to(torch.device('cpu'))
        if out.shape[1] % 2:
            out = out[:, 1:]
        if occ.shape[1] % 2:
            occ = occ[:, 1:]
        if gt_traj.shape[1] % 2:
            gt_traj = gt_traj[:, 1:]
        if gt_mask.shape[1] % 2:
            gt_mask = gt_mask[:, 1:]
        out = out[:, :6]
        gt_traj = gt_traj[:, :6]
        gt_mask = gt_mask[:, :6]
        occ = occ[:, :6]
        l2 = metric.compute_L2(out, gt_traj, gt_mask)[0].tolist()
        _, obj_box = metric.evaluate_coll(out[:, :, :2], gt_traj[:, :, :2], occ, token=None)
        obj_box = [float(x) for x in obj_box.tolist()]
        l2_1s = float(sum(l2[:2]) / 2.0)
        l2_2s = float(sum(l2[:4]) / 4.0)
        l2_3s = float(sum(l2[:6]) / 6.0)
        col_1s = float(sum(obj_box[:2]) / 2.0 * 100.0)
        col_2s = float(sum(obj_box[:4]) / 4.0 * 100.0)
        col_3s = float(sum(obj_box[:6]) / 6.0 * 100.0)
        rows.append({
            'sample_id': token,
            'l2_t': l2,
            'collision_t': obj_box,
            'l2_1s': l2_1s, 'l2_2s': l2_2s, 'l2_3s': l2_3s,
            'l2_avg': float((l2_1s + l2_2s + l2_3s) / 3.0),
            'collision_1s': col_1s, 'collision_2s': col_2s, 'collision_3s': col_3s,
            'collision_avg': float((col_1s + col_2s + col_3s) / 3.0),
        })

    out_path = result_dir / 'per_sample_planning_metrics.jsonl'
    with out_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = {
        'sample_count_requested': len(tokens),
        'sample_count_scored': len(rows),
        'missing_pred_count': len(missing_pred),
        'missing_gt_count': len(missing_gt),
        'missing_pred_examples': missing_pred[:10],
        'missing_gt_examples': missing_gt[:10],
    }
    return out_path, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--repo-root', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--expected-rows', type=int, default=4819)
    args = ap.parse_args()
    root = Path(args.root)
    repo = Path(args.repo_root)
    run = Path(args.run_dir)
    out = run / 'reviewer_evaluation'
    out.mkdir(parents=True, exist_ok=True)

    parse_info = build_pred_files_from_plan_conv(run)
    fault_metric_path, planning_summary = compute_subset_metrics_standalone(repo, run)
    planning_summary['parser'] = parse_info
    (out / 'fault_planning_metrics_summary.json').write_text(json.dumps(planning_summary, ensure_ascii=False, indent=2), encoding='utf-8')

    clean_path = root / 'work/inputs_extracted/clean_reference_official6019/per_sample_planning_metrics.jsonl'
    clean = {str(r['sample_id']): r for r in load_jsonl(clean_path) if r.get('sample_id')}
    fault_rows = load_jsonl(Path(fault_metric_path))
    fault = {str(r['sample_id']): r for r in fault_rows if r.get('sample_id')}
    online = load_jsonl(run / 'results/v53_inline_detector_records.jsonl')
    if len(online) != args.expected_rows or len(fault_rows) != args.expected_rows:
        raise SystemExit(f'Row count mismatch online={len(online)} fault_metrics={len(fault_rows)} expected={args.expected_rows}')

    policy_dir = root / 'frozen_policy'
    policy_features = json.loads((policy_dir / 'old_exact_2605_features.json').read_text(encoding='utf-8'))
    policy_model = joblib.load(policy_dir / 'table5_severe_old_exact_extratrees_final.joblib')

    parity_mismatch = 0
    parity_max_abs = 0.0
    rows = []
    for r in online:
        sid = str(r.get('sample_id', ''))
        if sid not in clean or sid not in fault:
            raise SystemExit(f'Missing clean/fault metric for {sid}')
        c = clean[sid]
        f = fault[sid]
        a = r.get('analysis_only') or {}
        t = r.get('r1c1_table5_frozen_policy') or {}
        if not t:
            raise SystemExit(f'Missing frozen policy decision for {sid}')
        inj = int(a.get('is_positive_effective', 0) or 0)
        dl2 = float(f['l2_avg']) - float(c['l2_avg'])
        dc = float(f['collision_avg']) - float(c['collision_avg'])
        severe = int(inj == 1 and (dl2 > 1.0 or dc > 0.0))
        faults = a.get('faults') or []
        f0 = faults[0] if faults and isinstance(faults[0], dict) else {}

        flat = r.get('deployable_feature_row') or {}
        vals = []
        for col in policy_features:
            try:
                v = float(flat.get(col, math.nan))
            except Exception:
                v = math.nan
            if not math.isfinite(v):
                v = math.nan
            elif v > 1e6:
                v = 1e6
            elif v < -1e6:
                v = -1e6
            vals.append(v)
        pp = policy_model.predict_proba(np.asarray([vals], dtype=np.float64))
        rs = float(pp[0, 1]) if pp.ndim == 2 and pp.shape[1] == 2 else 0.0
        diff = abs(rs - float(t['feature_model_score']))
        parity_max_abs = max(parity_max_abs, diff)
        if diff > 1e-12:
            parity_mismatch += 1

        rows.append({
            'sample_id': sid,
            'row_source': r.get('row_source', ''),
            'injected_effective': inj,
            'exact_layer': a.get('exact_layer', ''),
            'fault_target_exact': a.get('fault_target_exact', ''),
            'fault_op': f0.get('bit_op', f0.get('op', '')),
            'bit_count': f0.get('bit_count', ''),
            'clean_l2_avg': float(c['l2_avg']),
            'fault_l2_avg': float(f['l2_avg']),
            'delta_l2_avg': dl2,
            'clean_collision_avg': float(c['collision_avg']),
            'fault_collision_avg': float(f['collision_avg']),
            'delta_collision_avg': dc,
            'severe_harmful': severe,
            'feature_model_score': float(t['feature_model_score']),
            'feature_model_threshold': float(t['feature_model_threshold']),
            'feature_model_alarm': int(t['feature_model_alarm']),
            'base_alarm': int(t['base_alarm']),
            'old_hgb_score': float(t['old_hgb_score']),
            'speed_hgb_score': float(t['speed_hgb_score']),
            'hgbmax_score': float(t['hgbmax_score']),
            'veto': int(t['veto']),
            'pre_veto_alarm': int(t['pre_veto_alarm']),
            'final_alarm': int(t['final_alarm']),
        })

    df = pd.DataFrame(rows)
    if df.sample_id.nunique() != args.expected_rows:
        raise SystemExit('Formal evaluation has duplicate/missing sample IDs')
    mainm = metrics(df.severe_harmful, df.final_alarm)
    basem = metrics(df.severe_harmful, df.base_alarm)
    featm = metrics(df.severe_harmful, df.feature_model_alarm)

    op_rows = []
    injected = df[df.injected_effective == 1].copy()
    for op, g in injected.groupby('fault_op', dropna=False):
        m = metrics(g.severe_harmful, g.final_alarm)
        m.update({'fault_op': str(op), 'rows': len(g), 'severe_positive': int(g.severe_harmful.sum())})
        op_rows.append(m)
    pd.DataFrame(op_rows).to_csv(out / 'fault_operator_stratified_metrics.csv', index=False)

    layer_rows = []
    for layer, g in df.groupby('exact_layer', dropna=False):
        m = metrics(g.severe_harmful, g.final_alarm)
        m.update({'exact_layer': str(layer), 'rows': len(g), 'severe_positive': int(g.severe_harmful.sum())})
        layer_rows.append(m)
    pd.DataFrame(layer_rows).to_csv(out / 'exact_layer_stratified_metrics.csv', index=False)
    df.to_csv(out / 'r1c1_formal_per_sample_evaluation.csv', index=False)

    summary = {
        'version': 'R1C1_Table5_FrozenPolicy_Broader4819_v1_5_formal_evaluation',
        'test_design': {
            'previously_unused_samples': args.expected_rows,
            'new_fault_seed': 970001,
            'policy_frozen_before_test': True,
            'test_used_for_training_or_threshold_selection': False,
        },
        'label_policy': 'severe_harmful = injected_effective AND (delta_l2_avg > 1.0 OR delta_collision_avg > 0.0)',
        'information_boundary': 'clean reference and GT-derived metrics are used only here after online alarm generation',
        'rows': len(df),
        'effective_injection_count': int(df.injected_effective.sum()),
        'severe_positive': int(df.severe_harmful.sum()),
        'severe_negative': int((df.severe_harmful == 0).sum()),
        'online_offline_feature_model_parity': {
            'mismatch_count_at_1e-12': int(parity_mismatch),
            'max_abs_score_diff': float(parity_max_abs),
            'pass': bool(parity_mismatch == 0),
        },
        'frozen_table5_policy': mainm,
        'v35_base_only': basem,
        'feature_model_only': featm,
        'parse_info': parse_info,
        'planning_metric_summary': planning_summary,
    }
    if parity_mismatch:
        raise SystemExit(f'Online/offline frozen feature-model parity failed: mismatches={parity_mismatch}, max_abs={parity_max_abs}')
    (out / 'r1c1_formal_metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame([
        {'system': 'frozen_table5_policy', **mainm},
        {'system': 'v35_base_only', **basem},
        {'system': 'feature_model_only', **featm},
    ]).to_csv(out / 'r1c1_formal_metrics.csv', index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
