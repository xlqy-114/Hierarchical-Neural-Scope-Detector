#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

VERSION = "v59_3_targeted_feature_audit_sanitized"


def read_json(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def iter_jsonl(p: Path):
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def nseed(s):
    s = str(s).strip()
    return s[4:] if s.startswith('seed') else s


def group_for_stage(stage: str) -> str:
    s = str(stage)
    if s.startswith('v59_dout') or s.startswith('dout'):
        return 'D_out_boundary'
    if s.startswith('v59_d_llm') or s.startswith('d_llm'):
        return 'D_LLM_targeted'
    if s.startswith('v59_s4'):
        return 'S4_targeted'
    if s.startswith('v26_track') or s.startswith('v26_seg'):
        return 'S4_existing_or_targeted'
    if s.startswith('c_'):
        return 'C_existing'
    if s.startswith('s') or s.startswith('v26'):
        return 'S_existing'
    return 'other'


def group_for_feature(k: str) -> str:
    s = str(k)
    if 'feat__stage__v59_dout_' in s or 'feat__stage__stage__v59_dout_' in s or 'feat__stage__dout_' in s:
        return 'D_out_boundary'
    if 'feat__stage__v59_d_llm_' in s or 'feat__stage__stage__v59_d_llm_' in s or 'feat__stage__d_llm_' in s:
        return 'D_LLM_targeted'
    if 'feat__stage__v59_s4_' in s or 'feat__stage__stage__v59_s4_' in s:
        return 'S4_targeted'
    if any(x in s for x in ['feat__stage__v26_track_', 'feat__stage__v26_seg_', 'feat__stage__s4_']):
        return 'S4_existing_or_targeted'
    if 'feat__stage__' in s:
        return 'other_stage'
    return 'non_stage'


def metric_family(k: str) -> str:
    tail = str(k).split('__')[-1]
    if tail in {'available', 'numel', 'finite_ratio', 'zero_ratio', 'near_zero_ratio', 'pos_ratio', 'neg_ratio', 'call_count', 'summarized_call_count'}:
        return 'safe_ratio_or_count'
    if any(tail.endswith(x) or tail == x for x in ['mean', 'std', 'l2_norm', 'abs_max', 'mean_abs', 'top1_over_mean', 'p95_abs', 'p99_abs']):
        return 'magnitude_stat'
    return 'other_numeric'


def safe_float(x):
    try:
        if x is None:
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def leakage_name(k: str) -> bool:
    s = str(k).lower()
    bad = [
        'harmful', 'severe', 'delta_l2', 'delta_collision', 'gt_', 'clean_',
        'label', 'posthoc', 'fault_target', 'fault_scope', 'collision_avg',
    ]
    return any(b in s for b in bad)


def summarize(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {'count': 0}
    finite = np.isfinite(a)
    out = {
        'count': int(a.size),
        'finite_count': int(finite.sum()),
        'nan_count': int(np.isnan(a).sum()),
        'inf_count': int(np.isinf(a).sum()),
        'finite_ratio': float(finite.mean()) if a.size else 0.0,
    }
    if finite.any():
        y = a[finite]
        ay = np.abs(y)
        out.update({
            'mean': float(y.mean()),
            'std': float(y.std()),
            'min': float(y.min()),
            'max': float(y.max()),
            'abs_max': float(ay.max()),
            'p95_abs': float(np.quantile(ay, 0.95)),
            'p99_abs': float(np.quantile(ay, 0.99)),
            'nonzero_ratio': float((y != 0).mean()),
        })
    return out


def sanitize_col_name(prefix: str, k: str) -> str:
    # Keep enough of the original name to trace features, while avoiding separators that break tools.
    return prefix + re.sub(r'[^A-Za-z0-9_]+', '_', str(k))[:240]


def sanitize_value(v: float, cap: float) -> Tuple[float, float, int, int]:
    missing = int(not np.isfinite(v))
    if missing:
        raw = 0.0
        extreme = 1
    else:
        raw = float(v)
        extreme = int(abs(raw) > cap)
    clipped = float(np.clip(raw, -cap, cap))
    slog = math.copysign(math.log1p(abs(clipped)), clipped)
    return clipped, slog, missing, extreme


def default_training_action(feature: str, stats: Dict[str, Any], cap: float) -> Tuple[str, str]:
    """Return (action, reason). action in keep_raw_and_transforms, use_transforms_only, drop_raw_high_risk."""
    fam = metric_family(feature)
    finite_ratio = float(stats.get('finite_ratio', 1.0) or 0.0)
    abs_max = float(stats.get('abs_max', 0.0) or 0.0)
    p99_abs = float(stats.get('p99_abs', 0.0) or 0.0)
    nan_count = int(stats.get('nan_count', 0) or 0)
    inf_count = int(stats.get('inf_count', 0) or 0)
    s = str(feature)
    if 'v59_s4_seg_decoder' in s and fam == 'magnitude_stat':
        return 'drop_raw_high_risk', 's4_seg_decoder_magnitude_stat_default_raw_disabled'
    if nan_count > 0 or inf_count > 0:
        return 'use_transforms_only', 'nonfinite_values_present'
    if abs_max > cap or p99_abs > cap:
        return 'use_transforms_only', 'extreme_value_exceeds_cap'
    if finite_ratio < 0.999:
        return 'use_transforms_only', 'feature_vector_has_nonfinite_or_missing_values'
    return 'keep_raw_and_transforms', 'normal_range'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign-dir', required=True)
    ap.add_argument('--seed-root', required=True)
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--expected-rows', type=int, default=0)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--abs-cap', type=float, default=float(os.getenv('V59_SANITIZE_ABS_CAP', '1000000')))
    ap.add_argument('--high-risk-p99-cap', type=float, default=float(os.getenv('V59_HIGH_RISK_P99_CAP', '1000000')))
    args = ap.parse_args()

    campaign = Path(args.campaign_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [x.strip() for x in args.seeds.split(',') if x.strip()]

    records = []
    files = []
    for seed in seeds:
        p = campaign / args.seed_root / f'seed{nseed(seed)}' / 'results' / 'v53_inline_detector_records.jsonl'
        rows = 0
        if p.exists():
            for r in iter_jsonl(p):
                r['_seed'] = nseed(seed)
                records.append(r)
                rows += 1
        files.append({'seed': nseed(seed), 'path': str(p), 'exists': p.exists(), 'rows': rows})

    values: Dict[str, List[float]] = {}
    groups: Dict[str, set] = {}
    meta = []
    raw_matrix_rows = []
    bad_names = []

    for r in records:
        feat = r.get('deployable_feature_row') or {}
        analysis = r.get('analysis_only') or {}
        labels = analysis.get('posthoc_labels') or {}
        m = {
            'seed': r.get('_seed'),
            'sample_id': r.get('sample_id'),
            'row_source': r.get('row_source'),
            'exact_layer': analysis.get('exact_layer'),
            'harmful': labels.get('harmful'),
            'severe_harmful': labels.get('severe_harmful'),
            'targeted_feature_present_count': 0,
            'd_feature_present_count': 0,
            's4_feature_present_count': 0,
        }
        mat = {k: m.get(k) for k in ['seed', 'sample_id', 'row_source', 'exact_layer', 'harmful', 'severe_harmful']}
        for k, v in feat.items():
            if not str(k).startswith('feat__stage__'):
                continue
            g = group_for_feature(k)
            if g not in {'D_LLM_targeted', 'D_out_boundary', 'S4_targeted', 'S4_existing_or_targeted'}:
                continue
            if leakage_name(k):
                bad_names.append(k)
                continue
            fv = safe_float(v)
            values.setdefault(k, []).append(fv)
            groups.setdefault(g, set()).add(k)
            m['targeted_feature_present_count'] += 1
            if g in {'D_LLM_targeted', 'D_out_boundary'}:
                m['d_feature_present_count'] += 1
            if g in {'S4_targeted', 'S4_existing_or_targeted'}:
                m['s4_feature_present_count'] += 1
            mat[k] = fv
        meta.append(m)
        raw_matrix_rows.append(mat)

    # Value audit
    feat_rows = []
    action_by_feature = {}
    for k, vals in sorted(values.items()):
        stats = summarize(vals)
        action, reason = default_training_action(k, stats, args.high_risk_p99_cap)
        action_by_feature[k] = action
        d = {
            'feature': k,
            'group': group_for_feature(k),
            'metric_family': metric_family(k),
            'default_training_action': action,
            'default_action_reason': reason,
        }
        d.update(stats)
        feat_rows.append(d)
    fdf = pd.DataFrame(feat_rows)
    fdf.to_csv(out / 'v59_targeted_feature_value_audit.csv', index=False)

    # High-risk feature list
    if len(fdf):
        high = fdf[fdf['default_training_action'].isin(['use_transforms_only', 'drop_raw_high_risk'])].copy()
    else:
        high = pd.DataFrame(columns=['feature', 'group', 'metric_family', 'default_training_action', 'default_action_reason'])
    high.to_csv(out / 'v59_high_risk_feature_columns.csv', index=False)

    # Group summary
    g_rows = []
    for g, names in sorted(groups.items()):
        sub = fdf[fdf['group'].eq(g)] if len(fdf) else pd.DataFrame()
        g_rows.append({
            'group': g,
            'feature_count': len(names),
            'finite_ratio_min': float(sub['finite_ratio'].min()) if len(sub) and 'finite_ratio' in sub else math.nan,
            'finite_ratio_mean': float(sub['finite_ratio'].mean()) if len(sub) and 'finite_ratio' in sub else math.nan,
            'abs_max_max': float(sub['abs_max'].max()) if len(sub) and 'abs_max' in sub else math.nan,
            'p99_abs_max': float(sub['p99_abs'].max()) if len(sub) and 'p99_abs' in sub else math.nan,
            'high_risk_feature_count': int(sub['default_training_action'].isin(['use_transforms_only', 'drop_raw_high_risk']).sum()) if len(sub) else 0,
            'drop_raw_high_risk_count': int(sub['default_training_action'].eq('drop_raw_high_risk').sum()) if len(sub) else 0,
        })
    pd.DataFrame(g_rows).to_csv(out / 'v59_targeted_feature_group_summary.csv', index=False)
    pd.DataFrame(meta).to_csv(out / 'v59_targeted_feature_sample_coverage.csv', index=False)
    pd.DataFrame(raw_matrix_rows).to_csv(out / 'v59_targeted_feature_matrix_compact_raw.csv', index=False)

    # Sanitized matrix: always include clipped and signed-log transformed values, plus missing/extreme flags.
    sanitized_rows = []
    for row in raw_matrix_rows:
        base = {k: row.get(k) for k in ['seed', 'sample_id', 'row_source', 'exact_layer', 'harmful', 'severe_harmful']}
        for k in sorted(values.keys()):
            v = safe_float(row.get(k, math.nan))
            clipped, slog, miss, extreme = sanitize_value(v, args.abs_cap)
            action = action_by_feature.get(k, 'keep_raw_and_transforms')
            # Raw clipped value is safe, but name says clipped to avoid confusing it with original raw.
            base[sanitize_col_name('clip__', k)] = clipped
            base[sanitize_col_name('slog1p__', k)] = slog
            base[sanitize_col_name('miss__', k)] = miss
            base[sanitize_col_name('extreme__', k)] = extreme
            # For normal-range features, keep an additional alias with original-like prefix for quick ablations.
            if action == 'keep_raw_and_transforms':
                base[sanitize_col_name('safe_raw__', k)] = clipped
        sanitized_rows.append(base)
    sdf = pd.DataFrame(sanitized_rows)
    sdf.to_csv(out / 'v59_targeted_feature_matrix_sanitized.csv', index=False)

    # Hook status summary.
    hook_rows = []
    for seed in seeds:
        p = campaign / args.seed_root / f'seed{nseed(seed)}' / 'results' / 'stage_hook_status_v26.json'
        if p.exists():
            st = read_json(p)
            for name, row in st.items():
                rr = {'seed': nseed(seed), 'stage': name, 'group': group_for_stage(name)}
                if isinstance(row, dict):
                    rr.update(row)
                hook_rows.append(rr)
        else:
            hook_rows.append({'seed': nseed(seed), 'stage': '__missing__', 'group': 'unknown', 'registered': 0})
    hdf = pd.DataFrame(hook_rows)
    hdf.to_csv(out / 'v59_stage_hook_status_summary.csv', index=False)
    reg_counts = {}
    if len(hdf) and 'registered' in hdf:
        hdf['_reg'] = pd.to_numeric(hdf['registered'], errors='coerce').fillna(0)
        reg_counts = hdf.groupby('group')['_reg'].sum().astype(int).to_dict()

    pd.DataFrame({'feature': sorted(set(bad_names))}).to_csv(out / 'v59_targeted_feature_name_leakage_audit.csv', index=False)

    rows = len(records)
    expected = args.expected_rows * len(seeds) if args.expected_rows else None
    d_count = len(groups.get('D_LLM_targeted', set())) + len(groups.get('D_out_boundary', set()))
    s4_count = len(groups.get('S4_targeted', set())) + len(groups.get('S4_existing_or_targeted', set()))
    pass_rows = expected is None or rows == expected
    pass_features = d_count > 0 and s4_count > 0
    pass_leak = len(set(bad_names)) == 0
    pass_finite = True
    if len(fdf) and 'finite_ratio' in fdf:
        pass_finite = bool((pd.to_numeric(fdf['finite_ratio'], errors='coerce').fillna(0) > 0).all())
    # v59.2 does not fail on high-risk values; it records and sanitizes them.
    high_risk_count = int(len(high))

    summary = {
        'version': VERSION,
        'campaign_dir': str(campaign),
        'seed_root': args.seed_root,
        'seeds': [nseed(x) for x in seeds],
        'rows': rows,
        'expected_rows_total': expected,
        'file_rows': files,
        'targeted_feature_count_total': len(values),
        'targeted_feature_count_by_group': {g: len(v) for g, v in groups.items()},
        'd_related_feature_count': d_count,
        's4_related_feature_count': s4_count,
        'hook_registered_counts_by_group': reg_counts,
        'leakage_bad_feature_name_count': len(set(bad_names)),
        'high_risk_feature_count': high_risk_count,
        'abs_cap_for_sanitization': args.abs_cap,
        'high_risk_p99_cap': args.high_risk_p99_cap,
        'pass_rows': pass_rows,
        'pass_features': pass_features,
        'pass_leakage': pass_leak,
        'pass_finite': pass_finite,
        'pass': bool(pass_rows and pass_features and pass_leak and pass_finite),
        'policy': 'Scalar summaries only; no full tensor dumps. v59.2 adds clipped/log/missing/extreme sanitized matrix. High-risk raw values are diagnostic-only and not recommended for training.',
        'outputs': {
            'feature_value_audit_csv': str(out / 'v59_targeted_feature_value_audit.csv'),
            'high_risk_feature_columns_csv': str(out / 'v59_high_risk_feature_columns.csv'),
            'feature_group_summary_csv': str(out / 'v59_targeted_feature_group_summary.csv'),
            'sample_coverage_csv': str(out / 'v59_targeted_feature_sample_coverage.csv'),
            'targeted_feature_matrix_compact_raw_csv': str(out / 'v59_targeted_feature_matrix_compact_raw.csv'),
            'targeted_feature_matrix_sanitized_csv': str(out / 'v59_targeted_feature_matrix_sanitized.csv'),
            'stage_hook_status_summary_csv': str(out / 'v59_stage_hook_status_summary.csv'),
            'feature_name_leakage_audit_csv': str(out / 'v59_targeted_feature_name_leakage_audit.csv'),
        }
    }
    write_json(out / 'v59_targeted_feature_audit_summary.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.strict and not summary['pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
