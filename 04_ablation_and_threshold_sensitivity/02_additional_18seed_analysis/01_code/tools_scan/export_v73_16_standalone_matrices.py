#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export v73.16 standalone no-detector runtime/source and model-ready matrices from a real inference run.

Input is not an existing training matrix. It reads per-sample v73_16_feature_records.jsonl produced by the
OpenDriveVLA standalone inference loop and plan_conv.json predictions produced by that same run. It does not load
or execute any detector model. It only rebuilds the exact online feature rows and posthoc label/raw
ingredients needed for later training/testing/analysis.
"""
from __future__ import annotations
import argparse, ast, importlib.util, json, math, pickle, re, sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow importing package-local drivevla/v73_16_feature_runtime_filter.py after scripts are copied.
THIS = Path(__file__).resolve()
PKG_DIR = THIS.parents[1]
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
from drivevla.v73_16_data_contract_filter import V7314StandaloneDataContractFilter  # type: ignore

KEY_COLS = ["seed", "sample_id"]
V59_SANITIZE_PREFIXES = ("clip__", "slog1p__", "miss__", "extreme__", "safe_raw__")
FORBIDDEN_PREFIXES = ("score__", "alarm__", "threshold__")
FORBIDDEN_EXACT = {"hgbmax"}
FORBIDDEN_SUBSTRINGS = ("lineA_old", "lineA_speed", "linea_old", "linea_speed")

def log(msg: str) -> None:
    print(f"[v73.16 export] {msg}", flush=True)

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}

def import_inference_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"missing inference module: {path}")
    spec = importlib.util.spec_from_file_location('v7310_inference_module', str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import inference module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules['v7310_inference_module'] = mod
    spec.loader.exec_module(mod)
    return mod

def safe_float(x: Any, default: float = math.nan) -> float:
    try:
        v=float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def nseed(s: Any) -> str:
    s=str(s).strip()
    return s[4:] if s.startswith('seed') else s

def plan_lookup_from_file(plan_path: Path, inf) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not plan_path.is_file():
        return out
    with plan_path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                r=json.loads(line)
            except Exception:
                continue
            sid=inf.normalize_sample_token(str(r.get('id','')))
            if sid:
                out[sid]=r
    return out

class NoDetectorRuntimeRowBuilder:
    def __init__(self, inf, clean_ref: Optional[Path], output_sanity_json: Optional[Path], analysis_metrics: bool = True):
        self.inf=inf
        self.metric_runtime=inf.V53InlinePlanningMetricRuntime(enabled=analysis_metrics)
        self.clean_ref=Path(clean_ref) if clean_ref else None
        self.clean_metrics: Dict[str, Dict[str, Any]] = {}
        self.clean_ref_available=False
        self.clean_ref_reason='not_configured'
        if self.clean_ref is not None:
            try:
                if self.clean_ref.is_file():
                    self.clean_metrics={inf.normalize_sample_token(str(r.get('sample_id',''))): r for r in inf.load_jsonl(self.clean_ref) if r.get('sample_id')}
                    self.clean_ref_available=True
                    self.clean_ref_reason=str(self.clean_ref)
                else:
                    self.clean_ref_reason=f'missing:{self.clean_ref}'
            except Exception as e:
                self.clean_ref_reason=f'error:{type(e).__name__}:{e}'
        self.output_sanity_json=Path(output_sanity_json) if output_sanity_json else None
        self.output_sanity_thresholds=None
        self.output_sanity_thresholds_reason='not_configured'
        if self.output_sanity_json is not None:
            try:
                if self.output_sanity_json.is_file():
                    self.output_sanity_thresholds=inf._load_json(self.output_sanity_json)
                    self.output_sanity_thresholds_reason=str(self.output_sanity_json)
                else:
                    self.output_sanity_thresholds_reason=f'missing:{self.output_sanity_json}'
            except Exception as e:
                self.output_sanity_thresholds_reason=f'error:{type(e).__name__}:{e}'
        self.rows_for_temporal: List[Dict[str, Any]] = []

    def _build_fault_metric_row(self, sample_id: str, output_sanity: Dict[str, Any], e_parse: Dict[str, Any], planning_metrics: Dict[str, Any]) -> Dict[str, Any]:
        inf=self.inf
        sid=inf.normalize_sample_token(sample_id)
        f=dict(planning_metrics or {})
        c=self.clean_metrics.get(sid)
        out: Dict[str, Any] = {
            'sample_id': sid,
            'output_sanity_features': output_sanity or {},
            'e_parse_features': e_parse or {},
            'clean_ref_available': int(c is not None),
            'gt_metric_available': int(bool(f.get('available', False))),
        }
        for k in ['l2_1s','l2_2s','l2_3s','l2_avg','collision_1s','collision_2s','collision_3s','collision_avg']:
            out[f'fault_{k}']=safe_float(f.get(k))
            out[f'clean_{k}']=safe_float((c or {}).get(k)) if c is not None else math.nan
        for suffix in ['1s','2s','3s','avg']:
            fk=f'l2_{suffix}'; ck=f'collision_{suffix}'
            fv=safe_float(f.get(fk)); cv=safe_float((c or {}).get(fk)) if c is not None else math.nan
            out[f'delta_l2_{suffix}']=float(fv-cv) if math.isfinite(fv) and math.isfinite(cv) else math.nan
            fv=safe_float(f.get(ck)); cv=safe_float((c or {}).get(ck)) if c is not None else math.nan
            out[f'delta_collision_{suffix}']=float(fv-cv) if math.isfinite(fv) and math.isfinite(cv) else math.nan
        return out

    def prepare_runtime_row(self, record: Dict[str, Any], result: Dict[str, Any], sample_id: str, sample_id_full: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        inf=self.inf
        plan_row=dict(result or {})
        plan_row.setdefault('id', sample_id_full)
        ans=plan_row.get('answer','')
        forced_traj, forced_meta = inf.v48_plan_conv_forced_traj(plan_row)
        if forced_traj is not None:
            traj=forced_traj
            _, answer_parse_meta = inf.robust_retrieve_traj_with_meta(ans)
            e_parse=dict(answer_parse_meta); e_parse.update(forced_meta)
        else:
            traj, e_parse = inf.robust_retrieve_traj_with_meta(ans)
        output_sanity=inf.trajectory_sanity_features_from_pred(traj)
        planning_metrics=self.metric_runtime.compute(sample_id, traj)
        metric_row=self._build_fault_metric_row(sample_id, output_sanity, e_parse, planning_metrics)
        output_rep=inf.detect_output_sanity(metric_row, self.output_sanity_thresholds)
        metric_row['output_sanity_report']=output_rep
        injected=int(bool(record.get('faults')) and int(record.get('is_positive_effective', record.get('is_positive',0)) or 0)==1)
        # The following online proxy flags are not model features in v73.16 contract. They are kept only
        # to reproduce exact historical flattening behavior where present, and remain detector-free here.
        raw_pred=0; critical_pred=0; suspicious_pred=0
        output_pred=inf._output_alarm_from_row(metric_row)
        narrow_exception=inf._v19_patch_strong_exception_from_record(record)
        target=inf.v53_first_fault_target(record.get('faults', []))
        g=inf._fault_group_from_record(record)
        rr: Dict[str, Any] = {
            **{k:v for k,v in metric_row.items() if k != 'output_sanity_report'},
            'sample_id': inf.normalize_sample_token(sample_id),
            'row_source': 'experiment_injected' if injected else 'experiment_no_fault',
            'is_clean': 0,
            'is_experiment_row': 1,
            'is_fault_selected': int(record.get('is_fault_selected', record.get('is_positive',0)) or 0),
            'fault_spec_count': int(record.get('fault_spec_count',0) or 0),
            'fault_attempted_count': int(record.get('fault_attempted_count',0) or 0),
            'fault_applied_count': int(record.get('fault_applied_count',0) or 0),
            'fault_effective_count': int(record.get('fault_effective_count',0) or 0),
            'is_effective_injected': injected,
            'injected_effective': injected,
            'fault_target': target if target else 'none',
            'fault_group': g if injected else 'experiment_no_fault',
            'raw_input_alarm': raw_pred,
            'critical_input_alarm': critical_pred,
            'suspicious_input_alarm': suspicious_pred,
            'e_output_sanity_alarm': output_pred,
            'v11_p0_two_stage_harm_alarm': int(critical_pred == 1 or (suspicious_pred == 1 and output_pred == 1)),
            'v19_patch_strong_exception': narrow_exception,
            'v19_output_first_harm_alarm': int(output_pred == 1),
            'critical_reasons': [],
            'output_sanity_report': output_rep,
            'e_parse_features': e_parse,
            'object_vectors': record.get('object_vectors', {}),
            'faults': record.get('faults', []),
            'stage_summaries': record.get('stage_summaries', {}),
            'scene_context_summary': record.get('scene_context_summary', {}),
            'online_context_features': record.get('online_context_features', {}),
            'v26_online_head_features': record.get('v26_online_head_features', {}),
            'v26_agent_features': metric_row.get('v26_agent_features', {}) or inf.compute_v26_agent_context_features(sample_id, traj),
            'v26_ego_pose_features': metric_row.get('v26_ego_pose_features', {}) or inf.compute_v26_ego_pose_context_features(sample_id),
            'v35_behavior_features': inf.v35_behavior_features_from_plan_row(plan_row),
            'stage_feature_flat': inf.flatten_stage_summary_for_training(record.get('stage_summaries', {})),
            's0_feature_flat': inf._flatten_numeric_dict(record.get('features', {})),
        }
        rr['row_index']=len(self.rows_for_temporal)
        self.rows_for_temporal.append(rr)
        inf.add_v19_temporal_context_features(self.rows_for_temporal)
        rr=self.rows_for_temporal[-1]
        analysis={
            'parsed_traj': traj,
            'e_parse_features': e_parse,
            'output_sanity_features': output_sanity,
            'output_sanity_report': output_rep,
            'planning_metrics': planning_metrics,
            'clean_ref_available': int(inf.normalize_sample_token(sample_id) in self.clean_metrics),
            'posthoc_label_policy': 'labels_are_built_offline_only_not_detector_features',
            'delta_metrics': {k: metric_row.get(k) for k in metric_row.keys() if k.startswith('delta_') or k.startswith('clean_') or k.startswith('fault_')},
        }
        flat=inf.v53_deployable_feature_row_from_record(rr)
        return rr, analysis, flat

# v31-compatible behavior labels, copied into this no-detector exporter to avoid requiring score cache.
def clean_token(x: Any) -> str:
    s=str(x or '')
    return s[:-11] if s.endswith('_trajectory') else s

def pred_lookup(pred: Dict[str, Any], token: str) -> Optional[Any]:
    token=clean_token(token)
    for k in (token+'_trajectory', token):
        if k in pred:
            return pred[k]
    return None

def to_traj2(x: Any) -> Optional[np.ndarray]:
    try:
        arr=np.asarray(x, dtype=float)
        if arr.size == 0:
            return None
        arr=arr.reshape(-1,2)
        if arr.shape[0] > 6:
            arr=arr[:6]
        if not np.isfinite(arr).all() or arr.shape[0] < 2:
            return None
        return arr
    except Exception:
        return None

def gt_to_traj2(x: Any, mask: Any = None) -> Optional[np.ndarray]:
    try:
        arr=np.asarray(x, dtype=float)
        if arr.size == 0:
            return None
        arr=arr.reshape(-1,2)
        if arr.shape[0] > 6:
            arr=arr[:6]
        if mask is not None:
            m=np.asarray(mask).reshape(-1)
            if m.size >= arr.shape[0]:
                m=m[:arr.shape[0]].astype(bool)
                if m.any():
                    arr=arr[m]
        if not np.isfinite(arr).all() or arr.shape[0] < 2:
            return None
        return arr
    except Exception:
        return None

def angle_wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0*np.pi) - np.pi

def headings(traj: np.ndarray) -> np.ndarray:
    d=np.diff(traj, axis=0)
    return np.arctan2(d[:,1], d[:,0])

def path_length(traj: Optional[np.ndarray]) -> float:
    if traj is None or len(traj) < 2:
        return math.nan
    return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())

def step_norms(traj: Optional[np.ndarray]) -> np.ndarray:
    if traj is None or len(traj) < 2:
        return np.asarray([], dtype=float)
    return np.linalg.norm(np.diff(traj, axis=0), axis=1)

def curvature_proxy(traj: Optional[np.ndarray]) -> float:
    if traj is None or len(traj) < 3:
        return math.nan
    h=headings(traj)
    if len(h) < 2:
        return math.nan
    return float(np.abs(angle_wrap(np.diff(h))).mean())

def pair_metrics(a: Optional[np.ndarray], b: Optional[np.ndarray], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if a is None or b is None:
        out[f'{prefix}available']=0.0; return out
    n=min(len(a), len(b))
    if n < 2:
        out[f'{prefix}available']=0.0; return out
    a=a[:n]; b=b[:n]
    dist=np.linalg.norm(a-b, axis=1)
    ha=headings(a); hb=headings(b); m=min(len(ha), len(hb))
    hd=np.abs(angle_wrap(ha[:m]-hb[:m])) if m else np.asarray([], dtype=float)
    cos=np.cos(angle_wrap(ha[:m]-hb[:m])) if m else np.asarray([], dtype=float)
    sa=step_norms(a); sb=step_norms(b); sm=min(len(sa), len(sb))
    ca=curvature_proxy(a); cb=curvature_proxy(b)
    out.update({
        f'{prefix}available':1.0,
        f'{prefix}mean_point_dist':float(dist.mean()) if dist.size else math.nan,
        f'{prefix}max_point_dist':float(dist.max()) if dist.size else math.nan,
        f'{prefix}final_point_dist':float(dist[-1]) if dist.size else math.nan,
        f'{prefix}mean_heading_abs_delta':float(hd.mean()) if hd.size else math.nan,
        f'{prefix}max_heading_abs_delta':float(hd.max()) if hd.size else math.nan,
        f'{prefix}final_heading_abs_delta':float(hd[-1]) if hd.size else math.nan,
        f'{prefix}heading_cos_mean':float(cos.mean()) if cos.size else math.nan,
        f'{prefix}opposite_step_rate':float((cos < 0.0).mean()) if cos.size else math.nan,
        f'{prefix}mean_step_norm_abs_delta':float(np.abs(sa[:sm]-sb[:sm]).mean()) if sm else math.nan,
        f'{prefix}path_length_abs_delta':float(abs(path_length(a)-path_length(b))),
        f'{prefix}curvature_abs_delta':float(abs(ca-cb)) if math.isfinite(ca) and math.isfinite(cb) else math.nan,
    })
    return out

def flag_ge(x: Any, thr: float) -> bool:
    try:
        v=float(x)
        return math.isfinite(v) and v >= thr
    except Exception:
        return False

def intish(x: Any) -> int:
    try:
        if x is None or x == '':
            return 0
        return int(float(x))
    except Exception:
        return 0

def load_gt() -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    try:
        import eval_share.evaluation as evaluation_mod  # type: ignore
        current_dir=Path(evaluation_mod.__file__).resolve().parent
        with open(current_dir/'gt/gt_traj.pkl','rb') as f:
            gt=pickle.load(f)
        with open(current_dir/'gt/gt_traj_mask.pkl','rb') as f:
            mask=pickle.load(f)
        return gt, mask, str(current_dir)
    except Exception as e:
        return {}, {}, f'unavailable:{type(e).__name__}:{e}'

def parse_answer_to_traj(answer: Any) -> Optional[List[List[float]]]:
    try:
        if isinstance(answer, list) and answer:
            answer=answer[0]
        if isinstance(answer, str):
            s=answer.strip()
            try:
                obj=ast.literal_eval(s)
                arr=np.asarray(obj, dtype=float).reshape(-1,2)
                if arr.shape[0] >= 2:
                    return arr[:6].tolist()
            except Exception:
                pass
            nums=[float(x) for x in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', s)]
            if len(nums) >= 4:
                arr=np.asarray(nums[:12], dtype=float).reshape(-1,2)
                return arr[:6].tolist()
        arr=np.asarray(answer, dtype=float).reshape(-1,2)
        if arr.shape[0] >= 2:
            return arr[:6].tolist()
    except Exception:
        return None
    return None

def load_fault_preds_from_plan_conv(seed_dir: Path) -> Dict[str, Any]:
    path=seed_dir/'plan_conv.json'
    pred: Dict[str, Any] = {}
    if not path.is_file():
        return pred
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                row=json.loads(line)
            except Exception:
                continue
            sid=clean_token(str(row.get('id','')))
            traj=parse_answer_to_traj(row.get('answer',''))
            if sid and traj is not None:
                pred[sid]=traj
                pred[sid+'_trajectory']=traj
    return pred

def sanitize_value(v: Any, cap: float) -> Tuple[float, float, int, int]:
    vv=safe_float(v)
    missing=int(not math.isfinite(vv))
    raw=0.0 if missing else float(vv)
    extreme=int(abs(raw) > cap) if not missing else 1
    clipped=float(np.clip(raw, -cap, cap))
    slog=math.copysign(math.log1p(abs(clipped)), clipped)
    return clipped, slog, missing, extreme


AUX_MISSINGNESS_FEATURES = [
    'v73aux__runtime_raw_missing_any',
    'v73aux__runtime_raw_missing_count',
    'v73aux__runtime_raw_missing_rate',
    'v73aux__runtime_raw_missing_dllm_count',
    'v73aux__runtime_raw_missing_dout_count',
    'v73aux__runtime_raw_missing_s_count',
    'v73aux__runtime_raw_missing_c_count',
    'v73aux__runtime_raw_missing_other_count',
    'v73aux__fault_traj_missing',
    'v73aux__runtime_raw_missing_and_fault_traj_missing',
]

def _raw_missing_family(raw_feature: Any) -> str:
    s=str(raw_feature)
    if 'v59_d_llm' in s or 'd_llm' in s:
        return 'dllm'
    if 'v59_dout' in s or 'v63_dout' in s or 'lm_head' in s or 'decode_token' in s:
        return 'dout'
    if 'mm_projector_scene' in s or 'c_mm_projector' in s:
        return 'c'
    if 'v59_s' in s or 's_bev' in s or 'can_bus' in s or 'track' in s or 'seg' in s:
        return 's'
    return 'other'

def build_aux_missingness_features(source_df: pd.DataFrame, raw_missing_rows: List[Dict[str, Any]], label_df: pd.DataFrame, denominator: int) -> pd.DataFrame:
    """Build online-safe missingness indicators.

    These columns are not part of the v73.16 exact model-ready contract. They are saved as
    optional candidate features for future retraining so non-finite/missing runtime values
    are visible as missingness signals rather than being represented only by sanitized zeroes.
    They do not use harmful/severe labels, clean-vs-fault deltas, GT, or detector outputs.
    """
    keys=[c for c in ['seed','sample_id'] if c in source_df.columns]
    aux=source_df[keys].copy()
    if not keys:
        return pd.DataFrame()
    for c in AUX_MISSINGNESS_FEATURES:
        aux[c]=0.0
    denom=max(int(denominator), 1)
    if raw_missing_rows:
        rm=pd.DataFrame(raw_missing_rows)
        if 'seed' in rm.columns and 'sample_id' in rm.columns:
            rm['family']=rm.get('underlying_raw_feature', '').map(_raw_missing_family)
            cnt=rm.groupby(['seed','sample_id']).size().rename('v73aux__runtime_raw_missing_count').reset_index()
            aux=aux.merge(cnt, on=['seed','sample_id'], how='left', suffixes=('', '_new'))
            if 'v73aux__runtime_raw_missing_count_new' in aux.columns:
                aux['v73aux__runtime_raw_missing_count']=aux['v73aux__runtime_raw_missing_count_new'].fillna(aux['v73aux__runtime_raw_missing_count']).fillna(0.0)
                aux=aux.drop(columns=['v73aux__runtime_raw_missing_count_new'])
            fam=rm.groupby(['seed','sample_id','family']).size().unstack(fill_value=0).reset_index()
            rename={f:'v73aux__runtime_raw_missing_'+f+'_count' for f in ['dllm','dout','s','c','other'] if f in fam.columns}
            fam=fam.rename(columns=rename)
            aux=aux.merge(fam, on=['seed','sample_id'], how='left', suffixes=('', '_fam'))
            for f in ['dllm','dout','s','c','other']:
                col='v73aux__runtime_raw_missing_'+f+'_count'
                fam_col=col+'_fam'
                if fam_col in aux.columns:
                    aux[col]=aux[fam_col].fillna(aux[col]).fillna(0.0)
                    aux=aux.drop(columns=[fam_col])
    for c in [c for c in AUX_MISSINGNESS_FEATURES if c.endswith('_count')]:
        if c in aux.columns:
            aux[c]=pd.to_numeric(aux[c], errors='coerce').fillna(0.0)
    aux['v73aux__runtime_raw_missing_any']=(pd.to_numeric(aux['v73aux__runtime_raw_missing_count'], errors='coerce').fillna(0.0) > 0).astype(int)
    aux['v73aux__runtime_raw_missing_rate']=pd.to_numeric(aux['v73aux__runtime_raw_missing_count'], errors='coerce').fillna(0.0) / float(denom)
    if not label_df.empty and all(c in label_df.columns for c in ['seed','sample_id']):
        cols=['seed','sample_id']
        if 'beh__fault_traj_available' in label_df.columns:
            tmp=label_df[cols+['beh__fault_traj_available']].copy()
            tmp['v73aux__fault_traj_missing']=(1 - pd.to_numeric(tmp['beh__fault_traj_available'], errors='coerce').fillna(0).clip(0,1)).astype(int)
            tmp=tmp.drop(columns=['beh__fault_traj_available'])
            aux=aux.merge(tmp, on=['seed','sample_id'], how='left', suffixes=('', '_lbl'))
            if 'v73aux__fault_traj_missing_lbl' in aux.columns:
                aux['v73aux__fault_traj_missing']=aux['v73aux__fault_traj_missing_lbl'].fillna(aux['v73aux__fault_traj_missing']).fillna(0).astype(int)
                aux=aux.drop(columns=['v73aux__fault_traj_missing_lbl'])
    aux['v73aux__fault_traj_missing']=pd.to_numeric(aux['v73aux__fault_traj_missing'], errors='coerce').fillna(0).astype(int)
    aux['v73aux__runtime_raw_missing_and_fault_traj_missing']=(aux['v73aux__runtime_raw_missing_any'].astype(int) & aux['v73aux__fault_traj_missing'].astype(int)).astype(int)
    return aux[[c for c in keys + AUX_MISSINGNESS_FEATURES if c in aux.columns]]

def unsanitize_feature_name(sanitized_col: str) -> Tuple[str, str]:
    for p in V59_SANITIZE_PREFIXES:
        if sanitized_col.startswith(p):
            return p[:-2] if p.endswith('__') else p, sanitized_col[len(p):]
    return 'raw', sanitized_col

def is_forbidden_feature(c: str) -> bool:
    s=str(c)
    if s in FORBIDDEN_EXACT or s.startswith(FORBIDDEN_PREFIXES) or any(x in s for x in FORBIDDEN_SUBSTRINGS):
        return True
    return False

def build_behavior_labels(token: str, base_labels: Dict[str, Any], pred_fault: Dict[str, Any], pred_clean: Dict[str, Any], gt_dict: Dict[str, Any], gt_mask_dict: Dict[str, Any], thresholds: Dict[str, float]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    ft=to_traj2(pred_lookup(pred_fault, token))
    ct=to_traj2(pred_lookup(pred_clean, token))
    gt=gt_to_traj2(gt_dict.get(token), gt_mask_dict.get(token)) if token in gt_dict else None
    missing={
        'fault_traj_missing': int(ft is None),
        'clean_traj_missing': int(ct is None),
        'gt_traj_missing': int(gt is None),
    }
    out=dict(base_labels)
    out.update({
        'beh__fault_traj_available': 1 if ft is not None else 0,
        'beh__clean_traj_available': 1 if ct is not None else 0,
        'beh__gt_traj_available': 1 if gt is not None else 0,
    })
    for name, tr in [('fault', ft), ('clean', ct), ('gt', gt)]:
        if tr is not None:
            sn=step_norms(tr)
            out[f'beh__{name}_path_length']=path_length(tr)
            out[f'beh__{name}_final_disp']=float(np.linalg.norm(tr[-1]))
            out[f'beh__{name}_step_mean']=float(sn.mean()) if sn.size else math.nan
            out[f'beh__{name}_step_max']=float(sn.max()) if sn.size else math.nan
            out[f'beh__{name}_curvature_proxy']=curvature_proxy(tr)
    out.update(pair_metrics(ft, ct, 'beh__fault_vs_clean__'))
    out.update(pair_metrics(ft, gt, 'beh__fault_vs_gt__'))
    out.update(pair_metrics(ct, gt, 'beh__clean_vs_gt__'))
    for metric in ['mean_point_dist','max_point_dist','final_point_dist','mean_heading_abs_delta','max_heading_abs_delta','final_heading_abs_delta','mean_step_norm_abs_delta','path_length_abs_delta','curvature_abs_delta']:
        aa=out.get(f'beh__fault_vs_gt__{metric}', math.nan)
        bb=out.get(f'beh__clean_vs_gt__{metric}', math.nan)
        try:
            out[f'beh__degradation_vs_clean_gt__{metric}']=float(aa)-float(bb)
        except Exception:
            out[f'beh__degradation_vs_clean_gt__{metric}']=math.nan
    heading_clean_flag=flag_ge(out.get('beh__fault_vs_clean__mean_heading_abs_delta'), thresholds['heading_clean'])
    heading_gt_degrade_flag=flag_ge(out.get('beh__degradation_vs_clean_gt__mean_heading_abs_delta'), thresholds['heading_gt_degrade'])
    speed_clean_flag=flag_ge(out.get('beh__fault_vs_clean__mean_step_norm_abs_delta'), thresholds['speed_clean'])
    speed_gt_degrade_flag=flag_ge(out.get('beh__degradation_vs_clean_gt__mean_step_norm_abs_delta'), thresholds['speed_gt_degrade'])
    old=intish(out.get('harmful',0))
    heading=int(heading_clean_flag)
    heading_strict=int(bool(heading_clean_flag and heading_gt_degrade_flag))
    speed_harm=int(bool(speed_clean_flag and speed_gt_degrade_flag))
    out.update({
        'v31_old_harmful_compatible': old,
        'v31_heading_harmful_compatible': heading,
        'v31_heading_strict_harmful_compatible': heading_strict,
        'v31_heading_expanded_harmful_compatible': int(bool(old or heading)),
        'v31_speed_harmful_compatible': speed_harm,
        'v31_speed_alter_harmful_compatible': int(bool(old or speed_harm)),
        'v31_heading_speed_alter_harmful_compatible': int(bool(old or heading or speed_harm)),
        'beh__flag_fault_vs_clean_heading': int(heading_clean_flag),
        'beh__flag_gt_heading_degradation': int(heading_gt_degrade_flag),
        'beh__flag_fault_vs_clean_speed': int(speed_clean_flag),
        'beh__flag_gt_speed_degradation': int(speed_gt_degrade_flag),
    })
    return out, missing

HORIZON_ANALYSIS_COLUMNS = [
    'fault_l2_1s','fault_l2_2s','fault_l2_3s','fault_l2_avg',
    'clean_l2_1s','clean_l2_2s','clean_l2_3s','clean_l2_avg',
    'delta_l2_1s','delta_l2_2s','delta_l2_3s','delta_l2_avg',
    'fault_collision_1s','fault_collision_2s','fault_collision_3s','fault_collision_avg',
    'clean_collision_1s','clean_collision_2s','clean_collision_3s','clean_collision_avg',
    'delta_collision_1s','delta_collision_2s','delta_collision_3s','delta_collision_avg',
]

def json_dumps_compact(x: Any) -> str:
    try:
        if x is None:
            return ''
        return json.dumps(x, ensure_ascii=False, separators=(',', ':'))
    except Exception:
        return ''

def build_analysis_raw_ingredient_row(token: str, analysis: Dict[str, Any], clean_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten posthoc-only raw ingredients for sensitivity analysis.

    These columns are deliberately NOT part of model_features. They are kept so later
    analysis can rebuild harmful/severe/heading/speed variants without rerunning
    inference.
    """
    dm = analysis.get('delta_metrics', {}) or {}
    pm = analysis.get('planning_metrics', {}) or {}
    c = clean_metrics.get(token) or {}
    out: Dict[str, Any] = {
        'gt_metric_available': int(bool(pm.get('available', False))),
        'clean_ref_available': int(analysis.get('clean_ref_available', 0) or 0),
        'planning_metric_reason': pm.get('reason', ''),
        'fault_l2_t_json': json_dumps_compact(pm.get('l2_t')),
        'fault_collision_t_json': json_dumps_compact(pm.get('collision_t')),
        'clean_l2_t_json': json_dumps_compact(c.get('l2_t')),
        'clean_collision_t_json': json_dumps_compact(c.get('collision_t')),
    }
    for col in HORIZON_ANALYSIS_COLUMNS:
        out[col] = dm.get(col, math.nan)
    return out

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--campaign-dir', type=Path, required=True)
    ap.add_argument('--seed-root', required=True)
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--expected-rows', type=int, required=True)
    ap.add_argument('--resources-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--inference-module-path', type=Path, required=True)
    ap.add_argument('--clean-ref', type=Path, required=True)
    ap.add_argument('--output-sanity-json', type=Path, required=True)
    ap.add_argument('--clean-pred-trajs', type=Path, required=True)
    ap.add_argument('--abs-cap', type=float, default=1_000_000.0)
    ap.add_argument('--heading-clean-threshold', type=float, default=0.35)
    ap.add_argument('--heading-gt-degrade-threshold', type=float, default=0.20)
    ap.add_argument('--speed-clean-threshold', type=float, default=0.40)
    ap.add_argument('--speed-gt-degrade-threshold', type=float, default=0.20)
    ap.add_argument('--strict', action='store_true')
    args=ap.parse_args()

    out=args.output_dir; out.mkdir(parents=True, exist_ok=True)
    inf=import_inference_module(args.inference_module_path)
    filt=V7314StandaloneDataContractFilter(args.resources_dir)
    collect=pd.read_csv(args.resources_dir/'v73_9_hgb_runtime_collection_columns.csv')
    contract=pd.read_csv(args.resources_dir/'v73_9_hgb_online_feature_contract.csv')
    model_features=contract['feature'].drop_duplicates().tolist()
    source_features=collect['feature'].drop_duplicates().tolist()
    v63_cols=collect.loc[collect['source'].eq('v63_full'), 'feature'].drop_duplicates().tolist()
    old_cols=collect.loc[collect['source'].eq('old_exact'), 'feature'].drop_duplicates().tolist()
    bad_contract=[c for c in model_features if is_forbidden_feature(c)]
    if bad_contract:
        raise RuntimeError(f'Forbidden detector-derived model features in contract: {bad_contract[:20]}')

    seeds=[nseed(x) for x in args.seeds.split(',') if x.strip()]
    builder=NoDetectorRuntimeRowBuilder(inf, args.clean_ref, args.output_sanity_json, analysis_metrics=True)
    pred_clean=load_json(args.clean_pred_trajs)
    gt_dict, gt_mask_dict, gt_source=load_gt()
    beh_thr={
        'heading_clean': args.heading_clean_threshold,
        'heading_gt_degrade': args.heading_gt_degrade_threshold,
        'speed_clean': args.speed_clean_threshold,
        'speed_gt_degrade': args.speed_gt_degrade_threshold,
    }

    no_detector_records=[]
    source_rows=[]
    raw_source_rows=[]
    label_rows=[]
    analysis_rows=[]
    raw_missing_rows=[]  # Cells where a sanitized v63 feature had no finite underlying raw value. This is reported, not a hard failure.
    missing_counter=Counter()
    file_rows=[]
    forbidden_runtime_files=[]

    for seed in seeds:
        seed_dir=args.campaign_dir/args.seed_root/f'seed{seed}'
        results_dir=seed_dir/'results'
        # Detector outputs must not exist for this v73.16 run.
        for pat in ['v53_inline_detector_records*.jsonl','v53_inline_detector_summary.json','v54_inline_detector_records*.jsonl','detector_records*.jsonl','detector_report.json']:
            for p in results_dir.glob(pat):
                forbidden_runtime_files.append(str(p))
        rec_path=results_dir/'v73_16_feature_records.jsonl'
        plan_path=seed_dir/'plan_conv.json'
        if not rec_path.is_file():
            raise FileNotFoundError(rec_path)
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        plan_map=plan_lookup_from_file(plan_path, inf)
        pred_fault=load_fault_preds_from_plan_conv(seed_dir)
        if not pred_fault:
            missing_counter['missing_or_empty_fault_predictions'] += 1
        rows_this=0
        for rec in read_jsonl(rec_path):
            token=inf.normalize_sample_token(str(rec.get('sample_id','')))
            raw_id=str(rec.get('raw_sample_id') or rec.get('sample_id') or token)
            result=plan_map.get(token)
            if result is None:
                missing_counter['plan_conv_result_missing'] += 1
                result={'id': raw_id, 'answer': ''}
            rr, analysis, flat = builder.prepare_runtime_row(rec, result, token, raw_id)
            target=inf.v53_first_fault_target(rec.get('faults', []))
            exact_layer=inf.v53_exact_layer_from_fault_target(target)
            base_labels=inf.v53_analysis_label_row_from_deltas(
                token,
                rr.get('row_source', rec.get('row_source','')),
                int(rec.get('is_positive_effective',0) or 0),
                analysis.get('delta_metrics', {}) or {},
            )
            behavior_labels, miss = build_behavior_labels(token, base_labels, pred_fault, pred_clean, gt_dict, gt_mask_dict, beh_thr)
            analysis_flat = build_analysis_raw_ingredient_row(token, analysis, builder.clean_metrics)
            for k,v in miss.items():
                missing_counter[k] += int(v)
            label_row={
                'seed': seed,
                'sample_id': token,
                'raw_sample_id': raw_id,
                'row_source': rr.get('row_source',''),
                'exact_layer': exact_layer,
                'fault_target_exact': target,
                'fault_scope': rec.get('fault_scope',''),
                **behavior_labels,
                **analysis_flat,
            }
            label_rows.append(label_row)
            analysis_rows.append({
                'seed': seed,
                'sample_id': token,
                'raw_sample_id': raw_id,
                'row_source': rr.get('row_source',''),
                'exact_layer': exact_layer,
                'fault_target_exact': target,
                'fault_scope': rec.get('fault_scope',''),
                **analysis_flat,
                **{k: behavior_labels.get(k) for k in ['harmful','severe_harmful','v31_speed_alter_harmful_compatible'] if k in behavior_labels},
            })
            nd_record={
                'version':'v73_16_no_detector_feature_record',
                'seed': seed,
                'sample_id': token,
                'raw_sample_id': raw_id,
                'row_source': rr.get('row_source',''),
                'detector_timing': 'no_detector_not_applicable',
                'detector_runtime_disabled': 1,
                'analysis_only': {
                    'is_positive': int(rec.get('is_positive',0) or 0),
                    'is_positive_effective': int(rec.get('is_positive_effective',0) or 0),
                    'fault_scope_legacy': rec.get('fault_scope',''),
                    'fault_target_exact': target,
                    'exact_layer': exact_layer,
                    'faults': rec.get('faults', []),
                    'posthoc_label_policy': 'labels_are_built_offline_only_not_detector_features',
                    'delta_metrics': analysis.get('delta_metrics', {}),
                    'posthoc_labels': behavior_labels,
                    'planning_metrics': analysis.get('planning_metrics', {}),
                    'clean_ref_available': analysis.get('clean_ref_available', 0),
                    'gt_metric_available': int(bool((analysis.get('planning_metrics', {}) or {}).get('available', False))),
                    'e_parse_features': analysis.get('e_parse_features', {}),
                    'output_sanity_features': analysis.get('output_sanity_features', {}),
                    'output_sanity_report': analysis.get('output_sanity_report', {}),
                },
            }
            full_row={'seed': seed, 'sample_id': token, 'row_source': rr.get('row_source',''), 'exact_layer': exact_layer, 'fault_scope': rec.get('fault_scope',''), 'fault_target_exact': target}
            raw_full_row={'seed': seed, 'sample_id': token, 'row_source': rr.get('row_source',''), 'exact_layer': exact_layer, 'fault_scope': rec.get('fault_scope',''), 'fault_target_exact': target}
            # Direct old-exact features. These are already contract-level runtime/source features.
            for c in old_cols:
                full_row[c]=flat.get(c, math.nan)
                raw_full_row[c]=flat.get(c, math.nan)
            # v63 sanitized full-matrix features from exact raw stage feature names.
            # The sanitized columns remain part of the 2399 runtime contract, while the
            # corresponding underlying raw values are also saved in a separate raw-source matrix
            # for later audit/reprocessing. Missing/non-finite raw cells are valid data for severe
            # fault cases and are indicated by miss__/extreme__ contract columns.
            for c in v63_cols:
                kind, raw_name=unsanitize_feature_name(c)
                raw_val=flat.get(raw_name, math.nan)
                raw_full_row[raw_name]=raw_val
                clipped, slog, missv, extreme = sanitize_value(raw_val, args.abs_cap)
                if kind == 'clip':
                    full_row[c]=clipped
                elif kind == 'slog1p':
                    full_row[c]=slog
                elif kind == 'miss':
                    full_row[c]=missv
                elif kind == 'extreme':
                    full_row[c]=extreme
                elif kind == 'safe_raw':
                    full_row[c]=clipped
                else:
                    full_row[c]=flat.get(c, math.nan)
                if missv:
                    raw_missing_rows.append({'seed': seed, 'sample_id': token, 'requested_column': c, 'underlying_raw_feature': raw_name})
            # Contract filter guarantees consistent column order and emits every requested source feature.
            source_row=filt.filter_feature_row(full_row, keep_audit=True)
            source_rows.append(source_row)
            raw_source_rows.append(raw_full_row)
            # Store only the minimal, contract-relevant runtime row in the JSONL record. Do not store
            # legacy deployable_feature_row, detector-score placeholders, or large unfiltered legacy fields.
            nd_record['runtime_source_feature_row']=source_row
            no_detector_records.append(nd_record)
            rows_this += 1
        file_rows.append({'seed': seed, 'records_path': str(rec_path), 'plan_conv': str(plan_path), 'rows': rows_this})

    if forbidden_runtime_files:
        pd.DataFrame({'forbidden_runtime_file': forbidden_runtime_files}).to_csv(out/'v73_16_forbidden_detector_runtime_files.csv', index=False)
        raise RuntimeError('Detector runtime files were produced in a no-detector run; see v73_16_forbidden_detector_runtime_files.csv')

    feature_records_path=out/'v73_16_no_detector_feature_records.jsonl'
    with feature_records_path.open('w', encoding='utf-8') as f:
        for r in no_detector_records:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    label_df=pd.DataFrame(label_rows)
    label_path=out/'v73_16_posthoc_label_table_multi_policy.csv'
    label_df.to_csv(label_path, index=False)
    analysis_df=pd.DataFrame(analysis_rows)
    analysis_path=out/'v73_16_posthoc_analysis_raw_ingredients.csv'
    analysis_df.to_csv(analysis_path, index=False)

    source_df=pd.DataFrame(source_rows)
    # Enforce exact source schema.
    for c in KEY_COLS + source_features:
        if c not in source_df.columns:
            source_df[c]=math.nan
    source_cols=[c for c in ['seed','sample_id','row_source','exact_layer','fault_scope','fault_target_exact'] if c in source_df.columns] + source_features
    source_df=source_df[source_cols]
    source_path=out/'v73_16_runtime_source_matrix.csv'
    source_df.to_csv(source_path, index=False)

    raw_source_df=pd.DataFrame(raw_source_rows)
    raw_source_cols=[c for c in ['seed','sample_id','row_source','exact_layer','fault_scope','fault_target_exact'] if c in raw_source_df.columns] + sorted([c for c in raw_source_df.columns if c not in {'seed','sample_id','row_source','exact_layer','fault_scope','fault_target_exact'}])
    raw_source_df=raw_source_df[raw_source_cols] if len(raw_source_df.columns) else raw_source_df
    raw_source_path=out/'v73_16_underlying_raw_source_matrix.csv'
    raw_source_df.to_csv(raw_source_path, index=False)
    raw_source_numeric=raw_source_df[[c for c in raw_source_df.columns if c not in {'seed','sample_id','row_source','exact_layer','fault_scope','fault_target_exact'}]].apply(pd.to_numeric, errors='coerce') if len(raw_source_df.columns) else pd.DataFrame()
    raw_source_all_nan_columns=[c for c in raw_source_numeric.columns if int(raw_source_numeric[c].notna().sum()) == 0]

    G=filt.compute_generated_for_frame(source_df)
    gen_path=out/'v73_16_generated_consistency_matrix.csv'
    pd.concat([source_df[[c for c in KEY_COLS if c in source_df.columns]], G], axis=1).to_csv(gen_path, index=False)
    non_gen=[c for c in model_features if not c.startswith('v73gen__')]
    gen_cols=[c for c in model_features if c.startswith('v73gen__')]
    missing_non_gen=[c for c in non_gen if c not in source_df.columns]
    if missing_non_gen:
        pd.DataFrame({'missing_non_generated_model_feature': missing_non_gen}).to_csv(out/'v73_16_missing_non_generated_model_features.csv', index=False)
        raise RuntimeError(f'Missing non-generated model features: {len(missing_non_gen)}')
    model_df=pd.concat([source_df[[c for c in ['seed','sample_id','row_source','exact_layer','fault_scope','fault_target_exact'] if c in source_df.columns]], source_df[non_gen], G[gen_cols]], axis=1)
    if not label_df.empty:
        label_join_cols=[c for c in label_df.columns if c not in model_df.columns or c in KEY_COLS]
        model_df=model_df.merge(label_df[label_join_cols], on=KEY_COLS, how='left', validate='one_to_one')
    model_path=out/'v73_16_model_ready_matrix_with_labels.csv'
    model_df.to_csv(model_path, index=False)

    # Optional online-safe missingness indicators for future retraining. The exact v73.16
    # model-ready matrix above is preserved unchanged; this plus-aux matrix makes missing
    # raw values and parse failure visible to new models without using labels or GT/clean deltas.
    aux_df=build_aux_missingness_features(source_df, raw_missing_rows, label_df, denominator=max(1, len(v63_cols)))
    aux_path=out/'v73_16_aux_missingness_features.csv'
    aux_df.to_csv(aux_path, index=False)
    aux_manifest_path=out/'v73_16_aux_missingness_feature_manifest.csv'
    pd.DataFrame([{'feature': c, 'role': 'online_aux_candidate_not_in_v73_16_exact_model_ready_contract'} for c in AUX_MISSINGNESS_FEATURES]).to_csv(aux_manifest_path, index=False)
    source_plus_aux_path=out/'v73_16_runtime_source_plus_aux_matrix.csv'
    source_df.merge(aux_df, on=KEY_COLS, how='left').to_csv(source_plus_aux_path, index=False)
    model_plus_aux_path=out/'v73_16_model_ready_plus_aux_matrix_with_labels.csv'
    model_df.merge(aux_df, on=KEY_COLS, how='left').to_csv(model_plus_aux_path, index=False)

    # Audits.
    comp=[]
    X=model_df[[c for c in model_features if c in model_df.columns]].apply(pd.to_numeric, errors='coerce')
    for c in model_features:
        if c not in X.columns:
            comp.append({'feature': c, 'present': 0, 'finite_count': 0, 'row_count': int(len(model_df)), 'finite_rate': 0.0, 'all_nan': True})
        else:
            arr=X[c].to_numpy(dtype=float)
            finite=np.isfinite(arr)
            comp.append({'feature': c, 'present': 1, 'finite_count': int(finite.sum()), 'row_count': int(len(model_df)), 'finite_rate': float(finite.mean()) if len(arr) else 0.0, 'all_nan': bool(finite.sum()==0)})
    comp_df=pd.DataFrame(comp).sort_values(['present','all_nan','finite_rate'], ascending=[True,False,True])
    comp_df.to_csv(out/'v73_16_feature_completeness.csv', index=False)
    pd.DataFrame(raw_missing_rows).to_csv(out/'v73_16_underlying_raw_missing_audit.csv', index=False)

    rows_total=len(model_df)
    expected_total=args.expected_rows * len(seeds)
    missing_feature_count=int((comp_df['present'] == 0).sum())
    all_nan_feature_count=int(comp_df['all_nan'].sum())
    forbidden_feature_count=int(sum(1 for c in model_features if is_forbidden_feature(c)))
    label_totals={}
    for lab in ['harmful','severe_harmful','v31_speed_alter_harmful_compatible']:
        if lab in model_df.columns:
            label_totals[lab]=int(pd.to_numeric(model_df[lab], errors='coerce').fillna(0).astype(int).sum())
    required_analysis_cols = HORIZON_ANALYSIS_COLUMNS + ['fault_l2_t_json','fault_collision_t_json','clean_l2_t_json','clean_collision_t_json','gt_metric_available','clean_ref_available']
    analysis_missing_columns = [c for c in required_analysis_cols if c not in model_df.columns]
    analysis_all_nan_columns = []
    for c in HORIZON_ANALYSIS_COLUMNS:
        if c in model_df.columns:
            vals=pd.to_numeric(model_df[c], errors='coerce')
            if int(vals.notna().sum()) == 0:
                analysis_all_nan_columns.append(c)
    availability={k:int(v) for k,v in missing_counter.items()}
    pass_flag=bool(
        rows_total == expected_total
        and missing_feature_count == 0
        and forbidden_feature_count == 0
        and all_nan_feature_count == 0
        # Do not fail on fault_traj_missing or underlying raw missing cells: malformed / all-zero
        # fault outputs and non-finite activations are valid severe-fault outcomes. They are saved
        # as analysis signals and miss__/extreme__ feature flags.
        and availability.get('clean_traj_missing',0) == 0
        and availability.get('gt_traj_missing',0) == 0
        and availability.get('plan_conv_result_missing',0) == 0
        and len(analysis_missing_columns) == 0
        and len(analysis_all_nan_columns) == 0
    )
    summary={
        'version':'v73_16_standalone_no_detector_data_export',
        'campaign_dir':str(args.campaign_dir),
        'seed_root':args.seed_root,
        'seeds':seeds,
        'rows':rows_total,
        'expected_rows_total':expected_total,
        'file_rows':file_rows,
        'runtime_source_feature_count':int(len(source_features)),
        'model_feature_count':int(len(model_features)),
        'generated_model_feature_count':int(len(gen_cols)),
        'aux_missingness_feature_count':int(len(AUX_MISSINGNESS_FEATURES)),
        'missing_feature_count':missing_feature_count,
        'all_nan_feature_count':all_nan_feature_count,
        'forbidden_feature_count':forbidden_feature_count,
        'underlying_raw_missing_cell_count_report_only':int(len(raw_missing_rows)),
        'underlying_raw_source_column_count':int(max(0, len(raw_source_df.columns)-len([c for c in ['seed','sample_id','row_source','exact_layer','fault_scope','fault_target_exact'] if c in raw_source_df.columns]))),
        'underlying_raw_source_all_nan_column_count':int(len(raw_source_all_nan_columns)),
        'analysis_raw_ingredient_column_count':int(len(required_analysis_cols)),
        'analysis_missing_columns':analysis_missing_columns,
        'analysis_all_nan_columns':analysis_all_nan_columns,
        'horizon_metric_columns':HORIZON_ANALYSIS_COLUMNS,
        'label_totals':label_totals,
        'availability':availability,
        'gt_source':gt_source,
        'clean_ref_available':builder.clean_ref_available,
        'clean_ref_reason':builder.clean_ref_reason,
        'output_sanity_thresholds_reason':builder.output_sanity_thresholds_reason,
        'metric_runtime_available':builder.metric_runtime.available,
        'metric_runtime_reason':builder.metric_runtime.reason,
        'detector_runtime_disabled':True,
        'detector_artifact_loaded':False,
        'outputs':{
            'no_detector_feature_records_jsonl':str(feature_records_path),
            'runtime_source_matrix_csv':str(source_path),
            'generated_consistency_matrix_csv':str(gen_path),
            'model_ready_matrix_with_labels_csv':str(model_path),
            'posthoc_label_table_multi_policy_csv':str(label_path),
            'posthoc_analysis_raw_ingredients_csv':str(analysis_path),
            'feature_completeness_csv':str(out/'v73_16_feature_completeness.csv'),
            'underlying_raw_missing_audit_csv':str(out/'v73_16_underlying_raw_missing_audit.csv'),
            'underlying_raw_source_matrix_csv':str(raw_source_path),
            'aux_missingness_features_csv':str(aux_path),
            'aux_missingness_feature_manifest_csv':str(aux_manifest_path),
            'runtime_source_plus_aux_matrix_csv':str(source_plus_aux_path),
            'model_ready_plus_aux_matrix_with_labels_csv':str(model_plus_aux_path),
        },
        'pass':pass_flag,
    }
    write_json(out/'v73_16_no_detector_export_summary.json', summary)
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.strict and not pass_flag:
        raise SystemExit(7)

if __name__ == '__main__':
    main()
