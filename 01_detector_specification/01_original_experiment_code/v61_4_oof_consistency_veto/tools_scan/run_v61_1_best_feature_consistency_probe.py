#!/usr/bin/env python3
"""
v61.2 best-feature consistency probe with robust label fallback.

Posthoc only: does not rerun OpenDriveVLA inference.

Purpose:
  v61 tested consistency signals against the raw ExtraTrees base.  v61.2 tests whether
  those consistency signals add marginal value on top of the strongest currently-available
  feature/model pipelines:
    - selected v35.1 detector score/alarm features
    - old exact v35.1 feature matrix when available
    - v59 sanitized D_LLM/D_out/S4 targeted feature matrix
    - old+new+scores mixtures

Outputs compare:
  candidate alarm
  candidate OR consistency alarm
  candidate AND NOT detector-disagreement veto
  candidate OR consistency AND NOT detector-disagreement veto

This is a quickcheck/probe. With 2 seeds it is diagnostic only.
"""
import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

warnings.filterwarnings("ignore")

DETECTORS = [
    "lineA_old_extra_trees_high_recall_online_all_no_temporal",
    "lineA_old_hgb_online_all_no_temporal",
    "lineA_speed_alter_hgb_online_compact",
]
BASE_DETECTOR = "lineA_old_extra_trees_high_recall_online_all_no_temporal"
LABELS = ["harmful", "severe_harmful", "v31_speed_alter_harmful_compatible"]
ALL_LABELS = set([
    "harmful", "severe_harmful", "v31_old_harmful_compatible",
    "v31_heading_expanded_harmful_compatible", "v31_speed_alter_harmful_compatible",
    "v31_heading_speed_alter_harmful_compatible",
])
ID_COLS = {"seed", "sample_id", "raw_sample_id", "row_source", "exact_layer", "scope", "scope_group"}
LEAK_PATTERNS = [
    "harmful", "label", "delta_", "collision", "gt_", "clean_",
    "fault_target", "fault_scope", "bit_op", "bit_count", "is_positive",
    "effective", "applied", "attempted", "fault0_", "det_fault",
]


def parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def normalize_sample_id(x) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.endswith("_trajectory"):
        s = s[:-len("_trajectory")]
    return s


def prep_keys(df: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].map(normalize_sample_id)
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype(int)
    return df


def filter_seeds(df: pd.DataFrame, seeds: List[int]) -> pd.DataFrame:
    if "seed" in df.columns:
        return df[pd.to_numeric(df["seed"], errors="coerce").astype("Int64").isin(seeds)].copy()
    return df.copy()


def read_csv(p: Path, required: bool = True) -> Optional[pd.DataFrame]:
    if not p.exists():
        if required:
            raise FileNotFoundError(str(p))
        return None
    return pd.read_csv(p)


def scope_group_from_layer(x) -> str:
    s = str(x)
    if s in ["D_LLM", "D_out", "C_projector", "S0_canbus", "S1", "S2", "S3", "S4", "S5", "no_fault"]:
        return s
    if s.startswith("D"):
        return "D_LLM"
    if s.startswith("C"):
        return "C_projector"
    if s.startswith("S0"):
        return "S0_canbus"
    if s.startswith("S1"):
        return "S1"
    if s.startswith("S2"):
        return "S2"
    if s.startswith("S3"):
        return "S3"
    if s.startswith("S4"):
        return "S4"
    if s.startswith("S5"):
        return "S5"
    if "no_fault" in s:
        return "no_fault"
    return s


def is_feature_col(c: str) -> bool:
    if c in ID_COLS or c in ALL_LABELS:
        return False
    low = c.lower()
    if any(p in low for p in LEAK_PATTERNS):
        if not (c.startswith("score__") or c.startswith("alarm__") or c.startswith("threshold__")):
            return False
    return True


def sanitize_feature_names(cols: Iterable[str]) -> List[str]:
    return [str(c) for c in cols if is_feature_col(str(c))]


def load_v59_sanitized(formal_dir: Path, seeds: List[int]) -> pd.DataFrame:
    p = formal_dir / "v59_targeted_feature_audit" / "v59_targeted_feature_matrix_sanitized.csv"
    df = read_csv(p, required=True)
    df = filter_seeds(df, seeds)
    df = prep_keys(df)
    if "exact_layer" not in df.columns:
        df["exact_layer"] = "unknown"
    df["scope_group"] = df["exact_layer"].map(scope_group_from_layer)
    return df


def merge_label_table(formal_dir: Path, df: pd.DataFrame, seeds: List[int]) -> pd.DataFrame:
    """Merge per-sample posthoc labels.

    The quickcheck directories used in v59/v60 may contain either a wide label table,
    a long label table, or no label table at all.  v61.1 assumed the wide table always
    existed; that caused KeyError('harmful') when the quickcheck pack omitted it.
    v61.2 keeps the wide-table path but adds robust fallbacks.
    """
    candidate_paths = [
        formal_dir / "v54_online_exports" / "v54_2_posthoc_label_table_multi_policy.csv",
        formal_dir / "v54_online_exports" / "v54_posthoc_label_table_harmful_severe.csv",
        formal_dir / "v54_online_exports" / "v53_posthoc_label_table_harmful_severe.csv",
    ]
    for p in candidate_paths:
        lab = read_csv(p, required=False)
        if lab is None or lab.empty or "sample_id" not in lab.columns:
            continue
        lab = filter_seeds(lab, seeds)
        lab = prep_keys(lab)
        keys = [k for k in ["seed", "sample_id"] if k in df.columns and k in lab.columns]
        if "sample_id" not in keys:
            continue

        # Wide format: columns are directly harmful/severe/v31...
        wide_cols = [c for c in lab.columns if c in ALL_LABELS and c not in df.columns]
        if wide_cols:
            add_cols = keys + wide_cols
            df = df.merge(lab[add_cols].drop_duplicates(keys), on=keys, how="left")

        # Long format fallback: columns like label,value or label,label_value.
        label_col = next((c for c in ["label", "label_name", "policy", "label_policy"] if c in lab.columns), None)
        value_col = next((c for c in ["value", "label_value", "is_positive", "y", "target"] if c in lab.columns), None)
        if label_col and value_col:
            tmp = lab[keys + [label_col, value_col]].copy()
            tmp = tmp[tmp[label_col].astype(str).isin(ALL_LABELS)]
            if not tmp.empty:
                piv = tmp.pivot_table(index=keys, columns=label_col, values=value_col, aggfunc="max").reset_index()
                piv.columns = [str(c) for c in piv.columns]
                new_cols = [c for c in piv.columns if c in ALL_LABELS and c not in df.columns]
                if new_cols:
                    df = df.merge(piv[keys + new_cols].drop_duplicates(keys), on=keys, how="left")

    missing = [c for c in LABELS if c not in df.columns]
    if missing:
        df = merge_labels_from_inline_records(formal_dir, df, seeds, missing)
    return df


def _recursive_find_label(obj, label: str, depth: int = 0):
    if depth > 5:
        return None
    if isinstance(obj, dict):
        if label in obj:
            return obj.get(label)
        # Common nested containers first.
        for k in ["analysis_only", "labels", "posthoc_labels", "analysis_labels", "label_table", "metrics", "analysis"]:
            if k in obj:
                v = _recursive_find_label(obj[k], label, depth + 1)
                if v is not None:
                    return v
        for v0 in obj.values():
            v = _recursive_find_label(v0, label, depth + 1)
            if v is not None:
                return v
    elif isinstance(obj, list):
        for item in obj[:20]:
            v = _recursive_find_label(item, label, depth + 1)
            if v is not None:
                return v
    return None


def merge_labels_from_inline_records(formal_dir: Path, df: pd.DataFrame, seeds: List[int], missing_labels: List[str]) -> pd.DataFrame:
    """Last-resort label fallback from v53 inline records.

    This keeps v61.2 usable on quickcheck directories where the compact return package
    omitted the posthoc label table but inline records still contain analysis labels.
    """
    seed_root = formal_dir / "A1_internal_neural_scdout_v54_2_online_singlefault"
    rows = []
    for seed in seeds:
        p = seed_root / f"seed{seed}" / "results" / "v53_inline_detector_records.jsonl"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = normalize_sample_id(obj.get("sample_id") or obj.get("raw_sample_id"))
                if not sid:
                    continue
                r = {"seed": seed, "sample_id": sid}
                any_found = False
                for lab in missing_labels:
                    val = _recursive_find_label(obj, lab)
                    if val is not None:
                        r[lab] = val
                        any_found = True
                if any_found:
                    rows.append(r)
    if rows:
        labdf = pd.DataFrame(rows).drop_duplicates(["seed", "sample_id"])
        keys = [k for k in ["seed", "sample_id"] if k in df.columns and k in labdf.columns]
        add_cols = keys + [c for c in missing_labels if c in labdf.columns and c not in df.columns]
        if "sample_id" in keys and len(add_cols) > len(keys):
            df = df.merge(labdf[add_cols], on=keys, how="left")
    return df


def load_old_exact_features(formal_dir: Path, seeds: List[int], base_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
    for p in [
        formal_dir / "v54_online_exports" / "v54_online_deployable_feature_matrix.csv",
        formal_dir / "v54_online_exports" / "v53_online_deployable_feature_matrix.csv",
    ]:
        if p.exists():
            old = pd.read_csv(p)
            old = filter_seeds(old, seeds)
            old = prep_keys(old)
            keys = [k for k in ["seed", "sample_id"] if k in base_df.columns and k in old.columns]
            if "sample_id" not in keys:
                return base_df, [], f"no_keys:{p}"
            old_cols = sanitize_feature_names([c for c in old.columns if c not in keys])
            rename = {c: f"old__{c}" for c in old_cols}
            old2 = old[keys + old_cols].drop_duplicates(keys).rename(columns=rename)
            out = base_df.merge(old2, on=keys, how="left")
            return out, list(rename.values()), str(p)
    return base_df, [], "missing"


def load_diagnostics_subset(formal_dir: Path, seeds: List[int], base_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
    p = formal_dir / "v54_online_exports" / "v54_online_all_deployable_diagnostics_matrix.csv"
    if not p.exists():
        return base_df, [], "missing"
    diag = pd.read_csv(p)
    diag = filter_seeds(diag, seeds)
    diag = prep_keys(diag)
    keys = [k for k in ["seed", "sample_id"] if k in base_df.columns and k in diag.columns]
    if "sample_id" not in keys:
        return base_df, [], "no_keys"
    patterns = [
        "traj", "path", "final_disp", "step", "waypoint", "parse", "zero", "repeat",
        "ego", "speed", "heading", "yaw", "turn", "curv", "disp", "motion", "velocity",
    ]
    cols = []
    for c in diag.columns:
        if c in keys:
            continue
        low = str(c).lower()
        if any(pat in low for pat in patterns) and is_feature_col(str(c)):
            cols.append(c)
    if not cols:
        return base_df, [], "no_matching_cols"
    rename = {c: f"diag__{c}" for c in cols}
    diag2 = diag[keys + cols].drop_duplicates(keys).rename(columns=rename)
    out = base_df.merge(diag2, on=keys, how="left")
    return out, list(rename.values()), str(p)


def load_score_cache(formal_dir: Path, seeds: List[int]) -> Tuple[pd.DataFrame, str]:
    p = formal_dir / "v54_online_exports" / "v54_detector_score_cache.csv"
    if not p.exists():
        return pd.DataFrame(), "missing_score_cache"
    df = pd.read_csv(p)
    df = filter_seeds(df, seeds)
    df = prep_keys(df)
    det_col = next((c for c in ["detector", "detector_name", "model", "model_name"] if c in df.columns), None)
    score_col = next((c for c in ["score", "detector_score", "prob", "prediction_score"] if c in df.columns), None)
    alarm_col = next((c for c in ["alarm", "detector_alarm", "pred", "prediction"] if c in df.columns), None)
    keys = [k for k in ["seed", "sample_id"] if k in df.columns]
    if det_col and score_col:
        pieces = []
        w = df.pivot_table(index=keys, columns=det_col, values=score_col, aggfunc="first").reset_index()
        w = w.rename(columns={c: f"score__{c}" for c in w.columns if c not in keys})
        pieces.append(w)
        if alarm_col:
            a = df.pivot_table(index=keys, columns=det_col, values=alarm_col, aggfunc="first").reset_index()
            a = a.rename(columns={c: f"alarm__{c}" for c in a.columns if c not in keys})
            pieces.append(a)
        out = pieces[0]
        for z in pieces[1:]:
            out = out.merge(z, on=keys, how="outer")
        return out, str(p)
    score_like = [c for c in df.columns if c.startswith("score__") or c.startswith("alarm__") or c.startswith("threshold__")]
    if score_like:
        return df[keys + score_like].drop_duplicates(keys), str(p)
    return pd.DataFrame(), "unrecognized_score_cache"


def parse_inline_scores(formal_dir: Path, seeds: List[int]) -> Tuple[pd.DataFrame, str]:
    root = formal_dir / "A1_internal_neural_scdout_v54_2_online_singlefault"
    rows = []
    for seed in seeds:
        p = root / f"seed{seed}" / "results" / "v53_inline_detector_records.jsonl"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = normalize_sample_id(obj.get("sample_id") or obj.get("raw_sample_id"))
                r = {"seed": seed, "sample_id": sid}
                dets = obj.get("detectors") or {}
                if isinstance(dets, list):
                    dets = {str(d.get("name") or d.get("detector") or i): d for i, d in enumerate(dets) if isinstance(d, dict)}
                if isinstance(dets, dict):
                    for det in DETECTORS:
                        d = dets.get(det) or {}
                        if isinstance(d, dict):
                            r[f"score__{det}"] = d.get("score", np.nan)
                            r[f"alarm__{det}"] = d.get("alarm", np.nan)
                            r[f"threshold__{det}"] = d.get("threshold", np.nan)
                rows.append(r)
    if not rows:
        return pd.DataFrame(), "inline_missing_or_unrecognized"
    return pd.DataFrame(rows).drop_duplicates(["seed", "sample_id"]), "inline_records"


def merge_scores(formal_dir: Path, df: pd.DataFrame, seeds: List[int]) -> Tuple[pd.DataFrame, List[str], str]:
    scores, src = load_score_cache(formal_dir, seeds)
    if scores.empty:
        scores, src = parse_inline_scores(formal_dir, seeds)
    if scores.empty:
        raise FileNotFoundError("Could not load detector score/alarm features")
    scores = prep_keys(scores)
    keys = [k for k in ["seed", "sample_id"] if k in df.columns and k in scores.columns]
    score_cols = [c for c in scores.columns if c.startswith("score__") or c.startswith("alarm__") or c.startswith("threshold__")]
    out = df.merge(scores[keys + score_cols].drop_duplicates(keys), on=keys, how="left")
    return out, [c for c in score_cols if c in out.columns], src


def ensure_base_alarm(df: pd.DataFrame) -> pd.DataFrame:
    base_alarm = f"alarm__{BASE_DETECTOR}"
    base_score = f"score__{BASE_DETECTOR}"
    if base_alarm not in df.columns or df[base_alarm].isna().all():
        if base_score not in df.columns:
            raise ValueError("missing base ExtraTrees score/alarm")
        df[base_alarm] = (pd.to_numeric(df[base_score], errors="coerce").fillna(0.0) >= 0.2).astype(int)
    df[base_alarm] = pd.to_numeric(df[base_alarm], errors="coerce").fillna(0).astype(int)
    for det in DETECTORS:
        sc = f"score__{det}"
        al = f"alarm__{det}"
        if sc in df.columns:
            df[sc] = pd.to_numeric(df[sc], errors="coerce").fillna(0.0)
        if al in df.columns:
            df[al] = pd.to_numeric(df[al], errors="coerce").fillna(0).astype(int)
    return df


def make_model(name: str, seed: int = 0):
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=0.05, random_state=seed)
    if name == "extra_trees_depth5":
        return ExtraTreesClassifier(n_estimators=160, max_depth=5, min_samples_leaf=3, n_jobs=-1, class_weight="balanced_subsample", random_state=seed)
    raise ValueError(name)


def build_pipeline(model_name: str, k: int, seed: int = 0):
    def sanitize_np(X):
        X = np.asarray(X, dtype=np.float64)
        X[~np.isfinite(X)] = np.nan
        return np.clip(X, -1e6, 1e6)
    steps = [
        ("sanitize", FunctionTransformer(sanitize_np, validate=False)),
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
    ]
    if k and k > 0:
        steps.append(("select", SelectKBest(score_func=f_classif, k=k)))
    steps.append(("model", make_model(model_name, seed)))
    return Pipeline(steps)


def predict_score(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == 2:
            return p[:, 1]
        return np.zeros(X.shape[0])
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1 / (1 + np.exp(-s))
    return model.predict(X).astype(float)


def choose_threshold_for_recall(y_true: np.ndarray, score: np.ndarray, target_recall: float) -> float:
    y_true = y_true.astype(int)
    if y_true.sum() == 0:
        return 1.0
    vals = np.unique(score[np.isfinite(score)])
    if len(vals) == 0:
        return 1.0
    best_thr = float(vals.min() - 1e-12)
    best_fp = math.inf
    for thr in vals:
        pred = score >= thr
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        if rec + 1e-12 >= target_recall:
            if fp < best_fp or (fp == best_fp and float(thr) > best_thr):
                best_fp = fp
                best_thr = float(thr)
    return best_thr


def metrics(y, pred) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "far": fp / (fp + tn) if (fp + tn) else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
    }


def scope_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    scopes = ["ALL", "D_LLM", "D_out", "S4", "S1", "S3", "S0_canbus", "S2", "no_fault"]
    masks = {"ALL": np.ones(len(df), dtype=bool)}
    if "scope_group" in df.columns:
        s = df["scope_group"].astype(str)
    elif "exact_layer" in df.columns:
        s = df["exact_layer"].astype(str).map(scope_group_from_layer)
    else:
        s = pd.Series(["unknown"] * len(df))
    for scope in scopes[1:]:
        masks[scope] = s.eq(scope).to_numpy()
    return masks


def add_scope_results(rows: List[Dict], df: pd.DataFrame, pred: np.ndarray, label: str, meta: Dict, base_pred: Optional[np.ndarray] = None):
    if label not in df.columns:
        return
    y_all = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int).to_numpy()
    masks = scope_masks(df)
    for scope, mask in masks.items():
        if mask.sum() == 0:
            continue
        # keep no_fault and ALL; skip tiny empty scope rows for readability
        if scope not in ["ALL", "no_fault"] and y_all[mask].sum() == 0 and pred[mask].sum() == 0:
            continue
        r = metrics(y_all[mask], pred[mask])
        r.update(meta)
        r.update({"label": label, "scope": scope, "total": int(mask.sum()), "positive_total": int(y_all[mask].sum()), "negative_total": int((y_all[mask] == 0).sum())})
        if base_pred is not None:
            b = base_pred[mask]
            y = y_all[mask]
            bm = metrics(y, b)
            r["base_tp"] = bm["tp"]; r["base_fn"] = bm["fn"]; r["base_fp"] = bm["fp"]
            r["base_recall"] = bm["recall"]; r["base_far"] = bm["far"]
            r["recovered_fn_vs_base"] = int(((b == 0) & (pred[mask] == 1) & (y == 1)).sum())
            r["added_fp_vs_base"] = int(((b == 0) & (pred[mask] == 1) & (y == 0)).sum())
            r["filtered_fp_vs_base"] = int(((b == 1) & (pred[mask] == 0) & (y == 0)).sum())
            r["vetoed_tp_vs_base"] = int(((b == 1) & (pred[mask] == 0) & (y == 1)).sum())
        rows.append(r)


def numeric_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)
    return X.to_numpy(dtype=np.float32)


def robust_group_scores(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if not cols:
        return np.zeros(len(train)), np.zeros(len(test))
    Xtr = numeric_matrix(train, cols)
    Xte = numeric_matrix(test, cols)
    med = np.nanmedian(Xtr, axis=0)
    mad = np.nanmedian(np.abs(Xtr - med), axis=0)
    scale = 1.4826 * mad
    std = np.nanstd(Xtr, axis=0)
    scale = np.where(scale > 1e-9, scale, np.where(std > 1e-9, std, 1.0))
    Ztr = np.abs((Xtr - med) / scale)
    Zte = np.abs((Xte - med) / scale)
    def topk_mean(Z, k=5):
        if Z.shape[1] == 0:
            return np.zeros(Z.shape[0])
        kk = min(k, Z.shape[1])
        part = np.partition(Z, -kk, axis=1)[:, -kk:]
        return np.nanmean(part, axis=1)
    return topk_mean(Ztr), topk_mean(Zte)


def build_feature_sets(df: pd.DataFrame, old_cols: List[str], score_cols: List[str]) -> Dict[str, List[str]]:
    v59_cols = [c for c in df.columns if c.startswith(("clip__", "slog1p__", "miss__", "extreme__", "safe_raw__"))]
    d_cols = [c for c in v59_cols if ("v59_d_llm" in c or "v59_dout" in c)]
    s4_cols = [c for c in v59_cols if "v59_s4" in c]
    scores = [c for c in score_cols if c.startswith("score__") or c.startswith("alarm__")]
    fs = {
        "selected_score_features": scores,
        "old_exact": old_cols,
        "v59_all_sanitized": v59_cols,
        "v59_d_llm_dout_plus_scores": sorted(set(d_cols + scores)),
        "v59_s4_plus_scores": sorted(set(s4_cols + scores)),
        "old_exact_plus_scores": sorted(set(old_cols + scores)),
        "v59_all_plus_scores": sorted(set(v59_cols + scores)),
        "old_exact_plus_v59_all_plus_scores": sorted(set(old_cols + v59_cols + scores)),
    }
    return {k: [c for c in v if c in df.columns] for k, v in fs.items() if v}


def run_model_oof(df: pd.DataFrame, labels: List[str], fs: Dict[str, List[str]], models: List[str], targets: List[float], kbest: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seeds = sorted(pd.to_numeric(df["seed"], errors="coerce").dropna().astype(int).unique().tolist())
    pred_cols = []
    score_cols = []
    rows = []
    out = df[["seed", "sample_id", "exact_layer", "scope_group"] + [c for c in labels if c in df.columns]].copy()
    if len(seeds) < 2:
        return pd.DataFrame(rows), out
    for label in labels:
        if label not in df.columns:
            continue
        y_all = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int).to_numpy()
        for fs_name, cols in fs.items():
            if len(cols) == 0:
                continue
            X_all = df[cols]
            k = min(kbest, len(cols))
            for model_name in models:
                oof_score = np.full(len(df), np.nan)
                thr_by_target = {rt: [] for rt in targets}
                fold_count = 0
                for seed in seeds:
                    test_mask = df["seed"].astype(int).to_numpy() == seed
                    train_mask = ~test_mask
                    y_train = y_all[train_mask]
                    if test_mask.sum() == 0 or train_mask.sum() < 20 or len(set(y_train.tolist())) < 2:
                        continue
                    model = build_pipeline(model_name, k=k, seed=int(seed) % 100000)
                    model.fit(X_all.loc[train_mask, :], y_train)
                    tr_score = predict_score(model, X_all.loc[train_mask, :])
                    te_score = predict_score(model, X_all.loc[test_mask, :])
                    oof_score[test_mask] = te_score
                    for rt in targets:
                        thr_by_target[rt].append(choose_threshold_for_recall(y_train, tr_score, rt))
                    fold_count += 1
                if not np.isfinite(oof_score).any():
                    continue
                score_col = f"score__model__{label}__{fs_name}__{model_name}"
                out[score_col] = oof_score
                score_cols.append(score_col)
                for rt in targets:
                    if not thr_by_target[rt]:
                        continue
                    thr = float(np.median(thr_by_target[rt]))
                    pred = (oof_score >= thr).astype(int)
                    pred_col = f"alarm__model__{label}__{fs_name}__{model_name}__rt{str(rt).replace('.', 'p')}"
                    out[pred_col] = pred
                    pred_cols.append(pred_col)
                    meta = {"candidate_type": "trained_model", "candidate_name": pred_col, "feature_set": fs_name, "model": model_name, "target_train_recall": rt, "threshold": thr, "feature_count": len(cols), "selected_k": k, "fold_count": fold_count}
                    add_scope_results(rows, df, pred, label, meta)
    return pd.DataFrame(rows), out


def build_consistency_alarms(df: pd.DataFrame, seeds: List[int], quantiles: List[float], diag_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    all_cols = [c for c in df.columns if is_feature_col(str(c))]
    v59_cols = [c for c in all_cols if "v59_" in c]
    d_cols = [c for c in v59_cols if ("v59_d_llm" in c or "v59_dout" in c)]
    dout_cols = [c for c in v59_cols if "v59_dout" in c]
    s4_cols = [c for c in v59_cols if "v59_s4" in c]
    s4_track_cols = [c for c in s4_cols if "track" in c]
    s4_seg_cols = [c for c in s4_cols if "seg" in c]
    groups = {
        "v59_d_llm_dout_group_anomaly": d_cols,
        "v59_dout_group_anomaly": dout_cols,
        "v59_s4_group_anomaly": s4_cols,
        "v59_s4_track_group_anomaly": s4_track_cols,
        "v59_s4_seg_group_anomaly": s4_seg_cols,
        "output_ego_diag_group_anomaly": diag_cols,
    }
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    out = df[["seed", "sample_id", "exact_layer", "scope_group"]].copy()
    thresholds = []
    for name, cols in groups.items():
        out[f"score__consistency__{name}"] = np.nan
        for q in quantiles:
            out[f"alarm__consistency__{name}__q{q}"] = 0
        for seed in seeds:
            train = df[df["seed"] != seed]
            test = df[df["seed"] == seed]
            if train.empty or test.empty:
                continue
            tr_s, te_s = robust_group_scores(train, test, cols)
            out.loc[test.index, f"score__consistency__{name}"] = te_s
            for q in quantiles:
                thr = float(np.nanquantile(tr_s, q)) if len(tr_s) else np.inf
                out.loc[test.index, f"alarm__consistency__{name}__q{q}"] = (te_s >= thr).astype(int)
                thresholds.append({"signal": name, "seed_test": seed, "quantile": q, "threshold": thr, "feature_count": len(cols)})
    return out, pd.DataFrame(thresholds), {k: len(v) for k, v in groups.items()}


def add_disagreement_veto(df: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    hgb_score_cols = [f"score__{d}" for d in ["lineA_old_hgb_online_all_no_temporal", "lineA_speed_alter_hgb_online_compact"] if f"score__{d}" in df.columns]
    out = df[["seed", "sample_id"]].copy()
    if not hgb_score_cols:
        return out
    hgb_max = df[hgb_score_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).max(axis=1).to_numpy()
    base_alarm = df[f"alarm__{BASE_DETECTOR}"].astype(int).to_numpy()
    out["score__veto_hgbmax"] = hgb_max
    for t in thresholds:
        out[f"veto__hgbmax_le_{t}"] = ((base_alarm == 1) & (hgb_max <= t)).astype(int)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-dir", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--labels", default=",".join(LABELS))
    ap.add_argument("--models", default="extra_trees_depth5,hist_gradient_boosting")
    ap.add_argument("--feature-sets", default="selected_score_features,old_exact,v59_all_sanitized,v59_d_llm_dout_plus_scores,v59_s4_plus_scores,old_exact_plus_scores,v59_all_plus_scores,old_exact_plus_v59_all_plus_scores")
    ap.add_argument("--recall-targets", default="0.98,0.99,1.0")
    ap.add_argument("--quantiles", default="0.975,0.99,0.995")
    ap.add_argument("--veto-thresholds", default="0.005,0.01,0.02,0.05")
    ap.add_argument("--kbest", type=int, default=256)
    args = ap.parse_args()

    formal_dir = Path(args.formal_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    labels = [x for x in args.labels.split(",") if x]
    models = [x for x in args.models.split(",") if x]
    requested_fs = [x for x in args.feature_sets.split(",") if x]
    recall_targets = [float(x) for x in args.recall_targets.split(",") if x]
    quantiles = [float(x) for x in args.quantiles.split(",") if x]
    veto_thresholds = [float(x) for x in args.veto_thresholds.split(",") if x]

    df = load_v59_sanitized(formal_dir, seeds)
    df = merge_label_table(formal_dir, df, seeds)
    missing_after_label_merge = [c for c in LABELS if c not in df.columns]
    if missing_after_label_merge:
        raise KeyError(
            "missing required label columns after all fallbacks: "
            + ",".join(missing_after_label_merge)
            + ". Re-run v54 export/check for this quickcheck directory or include v54_2_posthoc_label_table_multi_policy.csv."
        )
    df, old_cols, old_src = load_old_exact_features(formal_dir, seeds, df)
    df, diag_cols, diag_src = load_diagnostics_subset(formal_dir, seeds, df)
    df, score_cols, score_src = merge_scores(formal_dir, df, seeds)
    df = ensure_base_alarm(df)
    labels = [l for l in labels if l in df.columns]

    fs_all = build_feature_sets(df, old_cols, score_cols)
    fs = {k: v for k, v in fs_all.items() if k in requested_fs and len(v) > 0}

    model_results, model_oof = run_model_oof(df, labels, fs, models, recall_targets, args.kbest)
    cons_oof, cons_thresholds, group_counts = build_consistency_alarms(df, seeds, quantiles, diag_cols)
    veto_oof = add_disagreement_veto(df, veto_thresholds)

    keys = ["seed", "sample_id"]
    pred_df = df[keys + ["exact_layer", "scope_group"] + labels + [f"alarm__{BASE_DETECTOR}"]].copy()
    if not model_oof.empty:
        # model_oof may contain label columns (harmful/severe/...) copied from the
        # training dataframe.  If we merge them directly, pandas will suffix the
        # already-present label columns in pred_df to harmful_x/harmful_y, and later
        # evaluation will fail with KeyError("harmful").  Keep only predictions and
        # metadata that are not labels.
        drop_cols = [c for c in (["exact_layer", "scope_group"] + sorted(ALL_LABELS) + [f"alarm__{BASE_DETECTOR}"]) if c in model_oof.columns]
        pred_df = pred_df.merge(model_oof.drop(columns=drop_cols, errors="ignore"), on=keys, how="left")
    pred_df = pred_df.merge(cons_oof.drop(columns=["exact_layer", "scope_group"], errors="ignore"), on=keys, how="left")
    pred_df = pred_df.merge(veto_oof, on=keys, how="left")

    result_rows: List[Dict] = []
    base_col = f"alarm__{BASE_DETECTOR}"
    base_pred = pred_df[base_col].astype(int).to_numpy()
    # Base rows.
    for lab in labels:
        add_scope_results(result_rows, pred_df, base_pred, lab, {"candidate_type": "baseline", "candidate_name": "base_extratrees", "combination": "base_only"})

    # Candidate model alarms include the raw trained model alarms and base OR trained model alarms.
    model_alarm_cols = [c for c in pred_df.columns if c.startswith("alarm__model__")]
    cons_alarm_cols = [c for c in pred_df.columns if c.startswith("alarm__consistency__")]
    veto_cols = [c for c in pred_df.columns if c.startswith("veto__hgbmax_le_")]

    for lab in labels:
        y = pred_df[lab].astype(int).to_numpy()
        for mcol in model_alarm_cols:
            # Only use model alarm for its own label.
            if f"__{lab}__" not in mcol:
                continue
            model_pred = pred_df[mcol].fillna(0).astype(int).to_numpy()
            base_or_model = np.maximum(base_pred, model_pred)
            meta_base = {"candidate_type": "feature_model", "candidate_name": mcol, "combination": "model_only"}
            add_scope_results(result_rows, pred_df, model_pred, lab, meta_base)
            meta_or = {"candidate_type": "best_feature_base", "candidate_name": mcol, "combination": "base_or_model"}
            add_scope_results(result_rows, pred_df, base_or_model, lab, meta_or, base_pred=base_pred)

            # Add consistency OR on top of feature model / base OR model.
            for ccol in cons_alarm_cols:
                cons_pred = pred_df[ccol].fillna(0).astype(int).to_numpy()
                model_or_cons = np.maximum(model_pred, cons_pred)
                base_model_cons = np.maximum(base_or_model, cons_pred)
                add_scope_results(result_rows, pred_df, model_or_cons, lab,
                                  {"candidate_type": "feature_model_plus_consistency", "candidate_name": mcol, "consistency_signal": ccol, "combination": "model_or_consistency"}, base_pred=model_pred)
                add_scope_results(result_rows, pred_df, base_model_cons, lab,
                                  {"candidate_type": "best_feature_base_plus_consistency", "candidate_name": mcol, "consistency_signal": ccol, "combination": "base_or_model_or_consistency"}, base_pred=base_or_model)
                for vcol in veto_cols:
                    veto = pred_df[vcol].fillna(0).astype(int).to_numpy()
                    final_pred = ((base_model_cons == 1) & (veto == 0)).astype(int)
                    add_scope_results(result_rows, pred_df, final_pred, lab,
                                      {"candidate_type": "best_feature_base_plus_consistency_and_veto", "candidate_name": mcol, "consistency_signal": ccol, "veto_signal": vcol, "combination": "base_or_model_or_consistency_and_not_veto"}, base_pred=base_or_model)
            # Veto only, without adding consistency.
            for vcol in veto_cols:
                veto = pred_df[vcol].fillna(0).astype(int).to_numpy()
                final_pred = ((base_or_model == 1) & (veto == 0)).astype(int)
                add_scope_results(result_rows, pred_df, final_pred, lab,
                                  {"candidate_type": "best_feature_base_and_veto", "candidate_name": mcol, "veto_signal": vcol, "combination": "base_or_model_and_not_veto"}, base_pred=base_or_model)

        # Consistency on raw ExtraTrees base too, as a control.
        for ccol in cons_alarm_cols:
            cons = pred_df[ccol].fillna(0).astype(int).to_numpy()
            base_cons = np.maximum(base_pred, cons)
            add_scope_results(result_rows, pred_df, base_cons, lab,
                              {"candidate_type": "base_plus_consistency_control", "candidate_name": "base_extratrees", "consistency_signal": ccol, "combination": "base_or_consistency"}, base_pred=base_pred)
            for vcol in veto_cols:
                veto = pred_df[vcol].fillna(0).astype(int).to_numpy()
                final_pred = ((base_cons == 1) & (veto == 0)).astype(int)
                add_scope_results(result_rows, pred_df, final_pred, lab,
                                  {"candidate_type": "base_plus_consistency_and_veto_control", "candidate_name": "base_extratrees", "consistency_signal": ccol, "veto_signal": vcol, "combination": "base_or_consistency_and_not_veto"}, base_pred=base_pred)

    results = pd.DataFrame(result_rows)
    # Sort best ALL by label: recall desc, FAR asc, precision desc.
    if not results.empty:
        all_rows = results[results["scope"].astype(str).eq("ALL")].copy()
        all_rows = all_rows.sort_values(["label", "recall", "far", "precision"], ascending=[True, False, True, False])
        best = all_rows.groupby("label", group_keys=False).head(60)
    else:
        best = pd.DataFrame()

    # Incremental candidates: rows where consistency is on top of a best_feature_base.
    inc = results[results["candidate_type"].astype(str).str.contains("best_feature_base", na=False)].copy() if not results.empty else pd.DataFrame()
    if not inc.empty:
        inc_all = inc[inc["scope"].astype(str).eq("ALL")].sort_values(["label", "recovered_fn_vs_base", "filtered_fp_vs_base", "recall", "far"], ascending=[True, False, False, False, True])
    else:
        inc_all = pd.DataFrame()

    model_results.to_csv(out_dir / "v61_1_feature_model_oof_results_by_scope.csv", index=False)
    cons_thresholds.to_csv(out_dir / "v61_1_consistency_thresholds.csv", index=False)
    pd.DataFrame([{"signal": k, "feature_count": v} for k, v in group_counts.items()]).to_csv(out_dir / "v61_1_consistency_feature_group_summary.csv", index=False)
    results.to_csv(out_dir / "v61_1_best_feature_consistency_results_by_scope.csv", index=False)
    best.to_csv(out_dir / "v61_1_best_candidates_ALL.csv", index=False)
    inc_all.to_csv(out_dir / "v61_1_incremental_improvement_ALL.csv", index=False)
    # Save compact per-row signals for debugging (not all feature columns).
    keep_cols = keys + ["exact_layer", "scope_group"] + labels + [base_col]
    keep_cols += [c for c in pred_df.columns if c.startswith("alarm__model__") or c.startswith("alarm__consistency__") or c.startswith("veto__") or c.startswith("score__consistency__") or c == "score__veto_hgbmax"]
    pred_df[keep_cols].to_csv(out_dir / "v61_1_oof_signal_table.csv", index=False)

    summary = {
        "version": "v61_1_best_feature_consistency_probe",
        "formal_dir": str(formal_dir),
        "seeds": seeds,
        "rows": int(len(df)),
        "labels": labels,
        "old_exact_source": old_src,
        "score_source": score_src,
        "diagnostics_source": diag_src,
        "feature_sets": {k: len(v) for k, v in fs.items()},
        "consistency_groups": group_counts,
        "models": models,
        "recall_targets": recall_targets,
        "quantiles": quantiles,
        "veto_thresholds": veto_thresholds,
        "note": "2-seed quickcheck is diagnostic only. Use more seeds before making formal claims.",
        "outputs": {
            "results_by_scope": str(out_dir / "v61_1_best_feature_consistency_results_by_scope.csv"),
            "best_all": str(out_dir / "v61_1_best_candidates_ALL.csv"),
            "incremental_all": str(out_dir / "v61_1_incremental_improvement_ALL.csv"),
        }
    }
    (out_dir / "v61_1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
