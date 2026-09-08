#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build v54.2 posthoc multi-policy labels compatible with v31/v35/v38.

This tool is OFFLINE/POSTHOC ONLY. It may use current fault predictions,
clean calibration predictions, and GT trajectories to construct evaluation labels.
Generated v31_* labels and beh__* metrics must never be used as online detector
features.

Main v31-compatible label modes:
  old:                 v31_old_harmful_compatible = harmful
  heading:             old OR fault-vs-clean mean heading gap >= 0.35
  speed_alter:         old OR (fault-vs-clean speed gap >= 0.40 AND GT-relative speed degradation >= 0.20)
  heading_speed_alter: old OR heading OR speed_alter
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd

MULTIPOLICY_METRIC_LABELS = [
    "harmful",
    "severe_harmful",
    "v31_old_harmful_compatible",
    "v31_heading_expanded_harmful_compatible",
    "v31_speed_alter_harmful_compatible",
    "v31_heading_speed_alter_harmful_compatible",
]


def clean_token(x: Any) -> str:
    s = str(x or "")
    return s[:-11] if s.endswith("_trajectory") else s


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def pred_lookup(pred: Dict[str, Any], token: str) -> Optional[Any]:
    token = clean_token(token)
    for k in (token + "_trajectory", token):
        if k in pred:
            return pred[k]
    return None


def to_traj2(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            return None
        arr = arr.reshape(-1, 2)
        if arr.shape[0] > 6:
            arr = arr[:6]
        if not np.isfinite(arr).all() or arr.shape[0] < 2:
            return None
        return arr
    except Exception:
        return None


def gt_to_traj2(x: Any, mask: Any = None) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            return None
        arr = arr.reshape(-1, 2)
        if arr.shape[0] > 6:
            arr = arr[:6]
        if mask is not None:
            m = np.asarray(mask).reshape(-1)
            if m.size >= arr.shape[0]:
                m = m[:arr.shape[0]].astype(bool)
                if m.any():
                    arr = arr[m]
        if not np.isfinite(arr).all() or arr.shape[0] < 2:
            return None
        return arr
    except Exception:
        return None


def angle_wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def headings(traj: np.ndarray) -> np.ndarray:
    d = np.diff(traj, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


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
    h = headings(traj)
    if len(h) < 2:
        return math.nan
    return float(np.abs(angle_wrap(np.diff(h))).mean())


def pair_metrics(a: Optional[np.ndarray], b: Optional[np.ndarray], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if a is None or b is None:
        out[f"{prefix}available"] = 0.0
        return out
    n = min(len(a), len(b))
    if n < 2:
        out[f"{prefix}available"] = 0.0
        return out
    a = a[:n]
    b = b[:n]
    dist = np.linalg.norm(a - b, axis=1)
    ha = headings(a)
    hb = headings(b)
    m = min(len(ha), len(hb))
    hd = np.abs(angle_wrap(ha[:m] - hb[:m])) if m else np.asarray([], dtype=float)
    cos = np.cos(angle_wrap(ha[:m] - hb[:m])) if m else np.asarray([], dtype=float)
    sa = step_norms(a)
    sb = step_norms(b)
    sm = min(len(sa), len(sb))
    curv_a = curvature_proxy(a)
    curv_b = curvature_proxy(b)
    out.update({
        f"{prefix}available": 1.0,
        f"{prefix}mean_point_dist": float(dist.mean()) if dist.size else math.nan,
        f"{prefix}max_point_dist": float(dist.max()) if dist.size else math.nan,
        f"{prefix}final_point_dist": float(dist[-1]) if dist.size else math.nan,
        f"{prefix}mean_heading_abs_delta": float(hd.mean()) if hd.size else math.nan,
        f"{prefix}max_heading_abs_delta": float(hd.max()) if hd.size else math.nan,
        f"{prefix}final_heading_abs_delta": float(hd[-1]) if hd.size else math.nan,
        f"{prefix}heading_cos_mean": float(cos.mean()) if cos.size else math.nan,
        f"{prefix}opposite_step_rate": float((cos < 0.0).mean()) if cos.size else math.nan,
        f"{prefix}mean_step_norm_abs_delta": float(np.abs(sa[:sm] - sb[:sm]).mean()) if sm else math.nan,
        f"{prefix}path_length_abs_delta": float(abs(path_length(a) - path_length(b))),
        f"{prefix}curvature_abs_delta": float(abs(curv_a - curv_b)) if math.isfinite(curv_a) and math.isfinite(curv_b) else math.nan,
    })
    return out


def flag_ge(x: Any, thr: float) -> bool:
    try:
        v = float(x)
        return math.isfinite(v) and v >= thr
    except Exception:
        return False


def intish(x: Any) -> int:
    try:
        if x is None or x == "":
            return 0
        return int(float(x))
    except Exception:
        return 0


def load_gt() -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    try:
        import eval_share.evaluation as evaluation_mod  # type: ignore
        current_dir = Path(evaluation_mod.__file__).resolve().parent
        with open(current_dir / "gt/gt_traj.pkl", "rb") as f:
            gt = pickle.load(f)
        with open(current_dir / "gt/gt_traj_mask.pkl", "rb") as f:
            mask = pickle.load(f)
        return gt, mask, str(current_dir)
    except Exception as e:
        return {}, {}, f"unavailable:{type(e).__name__}:{e}"


def parse_answer_to_traj(answer: Any) -> Optional[List[List[float]]]:
    """Parse common OpenDriveVLA answer formats into 6 xy waypoints."""
    try:
        if isinstance(answer, list) and answer:
            answer = answer[0]
        if isinstance(answer, str):
            s = answer.strip()
            # Often answer is a string containing a Python list of tuples.
            try:
                obj = ast.literal_eval(s)
                arr = np.asarray(obj, dtype=float).reshape(-1, 2)
                if arr.shape[0] >= 2:
                    return arr[:6].tolist()
            except Exception:
                pass
            # Fallback: extract all numbers and group xy.
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)]
            if len(nums) >= 4:
                arr = np.asarray(nums[:12], dtype=float).reshape(-1, 2)
                return arr[:6].tolist()
        # Some callers may pass a nested list directly.
        arr = np.asarray(answer, dtype=float).reshape(-1, 2)
        if arr.shape[0] >= 2:
            return arr[:6].tolist()
    except Exception:
        return None
    return None


def load_fault_preds_from_plan_conv(seed_dir: Path) -> Dict[str, Any]:
    path = seed_dir / "plan_conv.json"
    pred: Dict[str, Any] = {}
    if not path.is_file():
        return pred
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            sid_raw = str(row.get("id", ""))
            sid = clean_token(sid_raw)
            traj = parse_answer_to_traj(row.get("answer", ""))
            if sid and traj is not None:
                pred[sid] = traj
                pred[sid + "_trajectory"] = traj
    return pred


def load_fault_preds_robust(seed_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load fault predictions using the v31-compatible source order.

    Original v31 uses seed_dir/pred_trajs_dict.json. Some v54 smoke packs do not
    write or pack that file, so plan_conv parsing is retained as a fallback. This
    function does not invent trajectories. If an output cannot be parsed, the
    corresponding behavior label audit marks fault_traj_available=0 and keeps
    heading/speed flags false, matching the v31 audit behavior rather than
    treating it as an online detector failure.
    """
    pred: Dict[str, Any] = {}
    sources: Dict[str, Any] = {
        "loaded_sources": [],
        "source_counts": {},
    }
    candidates = [
        seed_dir / "pred_trajs_dict.json",
        seed_dir / "results" / "pred_trajs_dict.json",
        seed_dir / "results" / "pred_trajs_dict_rank0.json",
    ]
    for c in candidates:
        if c.is_file():
            try:
                d = load_json(c)
                if isinstance(d, dict):
                    pred.update(d)
                    sources["loaded_sources"].append(str(c))
                    sources["source_counts"][str(c)] = len(d)
            except Exception as e:
                sources.setdefault("load_errors", {})[str(c)] = f"{type(e).__name__}:{e}"
    plan_pred = load_fault_preds_from_plan_conv(seed_dir)
    if plan_pred:
        # Plan_conv is fallback only for keys not already present in pred_trajs_dict.
        added = 0
        for k, v in plan_pred.items():
            if k not in pred:
                pred[k] = v
                added += 1
        sources["loaded_sources"].append(str(seed_dir / "plan_conv.json"))
        sources["source_counts"][str(seed_dir / "plan_conv.json")] = len(plan_pred)
        sources["plan_conv_fallback_keys_added"] = added
    return pred, sources


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    yt = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    yp = pd.to_numeric(y_pred, errors="coerce").fillna(0).astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    return {
        "total": int(len(yt)),
        "positive_total": int((yt == 1).sum()),
        "negative_total": int((yt == 0).sum()),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": float(tp / (tp + fn)) if (tp + fn) else None,
        "far": float(fp / (fp + tn)) if (fp + tn) else None,
        "precision": float(tp / (tp + fp)) if (tp + fp) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--seed-root", default="A1_internal_neural_scdout_v54_2_online_singlefault")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--clean-pred-trajs", required=True)
    ap.add_argument("--heading-clean-threshold", type=float, default=0.35)
    ap.add_argument("--heading-gt-degrade-threshold", type=float, default=0.20)
    ap.add_argument("--speed-clean-threshold", type=float, default=0.40)
    ap.add_argument("--speed-gt-degrade-threshold", type=float, default=0.20)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    campaign = Path(args.campaign_dir)
    export_dir = Path(args.export_dir); export_dir.mkdir(parents=True, exist_ok=True)
    seeds = [s.strip().replace("seed", "") for s in args.seeds.split(",") if s.strip()]

    clean_pred_path = Path(args.clean_pred_trajs)
    pred_clean = load_json(clean_pred_path)
    gt_dict, gt_mask_dict, gt_source = load_gt()

    score_cache = export_dir / "v54_detector_score_cache.csv"
    base_label_table = export_dir / "v54_posthoc_label_table_harmful_severe.csv"
    if not score_cache.is_file():
        raise FileNotFoundError(score_cache)
    if not base_label_table.is_file():
        raise FileNotFoundError(base_label_table)
    scores_df = pd.read_csv(score_cache)

    audit_rows: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []
    counters = Counter()
    missing = Counter()
    fault_prediction_sources: Dict[str, Any] = {}

    for seed in seeds:
        seed_dir = campaign / args.seed_root / f"seed{seed}"
        rec_path = seed_dir / "results" / "v53_inline_detector_records.jsonl"
        if not rec_path.is_file():
            alt = seed_dir / "results" / "v54_inline_detector_records.jsonl"
            if alt.is_file():
                rec_path = alt
            else:
                raise FileNotFoundError(rec_path)
        pred_fault, fault_pred_source_info = load_fault_preds_robust(seed_dir)
        fault_prediction_sources[str(seed)] = fault_pred_source_info
        if not pred_fault:
            missing["missing_or_empty_fault_predictions"] += 1
        for r in read_jsonl(rec_path):
            token = clean_token(r.get("sample_id", ""))
            a = r.get("analysis_only", {}) or {}
            labels = a.get("posthoc_labels", {}) or {}
            deltas = a.get("delta_metrics", {}) or {}
            exact_layer = a.get("exact_layer", "")
            row_source = r.get("row_source", "")

            ft = to_traj2(pred_lookup(pred_fault, token))
            ct = to_traj2(pred_lookup(pred_clean, token))
            gt = gt_to_traj2(gt_dict.get(token), gt_mask_dict.get(token)) if token in gt_dict else None

            if ft is None:
                missing["fault_traj_missing"] += 1
            if ct is None:
                missing["clean_traj_missing"] += 1
            if gt is None:
                missing["gt_traj_missing"] += 1

            out: Dict[str, Any] = {
                "seed": seed,
                "sample_id": token,
                "raw_sample_id": r.get("raw_sample_id", ""),
                "row_source": row_source,
                "exact_layer": exact_layer,
                "fault_target_exact": a.get("fault_target_exact", ""),
                "injected_effective": labels.get("injected_effective", a.get("is_positive_effective", "")),
                "delta_l2_avg": labels.get("delta_l2_avg", deltas.get("delta_l2_avg", "")),
                "delta_collision_avg": labels.get("delta_collision_avg", deltas.get("delta_collision_avg", "")),
                "harmful": intish(labels.get("harmful", 0)),
                "severe_harmful": intish(labels.get("severe_harmful", 0)),
                "beh__fault_traj_available": 1 if ft is not None else 0,
                "beh__clean_traj_available": 1 if ct is not None else 0,
                "beh__gt_traj_available": 1 if gt is not None else 0,
            }

            for name, tr in [("fault", ft), ("clean", ct), ("gt", gt)]:
                if tr is not None:
                    sn = step_norms(tr)
                    out[f"beh__{name}_path_length"] = path_length(tr)
                    out[f"beh__{name}_final_disp"] = float(np.linalg.norm(tr[-1]))
                    out[f"beh__{name}_step_mean"] = float(sn.mean()) if sn.size else math.nan
                    out[f"beh__{name}_step_max"] = float(sn.max()) if sn.size else math.nan
                    out[f"beh__{name}_curvature_proxy"] = curvature_proxy(tr)

            out.update(pair_metrics(ft, ct, "beh__fault_vs_clean__"))
            out.update(pair_metrics(ft, gt, "beh__fault_vs_gt__"))
            out.update(pair_metrics(ct, gt, "beh__clean_vs_gt__"))

            for metric in [
                "mean_point_dist", "max_point_dist", "final_point_dist",
                "mean_heading_abs_delta", "max_heading_abs_delta", "final_heading_abs_delta",
                "mean_step_norm_abs_delta", "path_length_abs_delta", "curvature_abs_delta",
            ]:
                aa = out.get(f"beh__fault_vs_gt__{metric}", math.nan)
                bb = out.get(f"beh__clean_vs_gt__{metric}", math.nan)
                try:
                    out[f"beh__degradation_vs_clean_gt__{metric}"] = float(aa) - float(bb)
                except Exception:
                    out[f"beh__degradation_vs_clean_gt__{metric}"] = math.nan

            heading_clean_flag = flag_ge(out.get("beh__fault_vs_clean__mean_heading_abs_delta"), args.heading_clean_threshold)
            heading_gt_degrade_flag = flag_ge(out.get("beh__degradation_vs_clean_gt__mean_heading_abs_delta"), args.heading_gt_degrade_threshold)
            speed_clean_flag = flag_ge(out.get("beh__fault_vs_clean__mean_step_norm_abs_delta"), args.speed_clean_threshold)
            speed_gt_degrade_flag = flag_ge(out.get("beh__degradation_vs_clean_gt__mean_step_norm_abs_delta"), args.speed_gt_degrade_threshold)

            old_harmful = intish(out.get("harmful", 0))
            heading_harmful = int(heading_clean_flag)
            heading_strict = int(bool(heading_clean_flag and heading_gt_degrade_flag))
            speed_harmful = int(bool(speed_clean_flag and speed_gt_degrade_flag))
            heading_expanded = int(bool(old_harmful or heading_harmful))
            speed_alter = int(bool(old_harmful or speed_harmful))
            heading_speed_alter = int(bool(old_harmful or heading_harmful or speed_harmful))

            out.update({
                "v31_old_harmful_compatible": old_harmful,
                "v31_heading_harmful_compatible": heading_harmful,
                "v31_heading_strict_harmful_compatible": heading_strict,
                "v31_heading_expanded_harmful_compatible": heading_expanded,
                "v31_speed_harmful_compatible": speed_harmful,
                "v31_speed_alter_harmful_compatible": speed_alter,
                "v31_heading_speed_alter_harmful_compatible": heading_speed_alter,
                "beh__flag_fault_vs_clean_heading": int(heading_clean_flag),
                "beh__flag_gt_heading_degradation": int(heading_gt_degrade_flag),
                "beh__flag_heading_strict": heading_strict,
                "beh__flag_fault_vs_clean_speed": int(speed_clean_flag),
                "beh__flag_gt_speed_degradation": int(speed_gt_degrade_flag),
                "beh__flag_speed_alter": speed_harmful,
            })
            audit_rows.append(out)

            label_rows.append({k: out.get(k, "") for k in [
                "seed", "sample_id", "raw_sample_id", "row_source", "exact_layer", "fault_target_exact",
                "injected_effective", "delta_l2_avg", "delta_collision_avg",
                "harmful", "severe_harmful", "v31_old_harmful_compatible",
                "v31_heading_harmful_compatible", "v31_heading_strict_harmful_compatible",
                "v31_heading_expanded_harmful_compatible", "v31_speed_harmful_compatible",
                "v31_speed_alter_harmful_compatible", "v31_heading_speed_alter_harmful_compatible",
                "beh__fault_traj_available", "beh__clean_traj_available", "beh__gt_traj_available",
                "beh__flag_fault_vs_clean_heading", "beh__flag_gt_heading_degradation",
                "beh__flag_fault_vs_clean_speed", "beh__flag_gt_speed_degradation",
            ]})

            for name in [
                "harmful", "severe_harmful", "v31_old_harmful_compatible",
                "v31_heading_harmful_compatible", "v31_heading_expanded_harmful_compatible",
                "v31_speed_harmful_compatible", "v31_speed_alter_harmful_compatible",
                "v31_heading_speed_alter_harmful_compatible",
            ]:
                counters[name] += intish(out.get(name, 0))

    audit_df = pd.DataFrame(audit_rows)
    labels_df = pd.DataFrame(label_rows)
    audit_csv = export_dir / "v54_2_v31_behavior_label_audit.csv"
    label_csv = export_dir / "v54_2_posthoc_label_table_multi_policy.csv"
    audit_df.to_csv(audit_csv, index=False)
    labels_df.to_csv(label_csv, index=False)

    metric_rows: List[Dict[str, Any]] = []
    merge_cols = ["seed", "sample_id", "row_source", "exact_layer"]
    if not scores_df.empty and not labels_df.empty:
        scores_df["seed"] = scores_df["seed"].astype(str)
        labels_df["seed"] = labels_df["seed"].astype(str)
        flat = scores_df.merge(labels_df, on=merge_cols, how="left", suffixes=("", "_label"))
        for det, gdet in flat.groupby("detector"):
            for lab in MULTIPOLICY_METRIC_LABELS:
                if lab not in gdet.columns:
                    continue
                m = binary_metrics(gdet[lab], gdet["alarm"])
                m.update({"detector": det, "label": lab, "scope": "ALL"})
                metric_rows.append(m)
                for layer, gl in gdet.groupby("exact_layer", dropna=False):
                    mm = binary_metrics(gl[lab], gl["alarm"])
                    mm.update({"detector": det, "label": lab, "scope": str(layer)})
                    metric_rows.append(mm)
    metrics_csv = export_dir / "v54_2_posthoc_metrics_multi_policy.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_csv, index=False)

    rows = len(labels_df)
    availability = {
        "fault_traj_missing": int(missing.get("fault_traj_missing", 0)),
        "clean_traj_missing": int(missing.get("clean_traj_missing", 0)),
        "gt_traj_missing": int(missing.get("gt_traj_missing", 0)),
        "missing_or_empty_fault_predictions": int(missing.get("missing_or_empty_fault_predictions", 0)),
    }
    # v31-compatible behavior audit does not fail merely because a few fault
    # trajectories are unavailable/unparseable. In that case heading/speed flags
    # stay false for those rows and availability is reported for diagnostics.
    # Clean and GT trajectories are required because they are the reference sides
    # for all v31-compatible behavior comparisons.
    pass_flag = bool(
        rows > 0
        and availability.get("clean_traj_missing", 0) == 0
        and availability.get("gt_traj_missing", 0) == 0
        and availability.get("missing_or_empty_fault_predictions", 0) == 0
    )
    summary = {
        "version": "v54_3_v31_compatible_multipolicy_posthoc_labels",
        "campaign_dir": str(campaign),
        "seed_root": args.seed_root,
        "seeds": seeds,
        "rows": rows,
        "clean_pred_trajs": str(clean_pred_path),
        "gt_source": gt_source,
        "thresholds": {
            "heading_clean_threshold": args.heading_clean_threshold,
            "heading_gt_degrade_threshold": args.heading_gt_degrade_threshold,
            "speed_clean_threshold": args.speed_clean_threshold,
            "speed_gt_degrade_threshold": args.speed_gt_degrade_threshold,
        },
        "label_policy": "v31-compatible heading/speed labels are posthoc evaluation labels only; they are not deployable online features.",
        "label_modes": {
            "v31_old_harmful_compatible": "harmful = delta_l2_avg > 0.5 OR delta_collision_avg > 0",
            "v31_heading_expanded_harmful_compatible": "old OR fault-vs-clean mean heading abs delta >= 0.35",
            "v31_speed_alter_harmful_compatible": "old OR (fault-vs-clean mean step-norm abs delta >= 0.40 AND GT-relative mean step-norm degradation >= 0.20)",
            "v31_heading_speed_alter_harmful_compatible": "old OR heading_harmful OR speed_harmful",
        },
        "label_totals": {k: int(v) for k, v in counters.items()},
        "availability": availability,
        "availability_policy": "fault_traj_missing is diagnostic-only; rows with unavailable fault trajectory keep v31 heading/speed flags false, matching v31 audit behavior. clean_traj_missing/gt_traj_missing/missing_or_empty_fault_predictions are strict failures.",
        "fault_prediction_sources": fault_prediction_sources,
        "outputs": {
            "v31_behavior_label_audit_csv": str(audit_csv),
            "posthoc_label_table_multi_policy_csv": str(label_csv),
            "posthoc_metrics_multi_policy_csv": str(metrics_csv),
        },
        "pass": pass_flag,
    }
    summary_path = export_dir / "v54_2_multipolicy_label_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.strict and not pass_flag:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
