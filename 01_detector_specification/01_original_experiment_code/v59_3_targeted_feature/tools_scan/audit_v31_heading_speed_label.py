#!/usr/bin/env python3
"""v31 heading/speed label audit for OpenDriveVLA S0/CanBus experiments.

OFFLINE ONLY:
- This script may use paired clean predictions and GT trajectories to construct
  additional harmful labels.
- Generated beh__*, v31_* label columns are audit/evaluation columns.
- Detector training must exclude GT/clean/degradation/label columns from online inputs.

v31 label modes:
  old:
    v31_old_harmful = main_harmful

  heading:
    v31_heading_expanded_harmful = old OR heading_harmful

  speed_alter:
    v31_speed_alter_harmful = old OR speed_harmful

  heading_speed_alter:
    v31_heading_speed_alter_harmful = old OR heading_harmful OR speed_harmful

Default heading_harmful is based on fault-vs-clean mean heading gap.
A stricter heading flag requiring GT degradation is also exported for analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def clean_token(x: Any) -> str:
    s = str(x or "")
    return s[:-11] if s.endswith("_trajectory") else s


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    out.update(
        {
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
            f"{prefix}curvature_abs_delta": (
                float(abs(curvature_proxy(a) - curvature_proxy(b)))
                if math.isfinite(curvature_proxy(a)) and math.isfinite(curvature_proxy(b))
                else math.nan
            ),
        }
    )
    return out


def intish(x: Any) -> int:
    try:
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


def flag_ge(x: Any, thr: float) -> bool:
    try:
        v = float(x)
        return math.isfinite(v) and v >= thr
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--seed-subdir", required=True)
    ap.add_argument("--matrix-csv", required=True)
    ap.add_argument("--output-dir", required=True)

    # Main v31 label thresholds.
    ap.add_argument("--heading-clean-threshold", type=float, default=0.35)
    ap.add_argument("--heading-gt-degrade-threshold", type=float, default=0.20)

    # Speed is an auxiliary/alter line. It is intentionally stricter by default:
    # both fault-vs-clean speed change and GT-relative degradation are required.
    ap.add_argument("--speed-clean-threshold", type=float, default=0.40)
    ap.add_argument("--speed-gt-degrade-threshold", type=float, default=0.20)

    # Retained only as audit columns; these do NOT define v31 expanded labels.
    ap.add_argument("--shape-clean-threshold", type=float, default=0.75)
    ap.add_argument("--final-clean-threshold", type=float, default=1.0)
    ap.add_argument("--shape-gt-degrade-threshold", type=float, default=0.50)
    ap.add_argument("--final-gt-degrade-threshold", type=float, default=0.75)
    args = ap.parse_args()

    campaign = Path(args.campaign_dir)
    seed_dir = campaign / args.seed_subdir
    cal_dir = campaign / "calibration"
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(Path(args.matrix_csv))
    pred_fault = load_json(seed_dir / "pred_trajs_dict.json")
    pred_clean = load_json(cal_dir / "pred_trajs_dict.json")
    gt_dict, gt_mask_dict, gt_source = load_gt()

    audit_rows: List[Dict[str, Any]] = []
    counters = {
        "old_harmful": 0,
        "heading_harmful": 0,
        "heading_expanded_harmful": 0,
        "speed_harmful": 0,
        "speed_alter_harmful": 0,
        "heading_speed_alter_harmful": 0,
        "old_negative_heading_harmful": 0,
        "old_negative_speed_harmful": 0,
    }

    for r in rows:
        token = clean_token(r.get("sample_id", ""))
        if not token:
            continue
        source = r.get("row_source", "")
        is_cal_clean = source == "calibration_clean" or r.get("is_clean") == "1"

        # For calibration rows, "fault" trajectory is the clean trajectory itself.
        fault_pred = pred_clean if is_cal_clean else pred_fault
        ft = to_traj2(pred_lookup(fault_pred, token))
        ct = to_traj2(pred_lookup(pred_clean, token))
        gt = gt_to_traj2(gt_dict.get(token), gt_mask_dict.get(token)) if token in gt_dict else None

        out: Dict[str, Any] = {
            "sample_id": token,
            "row_source": source,
            "main_harmful": r.get("main_harmful", ""),
            "strict_harmful": r.get("strict_harmful", ""),
            "mild_harmful": r.get("mild_harmful", ""),
            "injected_effective": r.get("injected_effective", r.get("is_effective_injected", "")),
            "fault_group": r.get("fault_group", ""),
            "fault_target": r.get("fault_target", ""),
            "fault_bit_op": r.get("fault_bit_op", ""),
            "fault_bit_count": r.get("fault_bit_count", ""),
            "delta_l2_avg": r.get("delta_l2_avg", ""),
            "delta_collision_avg": r.get("delta_collision_avg", ""),
            "beh__fault_traj_available": 1 if ft is not None else 0,
            "beh__clean_traj_available": 1 if ct is not None else 0,
            "beh__gt_traj_available": 1 if gt is not None else 0,
        }

        for name, tr in [("fault", ft), ("clean", ct), ("gt", gt)]:
            if tr is not None:
                out[f"beh__{name}_path_length"] = path_length(tr)
                out[f"beh__{name}_final_disp"] = float(np.linalg.norm(tr[-1]))
                sn = step_norms(tr)
                out[f"beh__{name}_step_mean"] = float(sn.mean()) if sn.size else math.nan
                out[f"beh__{name}_step_max"] = float(sn.max()) if sn.size else math.nan
                out[f"beh__{name}_curvature_proxy"] = curvature_proxy(tr)

        out.update(pair_metrics(ft, ct, "beh__fault_vs_clean__"))
        out.update(pair_metrics(ft, gt, "beh__fault_vs_gt__"))
        out.update(pair_metrics(ct, gt, "beh__clean_vs_gt__"))

        for metric in [
            "mean_point_dist",
            "max_point_dist",
            "final_point_dist",
            "mean_heading_abs_delta",
            "max_heading_abs_delta",
            "final_heading_abs_delta",
            "mean_step_norm_abs_delta",
            "path_length_abs_delta",
            "curvature_abs_delta",
        ]:
            a = out.get(f"beh__fault_vs_gt__{metric}", math.nan)
            b = out.get(f"beh__clean_vs_gt__{metric}", math.nan)
            try:
                out[f"beh__degradation_vs_clean_gt__{metric}"] = float(a) - float(b)
            except Exception:
                out[f"beh__degradation_vs_clean_gt__{metric}"] = math.nan

        heading_clean_flag = flag_ge(out.get("beh__fault_vs_clean__mean_heading_abs_delta"), args.heading_clean_threshold)
        heading_gt_degrade_flag = flag_ge(
            out.get("beh__degradation_vs_clean_gt__mean_heading_abs_delta"),
            args.heading_gt_degrade_threshold,
        )
        heading_strict_flag = bool(heading_clean_flag and heading_gt_degrade_flag)

        # v31 main heading label uses the average fault-vs-clean heading gap.
        # The strict GT-degradation-gated version is exported for later analysis.
        heading_harmful = int(heading_clean_flag)

        speed_clean_flag = flag_ge(out.get("beh__fault_vs_clean__mean_step_norm_abs_delta"), args.speed_clean_threshold)
        speed_gt_degrade_flag = flag_ge(
            out.get("beh__degradation_vs_clean_gt__mean_step_norm_abs_delta"),
            args.speed_gt_degrade_threshold,
        )
        speed_harmful = int(bool(speed_clean_flag and speed_gt_degrade_flag))

        # Audit-only old broad behavior flags from v30 for decomposition only.
        shape_clean_flag = flag_ge(out.get("beh__fault_vs_clean__mean_point_dist"), args.shape_clean_threshold)
        final_clean_flag = flag_ge(out.get("beh__fault_vs_clean__final_point_dist"), args.final_clean_threshold)
        shape_gt_degrade_flag = flag_ge(
            out.get("beh__degradation_vs_clean_gt__mean_point_dist"),
            args.shape_gt_degrade_threshold,
        )
        final_gt_degrade_flag = flag_ge(
            out.get("beh__degradation_vs_clean_gt__final_point_dist"),
            args.final_gt_degrade_threshold,
        )

        old_harmful = intish(r.get("main_harmful", 0))
        heading_expanded = int(bool(old_harmful or heading_harmful))
        speed_alter = int(bool(old_harmful or speed_harmful))
        heading_speed_alter = int(bool(old_harmful or heading_harmful or speed_harmful))

        out.update(
            {
                "v31_old_harmful": old_harmful,
                "v31_heading_harmful": heading_harmful,
                "v31_heading_strict_harmful": int(heading_strict_flag),
                "v31_heading_expanded_harmful": heading_expanded,
                "v31_speed_harmful": speed_harmful,
                "v31_speed_alter_harmful": speed_alter,
                "v31_heading_speed_alter_harmful": heading_speed_alter,
                "beh__flag_fault_vs_clean_heading": int(heading_clean_flag),
                "beh__flag_gt_heading_degradation": int(heading_gt_degrade_flag),
                "beh__flag_heading_strict": int(heading_strict_flag),
                "beh__flag_fault_vs_clean_speed": int(speed_clean_flag),
                "beh__flag_gt_speed_degradation": int(speed_gt_degrade_flag),
                "beh__flag_speed_alter": int(speed_harmful),
                "beh__flag_fault_vs_clean_shape_audit_only": int(shape_clean_flag),
                "beh__flag_fault_vs_clean_final_audit_only": int(final_clean_flag),
                "beh__flag_gt_shape_degradation_audit_only": int(shape_gt_degrade_flag),
                "beh__flag_gt_final_degradation_audit_only": int(final_gt_degrade_flag),
            }
        )

        counters["old_harmful"] += old_harmful
        counters["heading_harmful"] += heading_harmful
        counters["heading_expanded_harmful"] += heading_expanded
        counters["speed_harmful"] += speed_harmful
        counters["speed_alter_harmful"] += speed_alter
        counters["heading_speed_alter_harmful"] += heading_speed_alter
        if old_harmful == 0 and heading_harmful == 1:
            counters["old_negative_heading_harmful"] += 1
        if old_harmful == 0 and speed_harmful == 1:
            counters["old_negative_speed_harmful"] += 1

        audit_rows.append(out)

    write_csv(outdir / "v31_heading_speed_label_audit.csv", audit_rows)
    summary = {
        "version": "v31_heading_speed_label_audit",
        "campaign_dir": str(campaign),
        "seed_subdir": args.seed_subdir,
        "matrix_csv": str(args.matrix_csv),
        "gt_source": gt_source,
        "rows": len(audit_rows),
        "counts": counters,
        "thresholds": vars(args),
        "label_modes": {
            "old": "v31_old_harmful = main_harmful",
            "heading": "v31_heading_expanded_harmful = old OR fault_vs_clean_mean_heading_abs_delta >= heading_clean_threshold",
            "speed_alter": "v31_speed_alter_harmful = old OR (speed_clean_flag AND speed_gt_degrade_flag)",
            "heading_speed_alter": "old OR heading OR speed_alter",
        },
        "note": "GT/clean-based behavior columns are audit-only and must not be used as online detector inputs.",
    }
    (outdir / "v31_heading_speed_label_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
