#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v54 preflight: exact no-guess checks before online detector smoke.

This script intentionally does not run inference. It verifies required paths,
selected v35.1 detector artifacts, v54 inference script invariants, and the
legacy calibration stats needed to construct exact v35.1 S0 feature columns.
"""
from __future__ import annotations
import argparse, json, math, tarfile, tempfile
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_DETECTOR_FEATURE_COUNTS = {
    "lineA_old_extra_trees_high_recall_online_all_no_temporal": 2597,
    "lineA_old_hgb_online_all_no_temporal": 2597,
    "lineA_speed_alter_hgb_online_compact": 707,
}
REQUIRED_DIM_STATS = [
    ["small_signed_structure", "per_index", "7", "log_abs_lower"],
    ["small_signed_structure", "per_index", "7", "log_abs_upper"],
    ["small_signed_structure", "per_index", "8", "log_abs_lower"],
    ["small_signed_structure", "per_index", "8", "log_abs_upper"],
    ["small_signed_structure", "per_index", "10", "log_abs_lower"],
    ["small_signed_structure", "per_index", "10", "log_abs_upper"],
    ["small_signed_structure", "per_index", "11", "log_abs_lower"],
    ["small_signed_structure", "per_index", "11", "log_abs_upper"],
    ["small_signed_structure", "per_index", "12", "log_abs_lower"],
    ["small_signed_structure", "per_index", "12", "log_abs_upper"],
    ["translation_shape", "xy_log_ratio_lower"],
    ["translation_shape", "xy_log_ratio_upper"],
    ["translation_shape", "xy_log_ratio_strong_lower"],
    ["translation_shape", "xy_log_ratio_strong_upper"],
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(root: Any, keys: List[str]) -> Any:
    cur = root
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def load_feature_counts(artifact: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="v54_artifact_") as td:
        with tarfile.open(artifact, "r:gz") as tf:
            tf.extractall(td)
        for p in Path(td).rglob("feature_columns.json"):
            try:
                cols = json.loads(p.read_text(encoding="utf-8"))
                out[p.parent.name] = len(cols) if isinstance(cols, list) else -1
            except Exception:
                out[p.parent.name] = -1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--package-dir", required=True)
    ap.add_argument("--detector-artifact", required=True)
    ap.add_argument("--subset-ids", required=True)
    ap.add_argument("--cal-json", required=True)
    ap.add_argument("--clean-ref", required=True)
    ap.add_argument("--output-sanity-json", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--registry-csv", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    pkg = Path(args.package_dir)
    inference_py = pkg / "drivevla" / "inference_drivevla_subset1200_v54_online_v35_1_scdout_record_audit.py"
    required_files = {
        "inference_py": inference_py,
        "detector_artifact": Path(args.detector_artifact),
        "subset_ids": Path(args.subset_ids),
        "cal_json": Path(args.cal_json),
        "clean_ref": Path(args.clean_ref),
        "output_sanity_json": Path(args.output_sanity_json),
        "registry_csv": Path(args.registry_csv),
    }
    required_dirs = {"repo_root": Path(args.repo_root), "model_path": Path(args.model_path)}
    missing = []
    for name, path in required_files.items():
        if not path.is_file():
            missing.append({"kind": "file", "name": name, "path": str(path)})
    for name, path in required_dirs.items():
        if not path.exists():
            missing.append({"kind": "path", "name": name, "path": str(path)})

    script_text = inference_py.read_text(encoding="utf-8") if inference_py.is_file() else ""
    script_checks = {
        "has_v54_target_mode": "SCDOUT_internal_online" in script_text,
        "target_mode_excludes_e_output": '"SCDOUT_internal_online": ["can_bus", "s1_activation", "c_activation", "d_activation"]' in script_text,
        "has_dout_model_norm_hook": "dout_model_norm=model.norm" in script_text,
        "has_dout_lm_head_hook": "dout_lm_head=lm_head" in script_text,
        "uses_exact_schema_reindex_without_fuzzy_matching": "exact reindex to feature_columns.json" in script_text and "absent_value_names" in script_text,
        "uses_deployable_feature_row": "v53_deployable_feature_row_from_record" in script_text,
        "legacy_post_eval_disabled": "v53 disables legacy --run-post-eval" in script_text and "v53 disables legacy --post-eval-only" in script_text,
    }

    feature_counts: Dict[str, int] = {}
    feature_counts_ok = False
    if Path(args.detector_artifact).is_file():
        feature_counts = load_feature_counts(Path(args.detector_artifact))
        feature_counts_ok = all(feature_counts.get(k) == v for k, v in EXPECTED_DETECTOR_FEATURE_COUNTS.items())

    dim_stats_missing = []
    if Path(args.cal_json).is_file():
        cal = load_json(Path(args.cal_json))
        dim_stats = cal.get("dim_stats", {}) if isinstance(cal, dict) else {}
        for keys in REQUIRED_DIM_STATS:
            val = nested_get(dim_stats, keys)
            if not is_finite_number(val):
                dim_stats_missing.append({"path": ".".join(keys), "value": repr(val)})

    d_targets_file = pkg / "resources" / "v49_d_full_targets.txt"
    d_targets_text = d_targets_file.read_text(encoding="utf-8") if d_targets_file.is_file() else ""
    d_targets_ok = ("d_lm_head=lm_head" in d_targets_text) and ("d_model_norm=model.norm" in d_targets_text)

    summary = {
        "version": "v54_preflight_exact_requirements",
        "policy": "No fuzzy matching, no guessed paths. Detector inputs are exact-reindexed to feature_columns.json; missing sparse per-row values are explicit NaN/0-sanitized values, not guessed columns. Labels are posthoc only and never detector features.",
        "package_dir": str(pkg),
        "missing_required_inputs": missing,
        "script_checks": script_checks,
        "detector_feature_counts": feature_counts,
        "expected_detector_feature_counts": EXPECTED_DETECTOR_FEATURE_COUNTS,
        "detector_feature_counts_ok": feature_counts_ok,
        "calibration_required_dim_stats_missing_or_nonfinite": dim_stats_missing,
        "d_targets_include_dout": d_targets_ok,
        "preflight_ok": bool((not missing) and all(script_checks.values()) and feature_counts_ok and not dim_stats_missing and d_targets_ok),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["preflight_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
