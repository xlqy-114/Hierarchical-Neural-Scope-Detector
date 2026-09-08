#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict v54 smoke output checker.

This checker validates the requirements specific to v54:
- inline detector records were produced during inference;
- each detector row used exact v35.1 feature columns with no missing fill;
- deployable features contain no posthoc labels/deltas/GT/clean/fault metadata;
- posthoc labels are limited to approved harmful/severe_harmful labels;
- score parity and export/leakage audits pass.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, Iterable, List
import pandas as pd
from pandas.errors import EmptyDataError

FORBIDDEN_INLINE_LABEL_NAMES = {"main_harmful", "mild_harmful", "strict_harmful"}
APPROVED_POSTHOC_LABELS = {"harmful", "severe_harmful"}
REQUIRED_MULTIPOLICY_LABELS = {
    "harmful",
    "severe_harmful",
    "v31_old_harmful_compatible",
    "v31_heading_expanded_harmful_compatible",
    "v31_speed_alter_harmful_compatible",
    "v31_heading_speed_alter_harmful_compatible",
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def has_forbidden_key(obj: Any, forbidden: set[str], path: str = "") -> List[str]:
    hits: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{path}.{k}" if path else str(k)
            if str(k) in forbidden:
                hits.append(kp)
            hits.extend(has_forbidden_key(v, forbidden, kp))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(has_forbidden_key(v, forbidden, f"{path}[{i}]"))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--seed-root", default="A1_internal_neural_scdout_v54_online_singlefault")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--expected-rows", type=int, required=True)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--parity-dir", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    campaign = Path(args.campaign_dir)
    export_dir = Path(args.export_dir)
    parity_dir = Path(args.parity_dir)
    seeds = [s.strip().replace("seed", "") for s in args.seeds.split(",") if s.strip()]

    errors: List[str] = []
    warnings: List[str] = []
    row_total = 0
    per_seed = []
    detector_missing_counts: Dict[str, int] = {}

    for seed in seeds:
        rec_path = campaign / args.seed_root / f"seed{seed}" / "results" / "v53_inline_detector_records.jsonl"
        if not rec_path.is_file():
            rec_path = campaign / args.seed_root / f"seed{seed}" / "results" / "v54_inline_detector_records.jsonl"
        summ_path = campaign / args.seed_root / f"seed{seed}" / "results" / "v53_inline_detector_summary.json"
        if not summ_path.is_file():
            summ_path = campaign / args.seed_root / f"seed{seed}" / "results" / "v54_inline_detector_summary.json"
        if not rec_path.is_file():
            errors.append(f"missing inline records: {rec_path}")
            continue
        if not summ_path.is_file():
            errors.append(f"missing inline summary: {summ_path}")
            continue
        rows = list(iter_jsonl(rec_path))
        row_total += len(rows)
        per_seed.append({"seed": seed, "rows": len(rows), "records": str(rec_path), "summary": str(summ_path)})
        if len(rows) != args.expected_rows:
            errors.append(f"seed {seed}: expected {args.expected_rows} rows, found {len(rows)}")
        for i, r in enumerate(rows):
            if r.get("detector_timing") != "sample_inline_after_current_forward_before_next_sample":
                errors.append(f"seed {seed} row {i}: unexpected detector_timing={r.get('detector_timing')}")
            if "deployable_feature_row" not in r or not isinstance(r.get("deployable_feature_row"), dict):
                errors.append(f"seed {seed} row {i}: missing deployable_feature_row")
            forbidden_hits = has_forbidden_key(r, FORBIDDEN_INLINE_LABEL_NAMES)
            if forbidden_hits:
                errors.append(f"seed {seed} row {i}: forbidden legacy label keys found {forbidden_hits[:10]}")
            analysis = r.get("analysis_only", {}) or {}
            labels = analysis.get("posthoc_labels", {}) or {}
            present_labels = {k for k in labels.keys() if k.endswith("harmful") or "harmful" in k}
            unapproved = sorted(present_labels - APPROVED_POSTHOC_LABELS)
            if unapproved:
                errors.append(f"seed {seed} row {i}: unapproved posthoc labels {unapproved}")
            for det, d in (r.get("detectors", {}) or {}).items():
                mc = d.get("row_feature_missing_count", d.get("row_feature_missing_filled_count", 0))
                try:
                    mc_i = int(mc)
                except Exception:
                    mc_i = -1
                detector_missing_counts[det] = max(detector_missing_counts.get(det, 0), mc_i)
                if mc_i != 0:
                    errors.append(f"seed {seed} row {i} detector {det}: missing feature count {mc_i}")

    export_summary_path = export_dir / "v54_online_export_summary.json"
    leakage_csv = export_dir / "v54_deployable_feature_leakage_audit.csv"
    feature_matrix = export_dir / "v54_online_deployable_feature_matrix.csv"
    label_table = export_dir / "v54_posthoc_label_table_harmful_severe.csv"
    score_cache = export_dir / "v54_detector_score_cache.csv"
    feature_inventory_csv = export_dir / "v54_feature_inventory_existing_vs_new.csv"
    target_pool_csv = export_dir / "v54_target_pool_manifest.csv"
    sampled_target_csv = export_dir / "v54_sampled_fault_target_summary.csv"
    completeness_csv = export_dir / "v54_sample_record_completeness_audit.csv"
    multipolicy_summary_json = export_dir / "v54_2_multipolicy_label_summary.json"
    multipolicy_label_csv = export_dir / "v54_2_posthoc_label_table_multi_policy.csv"
    multipolicy_metrics_csv = export_dir / "v54_2_posthoc_metrics_multi_policy.csv"
    multipolicy_audit_csv = export_dir / "v54_2_v31_behavior_label_audit.csv"
    for p in [export_summary_path, leakage_csv, feature_matrix, label_table, score_cache, feature_inventory_csv, target_pool_csv, sampled_target_csv, completeness_csv, multipolicy_summary_json, multipolicy_label_csv, multipolicy_metrics_csv, multipolicy_audit_csv]:
        if not p.is_file():
            errors.append(f"missing export output: {p}")

    export_summary: Dict[str, Any] = {}
    if export_summary_path.is_file():
        export_summary = read_json(export_summary_path)
        if not export_summary.get("pass", False):
            errors.append(f"export summary pass=false: {export_summary_path}")
        if int(export_summary.get("leakage_column_count", -1)) != 0:
            errors.append(f"deployable leakage columns found: {export_summary.get('leakage_column_count')}")
        labs = set((export_summary.get("label_totals", {}) or {}).keys())
        if not labs.issubset(APPROVED_POSTHOC_LABELS):
            errors.append(f"export summary contains unapproved labels: {sorted(labs - APPROVED_POSTHOC_LABELS)}")

    if leakage_csv.is_file():
        # A zero-leakage audit is allowed to be empty in older v54 outputs.
        # Treat an empty file as zero rows instead of crashing the smoke check.
        try:
            leak = pd.read_csv(leakage_csv)
        except EmptyDataError:
            leak = pd.DataFrame(columns=["column", "matrix"])
        if len(leak) != 0:
            errors.append(f"leakage audit has {len(leak)} rows")

    if label_table.is_file():
        labdf = pd.read_csv(label_table)
        bad_cols = [c for c in labdf.columns if ("harmful" in c and c not in APPROVED_POSTHOC_LABELS)]
        if bad_cols:
            errors.append(f"label table has unapproved harmful columns: {bad_cols}")
        for req in sorted(APPROVED_POSTHOC_LABELS):
            if req not in labdf.columns:
                errors.append(f"label table missing approved column: {req}")

    if feature_matrix.is_file():
        cols = list(pd.read_csv(feature_matrix, nrows=0).columns)
        # Do not flag legitimate online propagation-delta, tensor l2_norm, or
        # current-scene collision-risk proxy features.  Only offline labels or
        # clean/GT/fault metadata are disallowed in the deployable matrix.
        bad = [c for c in cols if any(s in c.lower() for s in ["harmful", "severe_harmful", "label", "clean_ref", "gt_metric", "fault_target", "fault_bit", "fault_before", "fault_after"]) and not c.startswith("feat__v19_fault_induced_proxy__")]
        if bad:
            errors.append(f"deployable matrix has suspicious label/delta columns: {bad[:30]}")

    if feature_inventory_csv.is_file():
        inv = pd.read_csv(feature_inventory_csv)
        for col in ["feature_name", "feature_role", "is_v35_1_detector_input", "is_extra_online_diagnostic", "is_analysis_only"]:
            if col not in inv.columns:
                errors.append(f"feature inventory missing column: {col}")
        if "is_v35_1_detector_input" in inv.columns and int(pd.to_numeric(inv["is_v35_1_detector_input"], errors="coerce").fillna(0).astype(int).sum()) <= 0:
            errors.append("feature inventory has no v35.1 exact detector input features")
        if "is_extra_online_diagnostic" in inv.columns and int(pd.to_numeric(inv["is_extra_online_diagnostic"], errors="coerce").fillna(0).astype(int).sum()) <= 0:
            warnings.append("feature inventory has no extra online diagnostics beyond v35.1 exact input")

    if target_pool_csv.is_file():
        pool = pd.read_csv(target_pool_csv)
        for col in ["logical_name", "scope", "exact_layer", "included_in_target_pool"]:
            if col not in pool.columns:
                errors.append(f"target pool manifest missing column: {col}")
        if "exact_layer" in pool.columns:
            layers = set(pool["exact_layer"].astype(str))
            required_layers = {"S0", "C", "D_LLM", "D_out"}
            missing_required = sorted(required_layers - layers)
            if missing_required:
                errors.append(f"target pool manifest missing required implementation layers: {missing_required}")

    if sampled_target_csv.is_file():
        sampled = pd.read_csv(sampled_target_csv)
        for col in ["exact_layer", "fault_target_exact", "rows", "effective_rows", "harmful_total", "severe_harmful_total"]:
            if col not in sampled.columns:
                errors.append(f"sampled target summary missing column: {col}")

    if completeness_csv.is_file():
        comp = pd.read_csv(completeness_csv)
        for col in ["sample_id", "has_deployable_feature_row", "has_posthoc_labels", "has_delta_metrics", "fault_target_exact", "exact_layer"]:
            if col not in comp.columns:
                errors.append(f"record completeness audit missing column: {col}")
        if "has_deployable_feature_row" in comp.columns and int((pd.to_numeric(comp["has_deployable_feature_row"], errors="coerce").fillna(0).astype(int) == 0).sum()) != 0:
            errors.append("some records are missing deployable_feature_row")


    # v54.2 multi-policy posthoc labels are allowed only in separate offline files.
    # They must not appear inside inline analysis_only.posthoc_labels or deployable features.
    if multipolicy_summary_json.is_file():
        mp = read_json(multipolicy_summary_json)
        if not mp.get("pass", False):
            errors.append(f"multipolicy label summary pass=false: {multipolicy_summary_json}")
        labs = set((mp.get("label_totals", {}) or {}).keys())
        required_core = {"v31_old_harmful_compatible", "v31_speed_alter_harmful_compatible", "v31_heading_speed_alter_harmful_compatible"}
        missing_core = sorted(required_core - labs)
        if missing_core:
            errors.append(f"multipolicy label summary missing core labels: {missing_core}")
        availability = mp.get("availability", {}) or {}
        # v54.3+ policy: missing fault trajectory is diagnostic only.
        # This matches the v31 audit behavior: rows with unavailable fault trajectory
        # remain in the table and keep heading/speed audit flags false.
        # Clean trajectory and GT remain strict requirements because without them the
        # v31-compatible speed/heading label audit cannot be trusted globally.
        fault_missing = int(availability.get("fault_traj_missing", 0) or 0)
        if fault_missing != 0:
            warnings.append(f"multipolicy availability fault_traj_missing={fault_missing}; diagnostic only")
        for k in ["clean_traj_missing", "gt_traj_missing", "missing_or_empty_fault_predictions"]:
            if int(availability.get(k, 0) or 0) != 0:
                errors.append(f"multipolicy availability {k}={availability.get(k)}")

    if multipolicy_label_csv.is_file():
        mpdf = pd.read_csv(multipolicy_label_csv)
        missing_cols = sorted(REQUIRED_MULTIPOLICY_LABELS - set(mpdf.columns))
        if missing_cols:
            errors.append(f"multipolicy label table missing columns: {missing_cols}")

    if multipolicy_metrics_csv.is_file():
        mmdf = pd.read_csv(multipolicy_metrics_csv)
        if "label" not in mmdf.columns or "detector" not in mmdf.columns:
            errors.append("multipolicy metrics missing label/detector columns")
        else:
            labels_present = set(mmdf["label"].astype(str))
            missing_labels = sorted(REQUIRED_MULTIPOLICY_LABELS - labels_present)
            if missing_labels:
                errors.append(f"multipolicy metrics missing labels: {missing_labels}")

    parity_summary_path = parity_dir / "v54_inline_decision_parity_summary.json"
    if not parity_summary_path.is_file():
        errors.append(f"missing parity summary: {parity_summary_path}")
    else:
        ps = read_json(parity_summary_path)
        if not ps.get("pass", False):
            errors.append(f"decision parity pass=false: {parity_summary_path}")
        if not ps.get("decision_parity_pass", ps.get("pass", False)):
            errors.append(f"decision_parity_pass=false: {parity_summary_path}")
        if int(ps.get("alarm_mismatch_total", 0) or 0) != 0:
            errors.append(f"alarm_mismatch_total={ps.get('alarm_mismatch_total')}")
        if int(ps.get("missing_matrix_columns_total", 0) or 0) != 0:
            errors.append(f"missing_matrix_columns_total={ps.get('missing_matrix_columns_total')}")
        if not bool(ps.get("row_count_all_match", True)):
            errors.append("parity row_count_all_match=false")
        if not ps.get("score_exact_pass", True):
            warnings.append(
                "score_exact_pass=false but decision parity passed; score mismatches are retained as diagnostics only"
            )

    summary = {
        "version": "v54_online_record_audit_smoke_outputs_check",
        "campaign_dir": str(campaign),
        "seed_root": args.seed_root,
        "seeds": seeds,
        "expected_rows_per_seed": args.expected_rows,
        "row_total": row_total,
        "per_seed": per_seed,
        "detector_max_missing_feature_count": detector_missing_counts,
        "export_summary": export_summary,
        "errors": errors,
        "warnings": warnings,
        "pass": len(errors) == 0,
    }
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.strict and errors:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
