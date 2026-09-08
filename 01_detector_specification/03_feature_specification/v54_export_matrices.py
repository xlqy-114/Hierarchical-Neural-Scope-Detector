#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export v54 online detector matrices, posthoc labels, and record-audit manifests.

Input is v53/v54 inline detector records generated sample-inline during inference.
The strict detector-input matrix contains only the exact union of feature_columns.json
from the exported v35.1 selected detectors. Extra online-safe diagnostics are kept
in a separate matrix for later detector improvement.

This tool also writes:
- feature inventory: exact v35.1 input vs extra diagnostics vs analysis-only fields;
- target pool manifest: implementation-level eligible target pool;
- sampled target summary: what was actually sampled in this run;
- record completeness audit: per-sample presence of feature/fault/label/metric fields.
"""
from __future__ import annotations
import argparse, csv, json, math, tarfile, tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple
from collections import Counter, defaultdict
import pandas as pd

LABEL_NAMES = ["harmful", "severe_harmful"]
LEAKAGE_FORBIDDEN_SUBSTRINGS = [
    "harmful", "severe_harmful", "label", "clean_ref", "gt_metric",
    "fault_target", "fault_bit", "fault_before", "fault_after",
]
ALLOWED_EXACT_ONLINE_COLUMNS = {
    "raw_input_alarm", "critical_input_alarm", "suspicious_input_alarm",
    "e_output_sanity_alarm", "v19_output_first_harm_alarm",
}
ALLOWED_NON_FEATURE_COLUMNS = {"seed", "sample_id", "raw_sample_id", "row_source"} | ALLOWED_EXACT_ONLINE_COLUMNS
S0_CAN_BUS_GROUPS = [
    "translation_xyz", "rotation_quat", "small_signed_state", "lp_9", "lp_13",
    "near_zero_slots", "patch_angle_pair",
]
DEFAULT_C_TARGET_SPECS = "c_mm_projector_scene=mm_projector_scene,c_mm_projector_scene_layer0=mm_projector_scene.0,c_mm_projector_scene_layer1=mm_projector_scene.1,c_mm_projector_scene_layer2=mm_projector_scene.2"


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_specs(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for part in str(text or "").strip().split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            continue
        name, suffix = part.split('=', 1)
        out.append((name.strip(), suffix.strip()))
    return [(n, s) for n, s in out if n]


def parse_weights(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in str(text or "").strip().split(','):
        part = part.strip()
        if not part or ':' not in part:
            continue
        k, v = part.split(':', 1)
        try:
            out[k.strip()] = float(v)
        except Exception:
            out[k.strip()] = math.nan
    return out


def load_feature_columns(artifact: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with tempfile.TemporaryDirectory(prefix="v54_feature_cols_") as td:
        with tarfile.open(artifact, "r:gz") as tf:
            tf.extractall(td)
        for p in Path(td).rglob("feature_columns.json"):
            cols = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(cols, list):
                out[p.parent.name] = [str(c) for c in cols]
    if not out:
        raise RuntimeError(f"no feature_columns.json loaded from {artifact}")
    return out


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


def feature_family(name: str) -> str:
    n = str(name)
    if n in ALLOWED_EXACT_ONLINE_COLUMNS:
        return "top_level_online_alarm"
    if n.startswith("feat__"):
        parts = n.split("__")
        if len(parts) >= 2:
            return parts[1]
    return "unknown"


def classify_exact_layer_from_target(scope: str, logical_name: str, module_name: str = "", s_layer: str = "", abcde_subregion: str = "") -> str:
    scope = str(scope or "")
    logical_name = str(logical_name or "")
    module_name = str(module_name or "")
    s_layer = str(s_layer or "")
    abcde_subregion = str(abcde_subregion or "")
    if scope == "S0":
        return "S0"
    if scope == "C":
        return "C"
    if scope == "D":
        if "D_out" in abcde_subregion or logical_name in {"d_lm_head", "d_model_norm"} or module_name in {"lm_head", "model.norm"}:
            return "D_out"
        return "D_LLM"
    if scope == "S":
        source = s_layer or logical_name
        for key in ["S0", "S1", "S2", "S3", "S4", "S5"]:
            if str(source).startswith(key) or f"_{key.lower()}" in logical_name.lower() or logical_name.lower().startswith(key.lower()):
                return key
        if "backbone" in logical_name or "neck" in logical_name:
            return "S1"
        if "encoder_l0" in logical_name or "bev" in logical_name or "can_bus_mlp" in logical_name:
            return "S2"
        if "encoder" in logical_name:
            return "S3"
        if "decoder" in logical_name or "track" in logical_name or "seg" in logical_name:
            return "S4"
        return "S_unknown"
    return scope or "unknown"


def build_target_pool_manifest(resources_dir: Path, c_target_specs: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    scope_weights = parse_weights((resources_dir / "v50_global_random_scope_weights.txt").read_text(encoding="utf-8") if (resources_dir / "v50_global_random_scope_weights.txt").is_file() else "")

    # S0 can_bus groups are sampled internally by the inference script.
    for g in S0_CAN_BUS_GROUPS:
        rows.append({
            "logical_name": f"s0_can_bus_{g}", "module_name": "can_bus", "scope": "S0",
            "exact_layer": "S0", "target_family": "can_bus_group", "sampling_weight_in_scope": 1.0,
            "scope_weight": scope_weights.get("S0_only", math.nan), "source_file": "static_v35_can_bus_groups",
            "included_in_target_pool": 1,
        })

    s_map = resources_dir / "v49_s_full_target_map.csv"
    s_weights = parse_weights((resources_dir / "v50_s_full_target_weights_global_random.txt").read_text(encoding="utf-8") if (resources_dir / "v50_s_full_target_weights_global_random.txt").is_file() else "")
    if s_map.is_file():
        with s_map.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                logical = str(r.get("logical_name", "")).strip()
                suffix = str(r.get("suffix", "")).strip()
                s_layer = str(r.get("s_layer", "")).strip()
                if not logical:
                    continue
                rows.append({
                    "logical_name": logical, "module_name": suffix, "scope": "S", "exact_layer": classify_exact_layer_from_target("S", logical, suffix, s_layer=s_layer),
                    "target_family": s_layer or "S_full_registry", "sampling_weight_in_scope": s_weights.get(logical, r.get("sampling_weight", "")),
                    "scope_weight": scope_weights.get("S1_only", math.nan), "source_file": str(s_map), "included_in_target_pool": 1,
                })
    else:
        s_specs = resources_dir / "v49_s_full_target_specs.txt"
        if s_specs.is_file():
            for logical, suffix in parse_specs(s_specs.read_text(encoding="utf-8")):
                rows.append({
                    "logical_name": logical, "module_name": suffix, "scope": "S", "exact_layer": classify_exact_layer_from_target("S", logical, suffix),
                    "target_family": "S_full_specs", "sampling_weight_in_scope": s_weights.get(logical, math.nan),
                    "scope_weight": scope_weights.get("S1_only", math.nan), "source_file": str(s_specs), "included_in_target_pool": 1,
                })

    c_weights = parse_weights((resources_dir / "v50_c_target_weights_global_random.txt").read_text(encoding="utf-8") if (resources_dir / "v50_c_target_weights_global_random.txt").is_file() else "")
    for logical, suffix in parse_specs(c_target_specs):
        rows.append({
            "logical_name": logical, "module_name": suffix, "scope": "C", "exact_layer": "C",
            "target_family": "mm_projector_scene", "sampling_weight_in_scope": c_weights.get(logical, 1.0),
            "scope_weight": scope_weights.get("C_only", math.nan), "source_file": "run_script_c_target_specs", "included_in_target_pool": 1,
        })

    d_weights = parse_weights((resources_dir / "v50_d_target_weights_global_random.txt").read_text(encoding="utf-8") if (resources_dir / "v50_d_target_weights_global_random.txt").is_file() else "")
    d_map = resources_dir / "v49_d_full_target_map.csv"
    d_map_rows: Dict[str, Dict[str, str]] = {}
    if d_map.is_file():
        with d_map.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                logical = str(r.get("logical_name", "")).strip()
                if logical:
                    d_map_rows[logical] = r
    d_targets = resources_dir / "v49_d_full_targets.txt"
    if d_targets.is_file():
        for logical, suffix in parse_specs(d_targets.read_text(encoding="utf-8")):
            mr = d_map_rows.get(logical, {})
            abcde = str(mr.get("abcde_subregion", ""))
            rows.append({
                "logical_name": logical, "module_name": suffix, "scope": "D", "exact_layer": classify_exact_layer_from_target("D", logical, suffix, abcde_subregion=abcde),
                "target_family": abcde or "D_target", "sampling_weight_in_scope": d_weights.get(logical, 1.0),
                "scope_weight": scope_weights.get("D_only", math.nan), "source_file": str(d_targets), "included_in_target_pool": 1,
            })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--seed-root", default="A1_internal_neural_scdout_v54_online_singlefault")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--detector-artifact", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resources-dir", default="")
    ap.add_argument("--c-target-specs", default=DEFAULT_C_TARGET_SPECS)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    campaign = Path(args.campaign_dir)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    resources_dir = Path(args.resources_dir) if args.resources_dir else campaign / "resources"
    seeds = [s.strip().replace("seed", "") for s in args.seeds.split(",") if s.strip()]
    required_cols_by_detector = load_feature_columns(Path(args.detector_artifact))
    required_union: Set[str] = set()
    for cols in required_cols_by_detector.values():
        required_union.update(cols)

    records: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    analysis_rows: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []
    file_rows: List[Dict[str, Any]] = []
    completeness_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        p = campaign / args.seed_root / f"seed{seed}" / "results" / "v53_inline_detector_records.jsonl"
        if not p.is_file():
            # allow v54-named records if future inference changes filename
            alt = campaign / args.seed_root / f"seed{seed}" / "results" / "v54_inline_detector_records.jsonl"
            if alt.is_file():
                p = alt
            else:
                raise FileNotFoundError(p)
        n = 0
        for r in read_jsonl(p):
            n += 1
            r["seed"] = seed
            records.append(r)
            feat = dict(r.get("deployable_feature_row", {}) or {})
            if not feat:
                raise RuntimeError(f"missing deployable_feature_row seed={seed} sample={r.get('sample_id')}")
            row = {"seed": seed, "sample_id": str(r.get("sample_id", feat.get("sample_id", ""))), "raw_sample_id": str(r.get("raw_sample_id", "")), "row_source": str(r.get("row_source", feat.get("row_source", "")))}
            for k, v in feat.items():
                ks = str(k)
                if ks.startswith("feat__") or ks in ALLOWED_EXACT_ONLINE_COLUMNS:
                    row[ks] = v
            feature_rows.append(row)

            a = r.get("analysis_only", {}) or {}
            deltas = a.get("delta_metrics", {}) or {}
            labels = a.get("posthoc_labels", {}) or {}
            ar = {
                "seed": seed,
                "sample_id": r.get("sample_id", ""),
                "raw_sample_id": r.get("raw_sample_id", ""),
                "row_source": r.get("row_source", ""),
                "is_positive": a.get("is_positive", ""),
                "is_positive_effective": a.get("is_positive_effective", ""),
                "fault_scope_legacy": a.get("fault_scope_legacy", ""),
                "fault_target_exact": a.get("fault_target_exact", ""),
                "exact_layer": a.get("exact_layer", ""),
                "clean_ref_available": a.get("clean_ref_available", ""),
                "gt_metric_available": a.get("gt_metric_available", ""),
            }
            for k in ["delta_l2_1s", "delta_l2_2s", "delta_l2_3s", "delta_l2_avg", "delta_collision_1s", "delta_collision_2s", "delta_collision_3s", "delta_collision_avg", "clean_l2_avg", "fault_l2_avg", "clean_collision_avg", "fault_collision_avg"]:
                ar[k] = deltas.get(k, "")
            ar["faults_json"] = json.dumps(a.get("faults", []), ensure_ascii=False)
            analysis_rows.append(ar)

            lr = {"seed": seed, "sample_id": r.get("sample_id", ""), "row_source": r.get("row_source", ""), "exact_layer": a.get("exact_layer", ""), "injected_effective": labels.get("injected_effective", a.get("is_positive_effective", ""))}
            for k in ["delta_l2_avg", "delta_collision_avg", "harmful", "severe_harmful", "label_policy"]:
                lr[k] = labels.get(k, "")
            label_rows.append(lr)

            dets = r.get("detectors", {}) or {}
            completeness_rows.append({
                "seed": seed,
                "sample_id": r.get("sample_id", ""),
                "raw_sample_id": r.get("raw_sample_id", ""),
                "row_source": r.get("row_source", ""),
                "has_deployable_feature_row": int(bool(feat)),
                "deployable_feature_key_count": len(feat),
                "detector_count": len(dets),
                "final_alarm_any_selected_detector": r.get("final_alarm_any_selected_detector", ""),
                "is_positive": a.get("is_positive", ""),
                "is_positive_effective": a.get("is_positive_effective", ""),
                "fault_scope_legacy": a.get("fault_scope_legacy", ""),
                "fault_target_exact": a.get("fault_target_exact", ""),
                "exact_layer": a.get("exact_layer", ""),
                "has_faults": int(bool(a.get("faults", []))),
                "has_delta_metrics": int(bool(deltas)),
                "has_posthoc_labels": int(bool(labels)),
                "has_harmful_label": int("harmful" in labels),
                "has_severe_harmful_label": int("severe_harmful" in labels),
                "has_planning_metrics": int(bool(a.get("planning_metrics", {}) or a.get("planning_metrics_available", False))),
                "clean_ref_available": a.get("clean_ref_available", ""),
                "gt_metric_available": a.get("gt_metric_available", ""),
            })
        file_rows.append({"seed": seed, "path": str(p), "rows": n})

    all_deployable_cols = sorted({k for row in feature_rows for k in row if str(k).startswith("feat__") or str(k) in ALLOWED_EXACT_ONLINE_COLUMNS})
    exact_ordered_feature_cols = sorted(required_union)
    ordered_all_online_cols: List[str] = []
    for c in exact_ordered_feature_cols:
        if c not in ordered_all_online_cols:
            ordered_all_online_cols.append(c)
    for c in all_deployable_cols:
        if c not in ordered_all_online_cols:
            ordered_all_online_cols.append(c)

    id_cols = ["seed", "sample_id", "raw_sample_id", "row_source"]
    exact_feature_cols = id_cols + exact_ordered_feature_cols
    feature_matrix = outdir / "v54_online_deployable_feature_matrix.csv"
    pd.DataFrame(feature_rows).reindex(columns=exact_feature_cols).to_csv(feature_matrix, index=False)

    all_online_feature_cols = id_cols + ordered_all_online_cols
    all_online_feature_matrix = outdir / "v54_online_all_deployable_diagnostics_matrix.csv"
    pd.DataFrame(feature_rows).reindex(columns=all_online_feature_cols).to_csv(all_online_feature_matrix, index=False)

    analysis_csv = outdir / "v54_analysis_raw_metrics_and_fault_metadata.csv"
    analysis_df = pd.DataFrame(analysis_rows)
    analysis_df.to_csv(analysis_csv, index=False)
    label_csv = outdir / "v54_posthoc_label_table_harmful_severe.csv"
    labels_df = pd.DataFrame(label_rows)
    labels_df.to_csv(label_csv, index=False)

    score_rows = []
    for r in records:
        base = {"seed": r.get("seed", ""), "sample_id": r.get("sample_id", ""), "row_source": r.get("row_source", ""), "exact_layer": (r.get("analysis_only", {}) or {}).get("exact_layer", ""), "fault_target_exact": (r.get("analysis_only", {}) or {}).get("fault_target_exact", ""), "final_alarm_any_selected_detector": r.get("final_alarm_any_selected_detector", 0)}
        for det, d in (r.get("detectors", {}) or {}).items():
            rr = dict(base)
            rr.update({"detector": det, "score": d.get("score", ""), "threshold": d.get("threshold", ""), "alarm": d.get("alarm", ""), "row_feature_missing_count": d.get("row_feature_missing_count", d.get("row_feature_missing_filled_count", "")), "row_feature_nonfinite_sanitized_count": d.get("row_feature_nonfinite_sanitized_count", "")})
            score_rows.append(rr)
    score_csv = outdir / "v54_detector_score_cache.csv"
    scores_df = pd.DataFrame(score_rows)
    scores_df.to_csv(score_csv, index=False)

    leakage_cols = []
    for c in exact_feature_cols:
        if c in ALLOWED_NON_FEATURE_COLUMNS:
            continue
        if not c.startswith("feat__"):
            leakage_cols.append(c)
            continue
        lc = c.lower()
        if c.startswith("feat__v19_fault_induced_proxy__"):
            continue
        if any(s in lc for s in LEAKAGE_FORBIDDEN_SUBSTRINGS):
            leakage_cols.append(c)
    leakage_csv = outdir / "v54_deployable_feature_leakage_audit.csv"
    # Always write headers even when there are zero leakage rows.  Pandas
    # otherwise writes a 0-byte file for an empty DataFrame with no columns,
    # and the downstream smoke checker cannot read it.
    pd.DataFrame(
        [{"column": c, "matrix": "exact_detector_input"} for c in leakage_cols],
        columns=["column", "matrix"],
    ).to_csv(leakage_csv, index=False)

    metric_rows = []
    flat_scores = scores_df.merge(labels_df, on=["seed", "sample_id", "row_source", "exact_layer"], how="left") if not scores_df.empty else pd.DataFrame()
    if not flat_scores.empty:
        for det, gdet in flat_scores.groupby("detector"):
            for lab in LABEL_NAMES:
                m = binary_metrics(gdet[lab], gdet["alarm"])
                m.update({"detector": det, "label": lab, "scope": "ALL"})
                metric_rows.append(m)
                for layer, gl in gdet.groupby("exact_layer", dropna=False):
                    mm = binary_metrics(gl[lab], gl["alarm"])
                    mm.update({"detector": det, "label": lab, "scope": str(layer)})
                    metric_rows.append(mm)
    metrics_csv = outdir / "v54_posthoc_metrics_harmful_severe.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_csv, index=False)

    # Feature inventory: exact input, extra diagnostics, and analysis-only fields.
    used_by: Dict[str, List[str]] = defaultdict(list)
    for det, cols in required_cols_by_detector.items():
        for c in cols:
            used_by[str(c)].append(det)
    inventory_rows: List[Dict[str, Any]] = []
    for c in ordered_all_online_cols:
        is_exact = c in required_union
        inventory_rows.append({
            "feature_name": c,
            "feature_family": feature_family(c),
            "feature_role": "v35_1_exact_detector_input" if is_exact else "extra_online_diagnostic",
            "is_v35_1_detector_input": int(is_exact),
            "is_extra_online_diagnostic": int(not is_exact),
            "is_analysis_only": 0,
            "used_by_detectors": ";".join(sorted(used_by.get(c, []))),
            "online_safe_policy": "current_sample_inference_feature_or_online_proxy",
        })
    analysis_columns = set(analysis_df.columns if not analysis_df.empty else []) | set(labels_df.columns if not labels_df.empty else [])
    for c in sorted(analysis_columns):
        if c in {"seed", "sample_id", "raw_sample_id", "row_source"}:
            continue
        inventory_rows.append({
            "feature_name": c,
            "feature_family": "analysis_only",
            "feature_role": "posthoc_analysis_or_evaluation_label",
            "is_v35_1_detector_input": 0,
            "is_extra_online_diagnostic": 0,
            "is_analysis_only": 1,
            "used_by_detectors": "",
            "online_safe_policy": "not_detector_input; may include fault metadata, GT/clean/delta, or evaluation labels",
        })
    feature_inventory_csv = outdir / "v54_feature_inventory_existing_vs_new.csv"
    feature_inventory_df = pd.DataFrame(inventory_rows)
    feature_inventory_df.to_csv(feature_inventory_csv, index=False)
    feature_inventory_summary = {
        "version": "v54_feature_inventory_v1",
        "exact_detector_input_feature_count": int(sum(1 for r in inventory_rows if r["is_v35_1_detector_input"] == 1)),
        "extra_online_diagnostic_feature_count": int(sum(1 for r in inventory_rows if r["is_extra_online_diagnostic"] == 1)),
        "analysis_only_field_count": int(sum(1 for r in inventory_rows if r["is_analysis_only"] == 1)),
        "detector_feature_counts": {k: len(v) for k, v in required_cols_by_detector.items()},
        "outputs": {"feature_inventory_csv": str(feature_inventory_csv)},
    }
    (outdir / "v54_feature_inventory_summary.json").write_text(json.dumps(feature_inventory_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Target pool manifest and sampled target summary.
    target_pool_df = build_target_pool_manifest(resources_dir, args.c_target_specs)
    target_pool_csv = outdir / "v54_target_pool_manifest.csv"
    target_pool_df.to_csv(target_pool_csv, index=False)
    target_pool_summary = {
        "version": "v54_target_pool_manifest_v1",
        "resources_dir": str(resources_dir),
        "target_count_total": int(len(target_pool_df)),
        "target_count_by_exact_layer": dict(Counter(target_pool_df.get("exact_layer", pd.Series(dtype=str)).astype(str))) if not target_pool_df.empty else {},
        "target_count_by_scope": dict(Counter(target_pool_df.get("scope", pd.Series(dtype=str)).astype(str))) if not target_pool_df.empty else {},
        "contains_required_groups": {k: bool((target_pool_df.get("exact_layer", pd.Series(dtype=str)).astype(str) == k).any()) for k in ["S0", "S1", "S2", "S3", "S4", "S5", "C", "D_LLM", "D_out"]},
        "outputs": {"target_pool_manifest_csv": str(target_pool_csv)},
    }
    (outdir / "v54_target_pool_manifest_summary.json").write_text(json.dumps(target_pool_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    comp_df = pd.DataFrame(completeness_rows)
    completeness_csv = outdir / "v54_sample_record_completeness_audit.csv"
    comp_df.to_csv(completeness_csv, index=False)
    completeness_summary = {
        "version": "v54_sample_record_completeness_audit_v1",
        "rows": int(len(comp_df)),
        "missing_deployable_feature_row": int((pd.to_numeric(comp_df.get("has_deployable_feature_row", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int) == 0).sum()) if not comp_df.empty else 0,
        "missing_posthoc_labels": int((pd.to_numeric(comp_df.get("has_posthoc_labels", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int) == 0).sum()) if not comp_df.empty else 0,
        "missing_delta_metrics": int((pd.to_numeric(comp_df.get("has_delta_metrics", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int) == 0).sum()) if not comp_df.empty else 0,
        "outputs": {"completeness_csv": str(completeness_csv)},
    }
    (outdir / "v54_sample_record_completeness_summary.json").write_text(json.dumps(completeness_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    sampled_rows: List[Dict[str, Any]] = []
    if not comp_df.empty:
        tmp = comp_df.merge(labels_df, on=["seed", "sample_id", "row_source", "exact_layer"], how="left", suffixes=("", "_label"))
        for (layer, target), g in tmp.groupby(["exact_layer", "fault_target_exact"], dropna=False):
            base = {
                "exact_layer": str(layer),
                "fault_target_exact": str(target),
                "rows": int(len(g)),
                "injected_rows": int(pd.to_numeric(g.get("is_positive", 0), errors="coerce").fillna(0).astype(int).sum()),
                "effective_rows": int(pd.to_numeric(g.get("is_positive_effective", 0), errors="coerce").fillna(0).astype(int).sum()),
                "harmful_total": int(pd.to_numeric(g.get("harmful", 0), errors="coerce").fillna(0).astype(int).sum()),
                "severe_harmful_total": int(pd.to_numeric(g.get("severe_harmful", 0), errors="coerce").fillna(0).astype(int).sum()),
            }
            if not scores_df.empty:
                sg = scores_df[(scores_df["exact_layer"].astype(str) == str(layer)) & (scores_df["fault_target_exact"].astype(str) == str(target))]
                for det, gd in sg.groupby("detector"):
                    base[f"alarm_count__{det}"] = int(pd.to_numeric(gd.get("alarm", 0), errors="coerce").fillna(0).astype(int).sum())
            sampled_rows.append(base)
    sampled_csv = outdir / "v54_sampled_fault_target_summary.csv"
    sampled_df = pd.DataFrame(sampled_rows)
    sampled_df.to_csv(sampled_csv, index=False)
    sampled_summary = {
        "version": "v54_sampled_fault_target_summary_v1",
        "rows": int(len(records)),
        "sampled_target_group_count": int(len(sampled_df)),
        "sampled_count_by_exact_layer": dict(Counter(comp_df.get("exact_layer", pd.Series(dtype=str)).astype(str))) if not comp_df.empty else {},
        "note": "This is what the random smoke/formal run actually sampled; absence here does not mean absence from target pool.",
        "outputs": {"sampled_fault_target_summary_csv": str(sampled_csv)},
    }
    (outdir / "v54_sampled_fault_target_summary.json").write_text(json.dumps(sampled_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "version": "v54_online_record_audit_matrix_and_posthoc_export",
        "campaign_dir": str(campaign),
        "seed_root": args.seed_root,
        "seeds": seeds,
        "rows": len(records),
        "files": file_rows,
        "detectors": sorted(required_cols_by_detector.keys()),
        "required_feature_counts": {k: len(v) for k, v in required_cols_by_detector.items()},
        "deployable_feature_column_count": len(exact_ordered_feature_cols),
        "all_online_diagnostic_feature_column_count": len(ordered_all_online_cols),
        "extra_online_diagnostic_feature_count": int(len(set(ordered_all_online_cols) - set(exact_ordered_feature_cols))),
        "leakage_column_count": len(leakage_cols),
        "label_policy": "harmful and severe_harmful are posthoc evaluation labels only; they are not in deployable features.",
        "label_totals": {lab: int(pd.to_numeric(labels_df.get(lab, pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int).sum()) for lab in LABEL_NAMES},
        "exact_layer_counts": dict(Counter(labels_df.get("exact_layer", pd.Series(dtype=str)).astype(str))) if not labels_df.empty else {},
        "record_audit": {
            "feature_inventory_summary": feature_inventory_summary,
            "target_pool_manifest_summary": target_pool_summary,
            "sample_record_completeness_summary": completeness_summary,
            "sampled_fault_target_summary": sampled_summary,
        },
        "outputs": {
            "deployable_feature_matrix_csv": str(feature_matrix),
            "all_online_diagnostics_matrix_csv": str(all_online_feature_matrix),
            "analysis_raw_metrics_csv": str(analysis_csv),
            "posthoc_label_table_csv": str(label_csv),
            "detector_score_cache_csv": str(score_csv),
            "posthoc_metrics_csv": str(metrics_csv),
            "deployable_feature_leakage_audit_csv": str(leakage_csv),
            "feature_inventory_csv": str(feature_inventory_csv),
            "target_pool_manifest_csv": str(target_pool_csv),
            "sampled_fault_target_summary_csv": str(sampled_csv),
            "sample_record_completeness_audit_csv": str(completeness_csv),
        },
        "pass": bool(len(leakage_cols) == 0),
    }
    (outdir / "v54_online_export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.strict and leakage_cols:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
