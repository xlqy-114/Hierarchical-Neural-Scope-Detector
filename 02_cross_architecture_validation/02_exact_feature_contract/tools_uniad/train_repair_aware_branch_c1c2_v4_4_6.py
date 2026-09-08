#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UniAD2 v4.4.6: strict nested repair-aware C1 and branch C2.

This experiment keeps the single-C1 deployment constraint but makes both C1
and C2 aware of telemetry produced by the online repair mechanism during the
same inference. The telemetry is aggregated from the event trace before the
risk decision and includes only repair actions, locations, scope, budget use,
and pre/post numerical recovery summaries. Clean/reference outputs, GT,
trajectory deltas, gate decisions, injected fault metadata, bit metadata, and
post-hoc safety labels are explicitly excluded from X.

Design
------
1. C1 uses the frozen activation feature contract plus deterministic online
   repair telemetry, so repair evidence can rescue candidates that an
   activation-only C1 scored too low.
2. The validated online nonfinite route remains available and separates C2-F
   and C2-NF, preserving the best v4.4.5 structure.
3. Both branch C2 models receive repair telemetry and the current C1's
   train-reference meta-scores, so they can distinguish successful repair from
   residual unsafe propagation.
4. Evaluation is strict outer/inner seed nesting. Route, calibration and all
   thresholds are selected from outer-training seeds only.
5. The default report contains only the user's decision metrics:
   highest Recall at FAR <= 2%, and minimum FAR at Recall >= 90%, 95%, 99%.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from model_utils_v4_4_6 import SafeFiniteClipper, SafeMedianImputer
from repair_telemetry_v4_4_6 import (
    DEBUG_SUMMARY_NAME,
    EVENTS_NAME,
    PREFIX as REPAIR_PREFIX,
    REPAIR_FEATURES,
    build_repair_telemetry,
    validate_source_allowlists,
)

VERSION = "uniad2_v4_4_6_2_repair_aware_full_exact_contract_branch_c1c2"
CSV_NAME = "detector_train_activation_only_v4_3_1.csv"
SUMMARY_NAME = "fault_repair_extended_feature_dataset_summary_v4_3_1.json"
AUDIT_NAME = "leakage_audit_v4_3_1.json"

TARGET = "derived__accepted_finite_unsafe_1p0"
ELIGIBLE = "derived__gate_accepted_finite_row"
VALIDITY = "derived__validity_alarm"
OVERALL_LABEL = "label__post_repair_unsafe_1p0_or_gate"
NONFINITE_SOURCE_FEATURE = "feature__act_nonfinite_count_sample_sum"
NONFINITE_ROUTE = "derived__internal_nonfinite_seen"
BRANCH_FINITE = "finite_only"
BRANCH_NONFINITE = "nonfinite_seen"

BASE_REQUIRED_LABELS = [
    "label__gate_accept",
    "label__gate_reject",
    "label__post_repair_unsafe_1p0_or_gate",
    "label__post_repair_accepted_safe_1p0",
    "label__repaired_output_nonfinite",
    "label__repaired_output_illegal_or_missing",
]

C1_META_FEATURES = [
    "v446_c1_train_percentile",
    "v446_c1_logit_robust_z",
    "v446_c1_top_0p1pct",
    "v446_c1_top_0p5pct",
    "v446_c1_top_1pct",
    "v446_c1_top_5pct",
    "v446_c1_top_10pct",
    "v446_c1_distance_to_route_cut_pct",
    "v446_c1_above_route_cut",
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def parse_tokens(s: str) -> List[str]:
    return [x for x in re.split(r"[\s,]+", str(s).strip()) if x]


def json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)


def sha256_text(lines: Sequence[str]) -> str:
    h = hashlib.sha256()
    for x in lines:
        h.update(str(x).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_seed(path: Path) -> str:
    m = re.search(r"_seed(\d+)_", path.parent.name)
    if not m:
        raise ValueError(f"Cannot infer seed from directory name: {path.parent.name}")
    return m.group(1)


@dataclass
class SeedMaterial:
    seed: str
    csv_path: Path
    events_path: Path
    debug_summary_path: Path
    out_dir: Path
    feature_columns: List[str]
    row_count: int
    header_sha256: str


def validate_seed_materials(
    data_root: Path,
    requested_seeds: List[str],
    expected_rows_per_seed: int,
) -> List[SeedMaterial]:
    all_paths = sorted(data_root.rglob(CSV_NAME))
    by_seed: Dict[str, List[Path]] = {}
    for p in all_paths:
        by_seed.setdefault(infer_seed(p), []).append(p)

    mats: List[SeedMaterial] = []
    errors: List[str] = []
    for seed in requested_seeds:
        paths = by_seed.get(seed, [])
        if len(paths) != 1:
            errors.append(f"seed={seed}: expected exactly one {CSV_NAME}, found {len(paths)}")
            continue
        csv_path = paths[0]
        out_dir = csv_path.parent
        summary_path = out_dir / SUMMARY_NAME
        audit_path = out_dir / AUDIT_NAME
        events_path = out_dir / EVENTS_NAME
        debug_summary_path = out_dir / DEBUG_SUMMARY_NAME
        if not summary_path.is_file():
            errors.append(f"seed={seed}: missing final summary {summary_path}")
            continue
        if not audit_path.is_file():
            errors.append(f"seed={seed}: missing leakage audit {audit_path}")
            continue
        if not events_path.is_file():
            errors.append(f"seed={seed}: missing online repair event trace {events_path}")
            continue
        if not debug_summary_path.is_file():
            errors.append(f"seed={seed}: missing debug case summary {debug_summary_path}")
            continue
        summary = load_json(summary_path)
        audit = load_json(audit_path)
        counters = summary.get("counters", {})
        bad: List[str] = []
        if int(counters.get("selected_cases_done", -1)) != expected_rows_per_seed:
            bad.append(f"selected_cases_done={counters.get('selected_cases_done')}")
        if int(counters.get("prefix_samples_processed", -1)) != expected_rows_per_seed:
            bad.append(f"prefix_samples_processed={counters.get('prefix_samples_processed')}")
        if int(counters.get("extra_feature_error_count", -1)) != 0:
            bad.append(f"extra_feature_error_count={counters.get('extra_feature_error_count')}")
        if int(audit.get("forbidden_hit_count", -1)) != 0:
            bad.append(f"forbidden_hit_count={audit.get('forbidden_hit_count')}")
        if audit.get("train_feature_forbidden_keys", []):
            bad.append(
                f"train_feature_forbidden_keys={len(audit.get('train_feature_forbidden_keys', []))}"
            )
        header = list(pd.read_csv(csv_path, nrows=0).columns)
        features = sorted([c for c in header if str(c).startswith("feature__")])
        if not features:
            bad.append("no feature__ columns")
        if len(set(header)) != len(header):
            bad.append("duplicate CSV header columns")
        missing_labels = [c for c in BASE_REQUIRED_LABELS if c not in header]
        if missing_labels:
            bad.append(f"missing labels={missing_labels}")
        if bad:
            errors.append(f"seed={seed}: " + "; ".join(bad))
            continue
        mats.append(
            SeedMaterial(
                seed=seed,
                csv_path=csv_path,
                events_path=events_path,
                debug_summary_path=debug_summary_path,
                out_dir=out_dir,
                feature_columns=features,
                row_count=expected_rows_per_seed,
                header_sha256=sha256_text(header),
            )
        )
    if errors:
        raise RuntimeError("Seed material validation failed:\n" + "\n".join(errors))
    return mats


def find_named_list(obj: Any, wanted_key: str, path: str = "root") -> List[Tuple[str, List[Any]]]:
    """Recursively find list-valued keys with an exact name."""
    hits: List[Tuple[str, List[Any]]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}"
            if key == wanted_key and isinstance(value, list):
                hits.append((child, value))
            hits.extend(find_named_list(value, wanted_key, child))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            hits.extend(find_named_list(value, wanted_key, f"{path}[{i}]"))
    return hits


def load_excluded_activation_features(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"Excluded activation feature file not found: {path}")
    features: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        features.append(value)
    if len(features) != len(set(features)):
        raise RuntimeError("Duplicate features in excluded activation feature file")
    return features


def build_frozen_feature_contract(
    mats: List[SeedMaterial],
    out_dir: Path,
    optional_v431_contract_json: Optional[Path],
    excluded_activation_feature_file: Optional[Path],
    expected_extended_source_feature_count: int,
    expected_runtime_summary_feature_count: int,
    expected_activation_feature_count: int,
) -> List[str]:
    """Build one exact, shared activation contract for all requested seeds.

    v4.3.1 has two legal online activation families:
      1. reviewed per-hook extended-source columns listed by
         allowed_activation_only_columns in feature_contract_v4_3_1.json;
      2. runtime global/group/role summary columns emitted directly by the
         activation logger into detector_train_activation_only_v4_3_1.csv.

    The v4.3.1 JSON intentionally enumerates family (1), not family (2).
    Therefore treating that JSON as the complete detector feature list drops
    the runtime summaries. v4.4.6.2 freezes both families explicitly:
      - reviewed source columns minus the four audited cohort-unstable columns;
      - exact runtime-summary feature columns present in every requested seed.

    Counts are mandatory and any mismatch aborts. CSV-only columns not shared
    by every seed are recorded and ignored. No value filling, fuzzy matching,
    or per-seed feature creation is performed.
    """
    validate_source_allowlists()
    if optional_v431_contract_json is None or not optional_v431_contract_json.is_file():
        raise FileNotFoundError(
            "An authoritative --feature-contract-json is required for v4.4.6.2 full exact-contract training"
        )

    raw = load_json(optional_v431_contract_json)
    hits = find_named_list(raw, "allowed_activation_only_columns")
    if len(hits) != 1:
        raise RuntimeError(
            "Expected exactly one allowed_activation_only_columns list in "
            f"{optional_v431_contract_json}, found={len(hits)} paths={[x[0] for x in hits]}"
        )
    allowed_path, allowed_raw = hits[0]
    reviewed_source_features = [str(c) for c in allowed_raw if str(c).startswith("feature__")]
    if not reviewed_source_features:
        raise RuntimeError("Authoritative v4.3.1 source contract contains no feature__ columns")
    if len(reviewed_source_features) != len(set(reviewed_source_features)):
        raise RuntimeError("Duplicate feature columns in authoritative v4.3.1 source contract")

    excluded_features = load_excluded_activation_features(excluded_activation_feature_file)
    reviewed_source_set = set(reviewed_source_features)
    unknown_exclusions = [c for c in excluded_features if c not in reviewed_source_set]
    if unknown_exclusions:
        raise RuntimeError(
            "Excluded feature list contains columns absent from the authoritative source contract: "
            + ", ".join(unknown_exclusions)
        )
    excluded_set = set(excluded_features)
    reduced_source_features = [c for c in reviewed_source_features if c not in excluded_set]

    if (
        expected_extended_source_feature_count > 0
        and len(reduced_source_features) != expected_extended_source_feature_count
    ):
        raise RuntimeError(
            f"Reduced reviewed-source feature count={len(reduced_source_features)} "
            f"expected={expected_extended_source_feature_count}. "
            "The v4.3.1 JSON enumerates only reviewed per-hook source features."
        )

    seed_feature_sets = {m.seed: set(m.feature_columns) for m in mats}
    common_csv_features = set.intersection(*(seed_feature_sets[m.seed] for m in mats))
    union_csv_features = set.union(*(seed_feature_sets[m.seed] for m in mats))

    # Runtime summaries are legal activation-only CSV features that are not
    # part of the reviewed per-hook source manifest and are present in every seed.
    runtime_summary_features = sorted(
        c for c in common_csv_features if c not in reviewed_source_set
    )
    if (
        expected_runtime_summary_feature_count > 0
        and len(runtime_summary_features) != expected_runtime_summary_feature_count
    ):
        raise RuntimeError(
            f"Exact common runtime-summary feature count={len(runtime_summary_features)} "
            f"expected={expected_runtime_summary_feature_count}. "
            "Do not train; inspect activation_full_exact_contract_audit_v4_4_6_2.json."
        )

    activation_features = sorted(set(reduced_source_features).union(runtime_summary_features))
    if expected_activation_feature_count > 0 and len(activation_features) != expected_activation_feature_count:
        raise RuntimeError(
            f"Full exact activation contract count={len(activation_features)} "
            f"expected={expected_activation_feature_count} "
            f"(reviewed_source={len(reduced_source_features)}, "
            f"runtime_summary={len(runtime_summary_features)})"
        )
    if NONFINITE_SOURCE_FEATURE not in activation_features:
        raise RuntimeError(
            f"Required online route feature is absent from full exact contract: {NONFINITE_SOURCE_FEATURE}"
        )

    selected_set = set(activation_features)
    dynamic_extras = sorted(union_csv_features - selected_set - excluded_set)
    fatal_errors: List[str] = []
    audit_rows: List[Dict[str, Any]] = []
    feature_presence: Dict[str, List[str]] = {c: [] for c in activation_features}

    for m in mats:
        csv_features = list(m.feature_columns)
        csv_set = set(csv_features)
        missing_selected = [c for c in activation_features if c not in csv_set]
        missing_reduced_source = [c for c in reduced_source_features if c not in csv_set]
        present_excluded = [c for c in excluded_features if c in csv_set]
        per_seed_unselected = sorted(csv_set - selected_set - excluded_set)
        for c in activation_features:
            if c in csv_set:
                feature_presence[c].append(m.seed)
        audit_rows.append(
            {
                "seed": m.seed,
                "csv_feature_count": len(csv_features),
                "reviewed_source_contract_feature_count": len(reviewed_source_features),
                "reduced_reviewed_source_feature_count": len(reduced_source_features),
                "runtime_summary_feature_count": len(runtime_summary_features),
                "selected_activation_feature_count": len(activation_features),
                "missing_selected_count": len(missing_selected),
                "missing_selected": " | ".join(missing_selected),
                "missing_reduced_source_count": len(missing_reduced_source),
                "missing_reduced_source": " | ".join(missing_reduced_source),
                "present_excluded_count": len(present_excluded),
                "present_excluded": " | ".join(present_excluded),
                "unselected_csv_feature_count": len(per_seed_unselected),
                "unselected_csv_features": " | ".join(per_seed_unselected),
                "header_sha256": m.header_sha256,
            }
        )
        if missing_selected:
            fatal_errors.append(
                f"seed={m.seed}: missing {len(missing_selected)} full exact activation features: "
                + ", ".join(missing_selected[:20])
            )

    audit_csv = out_dir / "activation_full_exact_contract_audit_v4_4_6_2.csv"
    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False)
    pd.DataFrame(
        {
            "feature": excluded_features,
            "reason": [
                "Absent from all 12 added seeds; excluded globally from every seed to prevent cohort-specific missingness"
            ] * len(excluded_features),
        }
    ).to_csv(out_dir / "excluded_activation_features_v4_4_6_2.csv", index=False)
    pd.DataFrame(
        {
            "feature": runtime_summary_features,
            "family": ["runtime_global_group_role_summary"] * len(runtime_summary_features),
            "present_seed_count": [len(mats)] * len(runtime_summary_features),
        }
    ).to_csv(out_dir / "runtime_summary_feature_contract_v4_4_6_2.csv", index=False)

    audit_payload = {
        "version": VERSION,
        "authoritative_source_contract_json": str(optional_v431_contract_json),
        "authoritative_allowed_list_path": allowed_path,
        "reviewed_source_contract_feature_count": len(reviewed_source_features),
        "explicitly_excluded_features": excluded_features,
        "reduced_reviewed_source_feature_count": len(reduced_source_features),
        "runtime_summary_feature_count": len(runtime_summary_features),
        "selected_activation_feature_count": len(activation_features),
        "selected_activation_feature_sha256": sha256_text(activation_features),
        "dynamic_or_cohort_specific_csv_features_ignored": dynamic_extras,
        "seed_audit": audit_rows,
    }
    write_json(out_dir / "activation_full_exact_contract_audit_v4_4_6_2.json", audit_payload)

    if fatal_errors:
        raise RuntimeError(
            "Full exact activation contract validation failed:\n"
            + "\n".join(fatal_errors)
            + f"\nAudit written to {audit_csv}"
        )

    rows: List[Dict[str, Any]] = []
    runtime_set = set(runtime_summary_features)
    for c in activation_features:
        if c in runtime_set:
            if c.startswith("feature__act_group__"):
                family = "runtime_group_summary"
            elif c.startswith("feature__act_role__"):
                family = "runtime_role_summary"
            else:
                family = "runtime_global_activation_summary"
            source_family = "runtime_summary"
        else:
            family = "v431_reviewed_extended_source_scalar"
            source_family = "reviewed_extended_source"
        rows.append(
            {
                "feature": c,
                "source_branch": CSV_NAME,
                "source_family": source_family,
                "decision": "allow_full_exact_shared_contract",
                "family": family,
                "present_seed_count": len(feature_presence[c]),
                "present_seeds": ",".join(sorted(feature_presence[c])),
                "uses_clean_reference": 0,
                "uses_fault_metadata": 0,
                "uses_repair_telemetry": 0,
                "uses_gate_decision_as_x": 0,
                "uses_label_or_delta_as_x": 0,
            }
        )
    for c in REPAIR_FEATURES:
        rows.append(
            {
                "feature": c,
                "source_branch": EVENTS_NAME,
                "source_family": "online_repair_telemetry",
                "decision": "allow_exact_frozen",
                "family": "online_repair_telemetry",
                "present_seed_count": len(mats),
                "present_seeds": ",".join(sorted(m.seed for m in mats)),
                "uses_clean_reference": 0,
                "uses_fault_metadata": 0,
                "uses_repair_telemetry": 1,
                "uses_gate_decision_as_x": 0,
                "uses_label_or_delta_as_x": 0,
            }
        )

    all_features = activation_features + list(REPAIR_FEATURES)
    pd.DataFrame(rows).to_csv(out_dir / "frozen_exact_feature_contract_v4_4_6_2.csv", index=False)
    pd.DataFrame([r for r in rows if r["uses_repair_telemetry"] == 1]).to_csv(
        out_dir / "repair_telemetry_feature_contract_v4_4_6_2.csv", index=False
    )
    write_json(
        out_dir / "frozen_exact_feature_contract_v4_4_6_2.json",
        {
            "version": VERSION,
            "reviewed_source_feature_count": len(reduced_source_features),
            "runtime_summary_feature_count": len(runtime_summary_features),
            "activation_feature_count": len(activation_features),
            "repair_telemetry_feature_count": len(REPAIR_FEATURES),
            "feature_count": len(all_features),
            "feature_sha256": sha256_text(all_features),
            "features": all_features,
            "activation_source": CSV_NAME,
            "repair_event_source": EVENTS_NAME,
            "repair_context_source": DEBUG_SUMMARY_NAME,
            "authoritative_v431_source_contract_json": str(optional_v431_contract_json),
            "authoritative_v431_allowed_list_path": allowed_path,
            "explicitly_excluded_activation_features": excluded_features,
            "dynamic_or_cohort_specific_csv_features_ignored": dynamic_extras,
            "policy": (
                "Full exact shared activation contract = reviewed v4.3.1 per-hook source columns "
                "minus four explicitly audited unstable columns + exact runtime global/group/role "
                "summary columns present in every requested seed. Counts are mandatory; every "
                "selected feature must physically exist in every seed. No value filling, fuzzy "
                "matching, or per-seed feature creation is permitted."
            ),
        },
    )
    return all_features

def read_one(
    m: SeedMaterial,
    all_features: List[str],
    repair_budget_max_default: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    header = list(pd.read_csv(m.csv_path, nrows=0).columns)
    header_set = set(header)
    activation_features = [c for c in all_features if not c.startswith(REPAIR_PREFIX)]
    missing_activation = [c for c in activation_features if c not in header_set]
    if missing_activation:
        raise RuntimeError(
            f"seed={m.seed}: exact selected activation columns missing before read: {missing_activation[:20]}"
        )
    required_nonfeatures = ["sample_i", "sample_id"] + list(BASE_REQUIRED_LABELS)
    missing_nonfeatures = [c for c in required_nonfeatures if c not in header_set]
    if missing_nonfeatures:
        raise RuntimeError(
            f"seed={m.seed}: required non-feature columns missing: {missing_nonfeatures}"
        )
    usecols = list(dict.fromkeys(required_nonfeatures + activation_features))
    dtype = {c: "float32" for c in activation_features}
    df = pd.read_csv(m.csv_path, usecols=usecols, dtype=dtype)
    if len(df) != m.row_count:
        raise RuntimeError(f"seed={m.seed}: CSV rows={len(df)} expected={m.row_count}")
    if "sample_id" not in df.columns:
        df["sample_id"] = np.arange(len(df)).astype(str)
    df["sample_i"] = pd.to_numeric(df["sample_i"], errors="raise").astype(int)
    df["sample_id"] = df["sample_id"].astype(str)
    if df["sample_i"].duplicated().any():
        raise RuntimeError(f"seed={m.seed}: duplicate sample_i in activation CSV")

    repair_df, repair_audit = build_repair_telemetry(
        events_path=m.events_path,
        debug_summary_path=m.debug_summary_path,
        sample_keys=df[["sample_i", "sample_id"]],
        default_budget_max=repair_budget_max_default,
    )
    repair_df["sample_i"] = pd.to_numeric(repair_df["sample_i"], errors="raise").astype(int)
    repair_df["sample_id"] = repair_df["sample_id"].astype(str)
    # sample_i is the canonical within-seed key. sample_id equality is audited separately because
    # historical exports may differ in CSV dtype formatting.
    sid = df[["sample_i", "sample_id"]].merge(
        repair_df[["sample_i", "sample_id"]], on="sample_i", how="left", suffixes=("_act", "_repair"), validate="one_to_one"
    )
    sid_mismatch = int((sid["sample_id_act"] != sid["sample_id_repair"]).sum())
    repair_audit["sample_id_mismatch_count"] = sid_mismatch
    if sid_mismatch != 0:
        raise RuntimeError(
            f"seed={m.seed}: sample_id mismatch between activation and repair telemetry rows: {sid_mismatch}"
        )
    repair_values = repair_df.drop(columns=["sample_id"])
    df = df.merge(repair_values, on="sample_i", how="left", validate="one_to_one")
    missing_repair = [c for c in REPAIR_FEATURES if c not in df.columns]
    if missing_repair:
        raise RuntimeError(f"seed={m.seed}: missing repair features after merge: {missing_repair[:10]}")
    if df[REPAIR_FEATURES].isna().any().any():
        bad = df[REPAIR_FEATURES].columns[df[REPAIR_FEATURES].isna().any()].tolist()
        raise RuntimeError(f"seed={m.seed}: NaN repair features after merge: {bad[:20]}")
    df.insert(0, "seed", str(m.seed))
    df.insert(1, "source_dir", str(m.out_dir))
    repair_audit["seed"] = str(m.seed)
    repair_audit["merged_rows"] = int(len(df))
    return df, repair_audit


def as01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).clip(0, 1).astype("int8")


def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    ga = as01(df["label__gate_accept"])
    gr = as01(df["label__gate_reject"])
    nf = as01(df["label__repaired_output_nonfinite"])
    illegal = as01(df["label__repaired_output_illegal_or_missing"])
    finite = ((nf == 0) & (illegal == 0)).astype("int8")
    safe1 = as01(df["label__post_repair_accepted_safe_1p0"])
    df[VALIDITY] = ((gr == 1) | (nf == 1) | (illegal == 1)).astype("int8")
    df[ELIGIBLE] = ((finite == 1) & (ga == 1)).astype("int8")
    df[TARGET] = ((df[ELIGIBLE] == 1) & (safe1 == 0)).astype("int8")
    return df


def make_hgb(stage: str, seed: int) -> HistGradientBoostingClassifier:
    # Internal package smoke tests may set UNIAD_V446_FAST_SMOKE=1. Formal scripts never set it.
    fast = str(os.environ.get("UNIAD_V446_FAST_SMOKE", "0")).strip() == "1"
    if stage == "c1":
        return HistGradientBoostingClassifier(
            max_iter=12 if fast else 220,
            learning_rate=0.08 if fast else 0.055,
            max_leaf_nodes=7 if fast else 31,
            min_samples_leaf=5 if fast else 20,
            l2_regularization=0.05,
            early_stopping=False if fast else True,
            validation_fraction=0.15,
            random_state=seed,
        )
    if stage == "c2_finite":
        return HistGradientBoostingClassifier(
            max_iter=10 if fast else 180,
            learning_rate=0.08 if fast else 0.06,
            max_leaf_nodes=7 if fast else 15,
            min_samples_leaf=5 if fast else 20,
            l2_regularization=0.12,
            early_stopping=False if fast else True,
            validation_fraction=0.15,
            random_state=seed,
        )
    if stage == "c2_nonfinite":
        return HistGradientBoostingClassifier(
            max_iter=8 if fast else 140,
            learning_rate=0.08 if fast else 0.05,
            max_leaf_nodes=5 if fast else 7,
            min_samples_leaf=4 if fast else 25,
            l2_regularization=0.25,
            early_stopping=False if fast else True,
            validation_fraction=0.15,
            random_state=seed,
        )
    raise ValueError(stage)


def make_pipeline(stage: str, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("clipper", SafeFiniteClipper(abs_cap=1_000_000.0)),
            ("imputer", SafeMedianImputer(fill_value=0.0, abs_cap=1_000_000.0)),
            ("model", make_hgb(stage, seed)),
        ]
    )


def fit_model(
    stage: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Pipeline:
    if len(np.unique(y_train)) < 2:
        raise RuntimeError(f"{stage}: single-class training data")
    pipe = make_pipeline(stage, seed)
    if sample_weight is None:
        pipe.fit(X_train, y_train)
    else:
        pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
    return pipe


def predict_score(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X)[:, 1].astype(np.float64)


def safe_auc(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    try:
        return float(roc_auc_score(y, score)) if len(np.unique(y)) >= 2 else None
    except Exception:
        return None


def safe_ap(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    try:
        return float(average_precision_score(y, score)) if len(np.unique(y)) >= 2 else None
    except Exception:
        return None


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return np.full(len(values), 0.5, dtype=np.float64)
    ref = np.sort(ref)
    vals = np.asarray(values, dtype=np.float64)
    return np.searchsorted(ref, vals, side="right").astype(np.float64) / float(len(ref))


def robust_z(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    vals = np.asarray(values, dtype=np.float64)
    if len(ref) == 0:
        return np.zeros_like(vals, dtype=np.float64)
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(ref))
    if not math.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return np.clip((vals - med) / scale, -20.0, 20.0)


def threshold_grid(scores: np.ndarray, max_points: int = 180) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.array([0.5], dtype=np.float64)
    uniq = np.unique(s)
    if len(uniq) <= max_points:
        grid = uniq
    else:
        grid = np.unique(np.quantile(s, np.linspace(0.0, 1.0, max_points)))
    return np.unique(np.concatenate(([0.0, 1e-12], grid, [1.0, 1.0 + 1e-12]))).astype(np.float64)


def threshold_for_target_recall(y: np.ndarray, score: np.ndarray, target: float) -> float:
    y = np.asarray(y, dtype=int)
    grid = threshold_grid(score, max_points=1200)
    best_t = float(np.min(grid))
    best_far = float("inf")
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    for t in grid:
        pred = score >= t
        recall = float(np.sum(pred & (y == 1)) / pos)
        far = float(np.sum(pred & (y == 0)) / neg)
        if recall >= target and far < best_far:
            best_t = float(t)
            best_far = far
    return best_t


def route_mask_for_c2(
    mode: str,
    y_train: np.ndarray,
    train_scores: np.ndarray,
    min_pool_rows: int,
) -> Tuple[np.ndarray, float, str]:
    def base(m: str) -> Tuple[np.ndarray, float, str]:
        if m.startswith("recall"):
            key = m.replace("recall", "")
            mapping = {"99": 0.99, "995": 0.995, "98": 0.98}
            target = mapping.get(key, float("0." + key) if key.isdigit() else 0.99)
            cut = threshold_for_target_recall(y_train, train_scores, target)
            return train_scores >= cut, cut, f"inner_c1_threshold_for_recall_{target}"
        if m.startswith("quantile"):
            q = float(m.replace("quantile", "")) / 100.0
            cut = float(np.quantile(train_scores, q))
            return train_scores >= cut, cut, f"inner_c1_score_quantile_{q}"
        cut = float(np.quantile(train_scores, 0.45))
        return train_scores >= cut, cut, "inner_c1_score_quantile_0.45_default"

    mask, cut, desc = base(mode)
    routed_y = y_train[mask]
    if int(mask.sum()) < min_pool_rows or len(np.unique(routed_y)) < 2:
        mask, cut, desc2 = base("quantile25")
        desc = desc2 + "__fallback_from_" + mode
        routed_y = y_train[mask]
    if len(np.unique(routed_y)) < 2:
        mask = np.ones_like(y_train, dtype=bool)
        cut = float("nan")
        desc = "all_rows_fallback"
    return mask, cut, desc


def build_c1_meta(
    raw_scores: np.ndarray,
    reference_scores: np.ndarray,
    route_cut: float,
) -> pd.DataFrame:
    pct = empirical_percentile(reference_scores, raw_scores)
    ref_logit = safe_logit(reference_scores)
    val_logit = safe_logit(raw_scores)
    z = robust_z(ref_logit, val_logit)
    if math.isfinite(route_cut):
        cut_pct = float(empirical_percentile(reference_scores, np.array([route_cut]))[0])
    else:
        cut_pct = 0.0
    return pd.DataFrame(
        {
            "v446_c1_train_percentile": pct,
            "v446_c1_logit_robust_z": z,
            "v446_c1_top_0p1pct": (pct >= 0.999).astype(np.float64),
            "v446_c1_top_0p5pct": (pct >= 0.995).astype(np.float64),
            "v446_c1_top_1pct": (pct >= 0.99).astype(np.float64),
            "v446_c1_top_5pct": (pct >= 0.95).astype(np.float64),
            "v446_c1_top_10pct": (pct >= 0.90).astype(np.float64),
            "v446_c1_distance_to_route_cut_pct": pct - cut_pct,
            "v446_c1_above_route_cut": (pct >= cut_pct).astype(np.float64),
        }
    )


def select_c2_features(features: List[str], branch: str, finite_mode: str) -> List[str]:
    """Return legal branch-specific activation feature names.

    finite_mode=exclude_nonfinite removes explicit nonfinite and finite-ratio
    summaries from C2-F only. C1 always keeps the complete frozen feature set.
    """
    if branch == BRANCH_NONFINITE:
        return list(features)
    if branch != BRANCH_FINITE:
        raise ValueError(branch)
    if finite_mode == "all":
        return list(features)
    if finite_mode != "exclude_nonfinite":
        raise ValueError(f"Unknown finite branch feature mode: {finite_mode}")
    out = []
    for c in features:
        if c.startswith(REPAIR_PREFIX):
            out.append(c)
            continue
        low = c.lower()
        if "nonfinite" in low or "finite_ratio" in low:
            continue
        out.append(c)
    return out


def branch_name(flag: int) -> str:
    return BRANCH_NONFINITE if int(flag) == 1 else BRANCH_FINITE


def criterion_target_met(metric_row: Dict[str, Any], criterion: str) -> int:
    kind, value = parse_criterion(criterion)
    if kind == "far":
        return int(float(metric_row["far"]) <= value + 1e-15)
    return int(float(metric_row["recall"]) >= value - 1e-15)


def metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(pred, dtype=bool)
    tp = int(np.sum(p & (y == 1)))
    fp = int(np.sum(p & (y == 0)))
    fn = int(np.sum((~p) & (y == 1)))
    tn = int(np.sum((~p) & (y == 0)))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": float(tp / (tp + fn)) if (tp + fn) else None,
        "far": float(fp / (fp + tn)) if (fp + tn) else None,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "pred_pos": int(tp + fp),
    }


def grid_1d_final(
    y_eligible: np.ndarray,
    score: np.ndarray,
    y_all: np.ndarray,
    validity_all: np.ndarray,
    max_grid: int,
    route: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in threshold_grid(score, max_points=max_grid):
        learned = score >= float(t)
        final = np.asarray(validity_all, dtype=bool).copy()
        eligible_positions = np.where(np.asarray(validity_all, dtype=int) == 0)[0]
        if len(eligible_positions) != len(learned):
            raise RuntimeError("grid_1d_final: eligible/validity alignment mismatch")
        final[eligible_positions] |= learned
        m = metrics(y_all, final)
        rows.append({"route": route, "c1_threshold": float(t), "c2_threshold": np.nan, **m})
    return pd.DataFrame(rows)


def _grid_indices(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.searchsorted(thresholds, scores, side="right") - 1


def _reverse_cumsum2d(hist: np.ndarray) -> np.ndarray:
    return hist[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]


def grid_2d_final(
    y_eligible: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    y_all: np.ndarray,
    validity_all: np.ndarray,
    max_grid: int,
    route: str,
) -> pd.DataFrame:
    """Efficient C1 x C2 threshold grid using a 2-D score histogram."""
    t1 = threshold_grid(c1, max_points=max_grid)
    t2 = threshold_grid(c2, max_points=max_grid)
    y_e = np.asarray(y_eligible, dtype=int)
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    if not (len(y_e) == len(c1) == len(c2)):
        raise RuntimeError("grid_2d_final: eligible score length mismatch")

    i1 = _grid_indices(c1, t1)
    i2 = _grid_indices(c2, t2)
    valid = (i1 >= 0) & (i2 >= 0)
    hist_pos = np.zeros((len(t1), len(t2)), dtype=np.int64)
    hist_neg = np.zeros((len(t1), len(t2)), dtype=np.int64)
    np.add.at(hist_pos, (i1[valid & (y_e == 1)], i2[valid & (y_e == 1)]), 1)
    np.add.at(hist_neg, (i1[valid & (y_e == 0)], i2[valid & (y_e == 0)]), 1)
    tp_grid = _reverse_cumsum2d(hist_pos)
    fp_grid = _reverse_cumsum2d(hist_neg)

    y_all = np.asarray(y_all, dtype=int)
    validity = np.asarray(validity_all, dtype=bool)
    base_tp = int(np.sum(validity & (y_all == 1)))
    base_fp = int(np.sum(validity & (y_all == 0)))
    pos_all = int(np.sum(y_all == 1))
    neg_all = int(np.sum(y_all == 0))

    rows: List[Dict[str, Any]] = []
    for a, thr1 in enumerate(t1):
        for b, thr2 in enumerate(t2):
            tp = base_tp + int(tp_grid[a, b])
            fp = base_fp + int(fp_grid[a, b])
            fn = pos_all - tp
            tn = neg_all - fp
            rows.append(
                {
                    "route": route,
                    "c1_threshold": float(thr1),
                    "c2_threshold": float(thr2),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "recall": float(tp / pos_all) if pos_all else None,
                    "far": float(fp / neg_all) if neg_all else None,
                    "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
                    "pred_pos": int(tp + fp),
                }
            )
    return pd.DataFrame(rows)

def choose_branch_3d_points(
    y_eligible: np.ndarray,
    branch_flag: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    y_all: np.ndarray,
    validity_all: np.ndarray,
    max_grid: int,
    route: str,
    criteria: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Select C1, C2-F and C2-NF thresholds without materializing a 3-D table.

    The two C2 branches are disjoint. For each C1 threshold, branch-local
    C1xC2 reverse-cumulative histograms are added, then each requested
    operating criterion is updated. Thresholds are selected only from the
    caller's calibration data.
    """
    y_e = np.asarray(y_eligible, dtype=int)
    z = np.asarray(branch_flag, dtype=int)
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    if not (len(y_e) == len(z) == len(c1) == len(c2)):
        raise RuntimeError("choose_branch_3d_points: eligible arrays mismatch")
    if np.any(~np.isfinite(c1)) or np.any(~np.isfinite(c2)):
        raise RuntimeError("choose_branch_3d_points: nonfinite policy scores")

    t1 = threshold_grid(c1, max_points=max_grid)
    branch_tables: Dict[int, Dict[str, Any]] = {}
    for flag in [0, 1]:
        bm = z == flag
        if not np.any(bm):
            raise RuntimeError(f"choose_branch_3d_points: empty branch {flag}")
        t2 = threshold_grid(c2[bm], max_points=max_grid)
        i1 = _grid_indices(c1[bm], t1)
        i2 = _grid_indices(c2[bm], t2)
        valid = (i1 >= 0) & (i2 >= 0)
        hist_pos = np.zeros((len(t1), len(t2)), dtype=np.int64)
        hist_neg = np.zeros((len(t1), len(t2)), dtype=np.int64)
        yb = y_e[bm]
        np.add.at(hist_pos, (i1[valid & (yb == 1)], i2[valid & (yb == 1)]), 1)
        np.add.at(hist_neg, (i1[valid & (yb == 0)], i2[valid & (yb == 0)]), 1)
        branch_tables[flag] = {
            "thresholds": t2,
            "tp": _reverse_cumsum2d(hist_pos),
            "fp": _reverse_cumsum2d(hist_neg),
        }

    y_all = np.asarray(y_all, dtype=int)
    validity = np.asarray(validity_all, dtype=bool)
    base_tp = int(np.sum(validity & (y_all == 1)))
    base_fp = int(np.sum(validity & (y_all == 0)))
    pos_all = int(np.sum(y_all == 1))
    neg_all = int(np.sum(y_all == 0))
    if pos_all <= 0 or neg_all <= 0:
        raise RuntimeError("choose_branch_3d_points: invalid all-row class counts")

    best: Dict[str, Optional[Dict[str, Any]]] = {c: None for c in criteria}
    tf = branch_tables[0]["thresholds"]
    tn = branch_tables[1]["thresholds"]

    for a, thr1 in enumerate(t1):
        tp = (
            base_tp
            + branch_tables[0]["tp"][a, :, None]
            + branch_tables[1]["tp"][a, None, :]
        )
        fp = (
            base_fp
            + branch_tables[0]["fp"][a, :, None]
            + branch_tables[1]["fp"][a, None, :]
        )
        fn = pos_all - tp
        tn_count = neg_all - fp
        recall = tp.astype(np.float64) / float(pos_all)
        far = fp.astype(np.float64) / float(neg_all)
        denom = tp + fp
        precision = np.divide(
            tp.astype(np.float64),
            denom.astype(np.float64),
            out=np.zeros_like(recall, dtype=np.float64),
            where=denom > 0,
        )

        flat_recall = recall.ravel()
        flat_far = far.ravel()
        flat_precision = precision.ravel()
        flat_tp = tp.ravel()
        flat_fp = fp.ravel()
        flat_fn = fn.ravel()
        flat_tn = tn_count.ravel()

        for criterion in criteria:
            kind, value = parse_criterion(criterion)
            if kind == "far":
                feasible_mask = flat_far <= value
                if np.any(feasible_mask):
                    idxs = np.where(feasible_mask)[0]
                    order = np.lexsort(
                        (
                            -flat_precision[idxs],
                            flat_far[idxs],
                            -flat_recall[idxs],
                        )
                    )
                    flat_idx = int(idxs[int(order[0])])
                    feasible = 1
                else:
                    idxs = np.arange(len(flat_far))
                    order = np.lexsort((-flat_recall[idxs], flat_far[idxs]))
                    flat_idx = int(idxs[int(order[0])])
                    feasible = 0
            else:
                feasible_mask = flat_recall >= value
                if np.any(feasible_mask):
                    idxs = np.where(feasible_mask)[0]
                    order = np.lexsort(
                        (
                            -flat_precision[idxs],
                            -flat_recall[idxs],
                            flat_far[idxs],
                        )
                    )
                    flat_idx = int(idxs[int(order[0])])
                    feasible = 1
                else:
                    idxs = np.arange(len(flat_far))
                    order = np.lexsort((flat_far[idxs], -flat_recall[idxs]))
                    flat_idx = int(idxs[int(order[0])])
                    feasible = 0

            i_f, i_nf = np.unravel_index(flat_idx, recall.shape)
            candidate = {
                "route": route,
                "c1_threshold": float(thr1),
                "c2_threshold": np.nan,
                "c2_threshold_finite": float(tf[i_f]),
                "c2_threshold_nonfinite": float(tn[i_nf]),
                "tp": int(flat_tp[flat_idx]),
                "fp": int(flat_fp[flat_idx]),
                "fn": int(flat_fn[flat_idx]),
                "tn": int(flat_tn[flat_idx]),
                "recall": float(flat_recall[flat_idx]),
                "far": float(flat_far[flat_idx]),
                "precision": float(flat_precision[flat_idx]),
                "pred_pos": int(flat_tp[flat_idx] + flat_fp[flat_idx]),
                "criterion": criterion,
                "calibration_feasible": feasible,
            }
            if best[criterion] is None:
                best[criterion] = candidate
            else:
                best[criterion] = better_point(best[criterion], candidate, criterion)

    return {c: dict(best[c]) for c in criteria if best[c] is not None}


def parse_criterion(criterion: str) -> Tuple[str, float]:
    if criterion.startswith("best_recall_at_far_le_"):
        return "far", float(criterion.rsplit("_", 1)[1])
    if criterion.startswith("min_far_at_recall_ge_"):
        return "recall", float(criterion.rsplit("_", 1)[1])
    raise ValueError(criterion)


def choose_point(grid: pd.DataFrame, criterion: str) -> Dict[str, Any]:
    kind, value = parse_criterion(criterion)
    if kind == "far":
        feasible = grid[pd.to_numeric(grid["far"], errors="coerce") <= value]
        if len(feasible):
            row = feasible.sort_values(["recall", "far", "precision"], ascending=[False, True, False]).iloc[0]
            ok = 1
        else:
            row = grid.sort_values(["far", "recall"], ascending=[True, False]).iloc[0]
            ok = 0
    else:
        feasible = grid[pd.to_numeric(grid["recall"], errors="coerce") >= value]
        if len(feasible):
            row = feasible.sort_values(["far", "recall", "precision"], ascending=[True, False, False]).iloc[0]
            ok = 1
        else:
            row = grid.sort_values(["recall", "far"], ascending=[False, True]).iloc[0]
            ok = 0
    out = row.to_dict()
    out["criterion"] = criterion
    out["calibration_feasible"] = ok
    return out


def better_point(a: Dict[str, Any], b: Dict[str, Any], criterion: str) -> Dict[str, Any]:
    kind, _ = parse_criterion(criterion)
    # Feasible calibration candidates always dominate fallback candidates.
    fa = int(a.get("calibration_feasible", 0))
    fb = int(b.get("calibration_feasible", 0))
    if fa != fb:
        return a if fa > fb else b
    if kind == "far":
        ka = (-float(a["recall"]), float(a["far"]), -float(a["precision"]))
        kb = (-float(b["recall"]), float(b["far"]), -float(b["precision"]))
    else:
        ka = (float(a["far"]), -float(a["recall"]), -float(a["precision"]))
        kb = (float(b["far"]), -float(b["recall"]), -float(b["precision"]))
    return a if ka <= kb else b


def safe_part(x: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))[:180]


def save_bundle(
    root: Path,
    stage: str,
    config: str,
    test_seed: str,
    pipe: Pipeline,
    feature_names: List[str],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    d = root / stage
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{safe_part(stage)}__{safe_part(config)}__outer_seed_{safe_part(test_seed)}.joblib"
    payload = {
        "version": VERSION,
        "stage": stage,
        "config": config,
        "outer_test_seed": str(test_seed),
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "pipe": pipe,
        "created_at": now(),
        **extra,
    }
    joblib.dump(payload, p, compress=3)
    return {
        "stage": stage,
        "config": config,
        "outer_test_seed": str(test_seed),
        "feature_count": len(feature_names),
        "bundle_path": str(p),
    }


def build_all_row_fold_arrays(
    df: pd.DataFrame,
    work: pd.DataFrame,
    test_seed: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_mask = df["seed"].astype(str).to_numpy() == str(test_seed)
    all_idx = np.where(all_mask)[0]
    y_all = as01(df.loc[all_mask, OVERALL_LABEL]).to_numpy(int)
    validity = as01(df.loc[all_mask, VALIDITY]).to_numpy(int)
    eligible_all_positions = np.where(validity == 0)[0]
    work_mask = work["seed"].astype(str).to_numpy() == str(test_seed)
    work_idx = np.where(work_mask)[0]
    if len(eligible_all_positions) != len(work_idx):
        raise RuntimeError(
            f"seed={test_seed}: eligible mapping mismatch all={len(eligible_all_positions)} work={len(work_idx)}"
        )
    return all_idx, y_all, validity, eligible_all_positions, work_idx



def run_training(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    requested_seeds = parse_tokens(args.seeds)
    if len(requested_seeds) < 4:
        raise RuntimeError("At least 4 seeds are required for outer/inner seed nesting")

    contract = Path(args.feature_contract_json).resolve() if args.feature_contract_json else None
    mats = validate_seed_materials(data_root, requested_seeds, args.expected_rows_per_seed)
    features = build_frozen_feature_contract(
        mats,
        out_dir,
        contract,
        Path(args.excluded_activation_features_file).resolve()
        if args.excluded_activation_features_file else None,
        int(args.expected_extended_source_feature_count),
        int(args.expected_runtime_summary_feature_count),
        int(args.expected_activation_feature_count),
    )
    if NONFINITE_SOURCE_FEATURE not in features:
        raise RuntimeError(
            f"Required online routing feature is absent from frozen feature contract: "
            f"{NONFINITE_SOURCE_FEATURE}"
        )
    missing_route_seeds = [
        m.seed for m in mats if NONFINITE_SOURCE_FEATURE not in set(m.feature_columns)
    ]
    if missing_route_seeds:
        raise RuntimeError(
            f"Routing feature {NONFINITE_SOURCE_FEATURE} is missing in seeds: "
            + ",".join(missing_route_seeds)
        )

    finite_c2_features = select_c2_features(
        features, BRANCH_FINITE, args.finite_branch_feature_mode
    )
    nonfinite_c2_features = select_c2_features(
        features, BRANCH_NONFINITE, args.finite_branch_feature_mode
    )
    if not finite_c2_features or not nonfinite_c2_features:
        raise RuntimeError("Branch feature selection produced an empty feature set")

    branch_contract_rows: List[Dict[str, Any]] = []
    finite_set = set(finite_c2_features)
    nf_set = set(nonfinite_c2_features)
    for c in features:
        branch_contract_rows.append(
            {
                "feature": c,
                "c1_used": 1,
                "c2_finite_used": int(c in finite_set),
                "c2_nonfinite_used": int(c in nf_set),
                "finite_exclusion_reason": (
                    "explicit_nonfinite_or_finite_ratio"
                    if c not in finite_set else ""
                ),
                "online_source": 1,
                "routing_source_feature": int(c == NONFINITE_SOURCE_FEATURE),
            }
        )
    pd.DataFrame(branch_contract_rows).to_csv(
        out_dir / "branch_feature_contract_v4_4_6.csv", index=False
    )

    log(
        f"PRECHECK OK seeds={len(mats)} frozen_features={len(features)} "
        f"c2_finite_features={len(finite_c2_features)} "
        f"c2_nonfinite_features={len(nonfinite_c2_features)}"
    )
    if args.preflight_only:
        preflight_audits: List[Dict[str, Any]] = []
        for i, m in enumerate(mats, 1):
            keys = pd.read_csv(m.csv_path, usecols=["sample_i", "sample_id"], low_memory=False)
            keys["sample_i"] = pd.to_numeric(keys["sample_i"], errors="raise").astype(int)
            keys["sample_id"] = keys["sample_id"].astype(str)
            _, audit = build_repair_telemetry(
                m.events_path, m.debug_summary_path, keys, args.repair_budget_max_default
            )
            audit["seed"] = str(m.seed)
            preflight_audits.append(audit)
            log(
                f"REPAIR TELEMETRY PRECHECK {i}/{len(mats)} seed={m.seed} "
                f"events={audit['event_rows_consumed']} samples_with_event={audit['samples_with_event']}"
            )
        pd.DataFrame(preflight_audits).to_csv(
            out_dir / "repair_telemetry_preflight_audit_v4_4_6.csv", index=False
        )
        write_json(
            out_dir / "repair_telemetry_preflight_audit_v4_4_6.json",
            {"version": VERSION, "seeds": preflight_audits},
        )
        return

    dfs: List[pd.DataFrame] = []
    repair_audits: List[Dict[str, Any]] = []
    for i, m in enumerate(mats, 1):
        log(f"READ {i}/{len(mats)} seed={m.seed} path={m.csv_path}")
        one, audit = read_one(m, features, args.repair_budget_max_default)
        dfs.append(one)
        repair_audits.append(audit)
        log(
            f"REPAIR TELEMETRY seed={m.seed} events={audit['event_rows_consumed']} "
            f"samples_with_event={audit['samples_with_event']} "
            f"samples_with_modified={audit['samples_with_modified']}"
        )
    pd.DataFrame(repair_audits).to_csv(
        out_dir / "repair_telemetry_materialization_audit_v4_4_6.csv", index=False
    )
    write_json(
        out_dir / "repair_telemetry_materialization_audit_v4_4_6.json",
        {"version": VERSION, "seeds": repair_audits},
    )
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df = derive_labels(df)
    missing_after_concat = [c for c in features if c not in df.columns]
    if missing_after_concat:
        raise RuntimeError(
            "Exact contract violation after concatenation; missing features: "
            + ", ".join(missing_after_concat[:20])
        )
    df["seed"] = df["seed"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)
    route_source = pd.to_numeric(
        df[NONFINITE_SOURCE_FEATURE], errors="coerce"
    ).fillna(0.0)
    df[NONFINITE_ROUTE] = (route_source > 0.0).astype("int8")
    df["row_uid"] = df["seed"] + "::" + df["sample_id"]
    df.reset_index(drop=True, inplace=True)
    df["global_all_index"] = np.arange(len(df), dtype=np.int64)

    eligible_mask = as01(df[ELIGIBLE]).to_numpy(bool)
    work_cols = [
        "global_all_index", "seed", "sample_id", "row_uid", NONFINITE_ROUTE
    ] + features + [TARGET]
    work = df.loc[eligible_mask, work_cols].copy()
    work[TARGET] = as01(work[TARGET])
    work[NONFINITE_ROUTE] = as01(work[NONFINITE_ROUTE])
    work.reset_index(drop=True, inplace=True)
    work["global_work_index"] = np.arange(len(work), dtype=np.int64)

    y = work[TARGET].to_numpy(int)
    z = work[NONFINITE_ROUTE].to_numpy(int)
    work_seeds = work["seed"].astype(str).to_numpy()
    unique_seeds = sorted(work["seed"].astype(str).unique())
    X_all = work[features]

    # Offline-only audit of how the online repair telemetry is distributed by seed and target.
    # TARGET is used only to summarize strata; it is never used to construct X.
    telemetry_flags = {
        "any_event": REPAIR_PREFIX + "any_event",
        "any_modified": REPAIR_PREFIX + "any_modified",
        "budget_saturated": REPAIR_PREFIX + "budget_saturated",
        "nonfinite_event_seen": REPAIR_PREFIX + "nonfinite_event_count",
        "huge_event_seen": REPAIR_PREFIX + "huge_event_count",
        "planning_modified_seen": REPAIR_PREFIX + "group_u7_planning_head_modified_count",
        "motion_modified_seen": REPAIR_PREFIX + "group_u5_motion_head_modified_count",
        "clip_median_seen": REPAIR_PREFIX + "action_clip_median_count",
        "zero_tensor_seen": REPAIR_PREFIX + "action_zero_tensor_count",
        "input_passthrough_seen": REPAIR_PREFIX + "action_input_passthrough_count",
    }
    strata_rows: List[Dict[str, Any]] = []
    for scope_seed in ["ALL"] + unique_seeds:
        scope = np.ones(len(work), dtype=bool) if scope_seed == "ALL" else (work_seeds == scope_seed)
        for flag_name, col in telemetry_flags.items():
            flag = pd.to_numeric(work[col], errors="coerce").fillna(0.0).to_numpy() > 0
            for value in [0, 1]:
                m = scope & (flag == bool(value))
                if not np.any(m):
                    continue
                pos = int(np.sum(y[m] == 1))
                rows_n = int(np.sum(m))
                strata_rows.append({
                    "seed": scope_seed,
                    "telemetry_flag": flag_name,
                    "flag_value": value,
                    "rows": rows_n,
                    "positive": pos,
                    "negative": rows_n - pos,
                    "positive_rate": float(pos / rows_n) if rows_n else 0.0,
                })
    pd.DataFrame(strata_rows).to_csv(
        out_dir / "repair_telemetry_target_strata_v4_4_6.csv", index=False
    )

    branch_counts = (
        work.groupby(["seed", NONFINITE_ROUTE, TARGET], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    branch_counts["branch"] = branch_counts[NONFINITE_ROUTE].map(
        {0: BRANCH_FINITE, 1: BRANCH_NONFINITE}
    )
    branch_counts.to_csv(out_dir / "nonfinite_branch_counts_v4_4_6.csv", index=False)

    overall_branch_counts = (
        work.groupby([NONFINITE_ROUTE, TARGET], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    log("===== ELIGIBLE NONFINITE ROUTE COUNTS =====")
    print(overall_branch_counts.to_string(index=False), flush=True)

    for branch_flag in [0, 1]:
        bm = z == branch_flag
        pos = int(np.sum(y[bm] == 1))
        neg = int(np.sum(y[bm] == 0))
        if pos < args.c2_min_total_branch_positives:
            raise RuntimeError(
                f"branch={branch_name(branch_flag)} has only {pos} positives; "
                f"minimum required={args.c2_min_total_branch_positives}"
            )
        log(
            f"BRANCH {branch_name(branch_flag)} rows={int(bm.sum())} "
            f"pos={pos} neg={neg}"
        )

    far_budgets = [float(x) for x in parse_tokens(args.far_budgets)]
    recall_targets = [float(x) for x in parse_tokens(args.recall_targets)]
    criteria = [f"best_recall_at_far_le_{b:g}" for b in far_budgets] + [
        f"min_far_at_recall_ge_{r:g}" for r in recall_targets
    ]
    c2_modes = parse_tokens(args.c2_route_modes)

    systems = [
        "strict_repair_c1_only",
        "strict_repair_branch_c1c2_best",
        "strict_repair_selected_best",
    ]
    pred_store: Dict[Tuple[str, str], np.ndarray] = {}
    for system in systems:
        for criterion in criteria:
            pred_store[(system, criterion)] = np.zeros(len(df), dtype=bool)

    outer_rows: List[Dict[str, Any]] = []
    outer_branch_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    route_audit_rows: List[Dict[str, Any]] = []
    bundle_rows: List[Dict[str, Any]] = []
    score_rows: List[pd.DataFrame] = []
    bundle_root = out_dir / "model_bundles"

    write_json(
        out_dir / "run_config_v4_4_6.json",
        {
            "version": VERSION,
            "target": TARGET,
            "eligible": ELIGIBLE,
            "validity_alarm": VALIDITY,
            "overall_label": OVERALL_LABEL,
            "seeds": unique_seeds,
            "feature_count": len(features),
            "activation_feature_count": len([c for c in features if not c.startswith(REPAIR_PREFIX)]),
            "repair_telemetry_feature_count": len([c for c in features if c.startswith(REPAIR_PREFIX)]),
            "feature_sha256": sha256_text(features),
            "nonfinite_route": NONFINITE_ROUTE,
            "nonfinite_source_feature": NONFINITE_SOURCE_FEATURE,
            "nonfinite_route_definition": f"{NONFINITE_SOURCE_FEATURE} > 0",
            "nonfinite_is_routing_not_target": True,
            "c1": "one HGB model using activation plus allowlisted online repair telemetry, all eligible rows, unweighted",
            "c2_finite": {
                "model": "HGB finite-only branch",
                "feature_mode": args.finite_branch_feature_mode,
                "feature_count": len(finite_c2_features),
                "negative_weight": args.c2_finite_neg_weight,
            },
            "c2_nonfinite": {
                "model": "shallower HGB nonfinite-seen branch",
                "feature_count": len(nonfinite_c2_features),
                "negative_weight": args.c2_nonfinite_neg_weight,
            },
            "c2_route_modes": c2_modes,
            "method": (
                "outer-seed holdout; inner seed-OOF repair-aware C1; separate inner seed-OOF "
                "repair-aware C2-F/C2-NF; branch-local score calibration; fold-local policy selection"
            ),
            "c1_policy_score": (
                "empirical percentile against outer-training inner-OOF C1 reference"
            ),
            "c2_policy_score": (
                "branch-local empirical percentile against each branch C2-training "
                "prediction reference; merged after calibration"
            ),
            "final_alarm": "validity_alarm OR learned_alarm",
            "learned_alarm": (
                "C1 threshold AND branch-specific C2-F/C2-NF thresholds; "
                "one C1, two routed C2 models"
            ),
            "allowed_repair_x": (
                "current-inference repair action/location/scope/budget/pre-post numerical recovery "
                "aggregated through an exact source-field allowlist"
            ),
            "forbidden_x": [
                "clean reference", "GT/L2/collision", "fault-plan/bit/injection metadata",
                "gate decision as X", "post-hoc repair success or safety label",
                "true fault trace order/detection distance",
                "old detector outputs or a second C1",
            ],
        },
    )

    for outer_idx, test_seed in enumerate(unique_seeds):
        outer_train_seeds = [s for s in unique_seeds if s != test_seed]
        tr_mask = np.isin(work_seeds, outer_train_seeds)
        te_mask = work_seeds == test_seed
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]
        y_tr = y[tr_mask]
        y_te = y[te_mask]
        z_tr = z[tr_mask]
        z_te = z[te_mask]
        tr_seeds = work_seeds[tr_mask]

        log(
            f"===== OUTER {outer_idx+1}/{len(unique_seeds)} test_seed={test_seed} "
            f"train_seeds={','.join(outer_train_seeds)} train={len(tr_idx)} "
            f"test={len(te_idx)} ====="
        )
        for branch_flag in [0, 1]:
            btr = z_tr == branch_flag
            bte = z_te == branch_flag
            log(
                f"[OUTER {test_seed}] branch={branch_name(branch_flag)} "
                f"train_rows={int(btr.sum())} train_pos={int(y_tr[btr].sum())} "
                f"test_rows={int(bte.sum())} test_pos={int(y_te[bte].sum())}"
            )

        # C1 inner seed-OOF scores for outer training only.
        inner_c1_raw = np.full(len(tr_idx), np.nan, dtype=np.float64)
        for inner_idx, val_seed in enumerate(outer_train_seeds):
            iv = tr_seeds == val_seed
            it = ~iv
            log(
                f"[OUTER {test_seed}] C1 inner {inner_idx+1}/{len(outer_train_seeds)} "
                f"val_seed={val_seed} train={int(it.sum())} val={int(iv.sum())}"
            )
            pipe = fit_model(
                "c1",
                X_all.iloc[tr_idx[it]],
                y_tr[it],
                args.random_seed + outer_idx * 100 + inner_idx,
            )
            inner_c1_raw[iv] = predict_score(pipe, X_all.iloc[tr_idx[iv]])
            del pipe
            gc.collect()
        if np.any(~np.isfinite(inner_c1_raw)):
            raise RuntimeError(f"outer={test_seed}: nonfinite inner C1 OOF scores")

        # Final outer C1 model and held-out-seed score.
        outer_c1 = fit_model(
            "c1",
            X_all.iloc[tr_idx],
            y_tr,
            args.random_seed + 10_000 + outer_idx,
        )
        outer_test_c1_raw = predict_score(outer_c1, X_all.iloc[te_idx])
        c1_train_policy = empirical_percentile(inner_c1_raw, inner_c1_raw)
        c1_test_policy = empirical_percentile(inner_c1_raw, outer_test_c1_raw)
        if args.save_models:
            bundle_rows.append(
                save_bundle(
                    bundle_root,
                    "C1",
                    "hgb_full_nested_calibrated",
                    test_seed,
                    outer_c1,
                    features,
                    {
                        "inner_oof_reference_scores_sorted": np.sort(inner_c1_raw),
                        "policy_score": (
                            "empirical percentile against inner_oof_reference_scores_sorted"
                        ),
                        "outer_train_seeds": outer_train_seeds,
                        "nonfinite_route_source_feature": NONFINITE_SOURCE_FEATURE,
                    },
                )
            )

        test_all_idx, y_test_all, validity_test, eligible_test_positions, work_test_idx = (
            build_all_row_fold_arrays(df, work, test_seed)
        )
        if not np.array_equal(work_test_idx, te_idx):
            raise RuntimeError(f"outer={test_seed}: work test index mismatch")

        train_all_mask = df["seed"].astype(str).isin(outer_train_seeds).to_numpy()
        y_train_all = as01(df.loc[train_all_mask, OVERALL_LABEL]).to_numpy(int)
        validity_train = as01(df.loc[train_all_mask, VALIDITY]).to_numpy(int)
        train_eligible_positions = np.where(validity_train == 0)[0]
        if len(train_eligible_positions) != len(tr_idx):
            raise RuntimeError(f"outer={test_seed}: outer train eligible alignment mismatch")

        c1_grid = grid_1d_final(
            y_tr,
            c1_train_policy,
            y_train_all,
            validity_train,
            args.policy_grid_max,
            "C1_only",
        )
        c1_points = {c: choose_point(c1_grid, c) for c in criteria}

        route_outputs: Dict[str, Dict[str, Any]] = {}
        for mode_idx, mode in enumerate(c2_modes):
            route_name = f"C1_plus_branch_C2_veto__{mode}"
            log(
                f"[OUTER {test_seed}] ===== BRANCH C2 route={mode} "
                f"calibration_folds={len(outer_train_seeds)} ====="
            )
            c2_oof_policy = np.full(len(tr_idx), np.nan, dtype=np.float64)

            # Branch-specific C2 seed-OOF calibration entirely inside outer training.
            for c2_fold, val_seed in enumerate(outer_train_seeds):
                cv = tr_seeds == val_seed
                ct = ~cv
                ct_positions = np.where(ct)[0]
                cv_positions = np.where(cv)[0]

                for branch_flag in [0, 1]:
                    branch = branch_name(branch_flag)
                    stage = "c2_nonfinite" if branch_flag == 1 else "c2_finite"
                    branch_features = (
                        nonfinite_c2_features if branch_flag == 1 else finite_c2_features
                    )
                    min_pool = (
                        args.c2_min_pool_rows_nonfinite
                        if branch_flag == 1
                        else args.c2_min_pool_rows_finite
                    )
                    neg_weight = (
                        args.c2_nonfinite_neg_weight
                        if branch_flag == 1
                        else args.c2_finite_neg_weight
                    )

                    ct_branch = z_tr[ct] == branch_flag
                    cv_branch = z_tr[cv] == branch_flag
                    ct_b_positions = ct_positions[ct_branch]
                    cv_b_positions = cv_positions[cv_branch]
                    y_ct_b = y_tr[ct][ct_branch]
                    c1_ct_b = inner_c1_raw[ct][ct_branch]
                    c1_cv_b = inner_c1_raw[cv][cv_branch]

                    if len(cv_b_positions) == 0:
                        continue
                    if len(y_ct_b) == 0 or len(np.unique(y_ct_b)) < 2:
                        raise RuntimeError(
                            f"outer={test_seed} val={val_seed} branch={branch}: "
                            "single-class branch training data"
                        )
                    if int(np.sum(y_ct_b == 1)) < args.c2_min_inner_branch_positives:
                        raise RuntimeError(
                            f"outer={test_seed} val={val_seed} branch={branch}: "
                            f"positives={int(np.sum(y_ct_b == 1))} below minimum "
                            f"{args.c2_min_inner_branch_positives}"
                        )

                    route_mask, route_cut, route_desc = route_mask_for_c2(
                        mode, y_ct_b, c1_ct_b, min_pool
                    )
                    meta_ct = build_c1_meta(c1_ct_b, c1_ct_b, route_cut)
                    meta_cv = build_c1_meta(c1_cv_b, c1_ct_b, route_cut)
                    X2_ct_all = pd.concat(
                        [
                            X_all.iloc[tr_idx[ct_b_positions]][branch_features]
                            .reset_index(drop=True),
                            meta_ct.reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    X2_cv = pd.concat(
                        [
                            X_all.iloc[tr_idx[cv_b_positions]][branch_features]
                            .reset_index(drop=True),
                            meta_cv.reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    routed_y = y_ct_b[route_mask]
                    sw = np.where(routed_y == 0, float(neg_weight), 1.0).astype(
                        np.float64
                    )
                    log(
                        f"[OUTER {test_seed}] C2 {mode} branch={branch} "
                        f"cv {c2_fold+1}/{len(outer_train_seeds)} val_seed={val_seed} "
                        f"branch_train={len(y_ct_b)} branch_pos={int(y_ct_b.sum())} "
                        f"branch_val={len(cv_b_positions)} route_rows={int(route_mask.sum())} "
                        f"route_pos={int(routed_y.sum())} "
                        f"route_neg={int((routed_y==0).sum())} cut={route_cut:.6g} "
                        f"desc={route_desc} features={len(branch_features)}"
                    )
                    pipe2 = fit_model(
                        stage,
                        X2_ct_all.iloc[np.where(route_mask)[0]],
                        routed_y,
                        args.random_seed
                        + 20_000
                        + outer_idx * 2000
                        + mode_idx * 200
                        + c2_fold * 2
                        + branch_flag,
                        sample_weight=sw,
                    )
                    train_raw2 = predict_score(pipe2, X2_ct_all)
                    val_raw2 = predict_score(pipe2, X2_cv)
                    c2_oof_policy[cv_b_positions] = empirical_percentile(
                        train_raw2, val_raw2
                    )
                    route_audit_rows.append(
                        {
                            "outer_test_seed": test_seed,
                            "route_mode": mode,
                            "phase": "inner_c2_calibration",
                            "validation_seed": val_seed,
                            "branch": branch,
                            "branch_train_rows": len(y_ct_b),
                            "branch_train_positive": int(y_ct_b.sum()),
                            "branch_validation_rows": len(cv_b_positions),
                            "route_rows": int(route_mask.sum()),
                            "route_positive": int(routed_y.sum()),
                            "route_negative": int(np.sum(routed_y == 0)),
                            "route_cut_raw": route_cut,
                            "route_desc": route_desc,
                            "feature_count": len(branch_features),
                            "negative_weight": neg_weight,
                        }
                    )
                    del pipe2, X2_ct_all, X2_cv
                    gc.collect()

            if np.any(~np.isfinite(c2_oof_policy)):
                bad = int(np.sum(~np.isfinite(c2_oof_policy)))
                raise RuntimeError(
                    f"outer={test_seed} route={mode}: {bad} nonfinite branch C2 OOF scores"
                )

            # Final branch C2 models for this outer fold.
            c2_test_policy = np.full(len(te_idx), np.nan, dtype=np.float64)
            branch_model_meta: Dict[str, Dict[str, Any]] = {}
            for branch_flag in [0, 1]:
                branch = branch_name(branch_flag)
                stage = "c2_nonfinite" if branch_flag == 1 else "c2_finite"
                branch_features = (
                    nonfinite_c2_features if branch_flag == 1 else finite_c2_features
                )
                min_pool = (
                    args.c2_min_pool_rows_nonfinite
                    if branch_flag == 1
                    else args.c2_min_pool_rows_finite
                )
                neg_weight = (
                    args.c2_nonfinite_neg_weight
                    if branch_flag == 1
                    else args.c2_finite_neg_weight
                )
                tr_branch = z_tr == branch_flag
                te_branch = z_te == branch_flag
                tr_b_positions = np.where(tr_branch)[0]
                te_b_positions = np.where(te_branch)[0]
                y_tr_b = y_tr[tr_branch]
                c1_tr_b = inner_c1_raw[tr_branch]
                c1_te_b = outer_test_c1_raw[te_branch]

                if len(np.unique(y_tr_b)) < 2:
                    raise RuntimeError(
                        f"outer={test_seed} branch={branch}: single-class final C2 data"
                    )
                route_mask, route_cut, route_desc = route_mask_for_c2(
                    mode, y_tr_b, c1_tr_b, min_pool
                )
                meta_train = build_c1_meta(c1_tr_b, c1_tr_b, route_cut)
                meta_test = build_c1_meta(c1_te_b, c1_tr_b, route_cut)
                X2_train_all = pd.concat(
                    [
                        X_all.iloc[tr_idx[tr_b_positions]][branch_features]
                        .reset_index(drop=True),
                        meta_train.reset_index(drop=True),
                    ],
                    axis=1,
                )
                X2_test = pd.concat(
                    [
                        X_all.iloc[te_idx[te_b_positions]][branch_features]
                        .reset_index(drop=True),
                        meta_test.reset_index(drop=True),
                    ],
                    axis=1,
                )
                routed_y = y_tr_b[route_mask]
                sw = np.where(routed_y == 0, float(neg_weight), 1.0).astype(
                    np.float64
                )
                final_c2 = fit_model(
                    stage,
                    X2_train_all.iloc[np.where(route_mask)[0]],
                    routed_y,
                    args.random_seed
                    + 30_000
                    + outer_idx * 200
                    + mode_idx * 10
                    + branch_flag,
                    sample_weight=sw,
                )
                train_raw2 = predict_score(final_c2, X2_train_all)
                test_raw2 = predict_score(final_c2, X2_test)
                c2_test_policy[te_b_positions] = empirical_percentile(
                    train_raw2, test_raw2
                )
                branch_model_meta[branch] = {
                    "route_cut": route_cut,
                    "route_desc": route_desc,
                    "train_rows": len(y_tr_b),
                    "train_positive": int(y_tr_b.sum()),
                    "test_rows": len(te_b_positions),
                    "feature_count": len(branch_features),
                }
                route_audit_rows.append(
                    {
                        "outer_test_seed": test_seed,
                        "route_mode": mode,
                        "phase": "final_outer_model",
                        "validation_seed": "",
                        "branch": branch,
                        "branch_train_rows": len(y_tr_b),
                        "branch_train_positive": int(y_tr_b.sum()),
                        "branch_validation_rows": len(te_b_positions),
                        "route_rows": int(route_mask.sum()),
                        "route_positive": int(routed_y.sum()),
                        "route_negative": int(np.sum(routed_y == 0)),
                        "route_cut_raw": route_cut,
                        "route_desc": route_desc,
                        "feature_count": len(branch_features),
                        "negative_weight": neg_weight,
                    }
                )
                if args.save_models:
                    bundle_rows.append(
                        save_bundle(
                            bundle_root,
                            "C2_NF" if branch_flag == 1 else "C2_F",
                            f"hgb_branch__{mode}",
                            test_seed,
                            final_c2,
                            branch_features + C1_META_FEATURES,
                            {
                                "branch": branch,
                                "branch_rule": f"{NONFINITE_SOURCE_FEATURE} > 0",
                                "outer_train_seeds": outer_train_seeds,
                                "c1_reference_scores_sorted": np.sort(c1_tr_b),
                                "route_cut_raw": (
                                    route_cut if math.isfinite(route_cut) else None
                                ),
                                "route_desc": route_desc,
                                "c2_output_reference_scores_sorted": np.sort(train_raw2),
                                "c1_meta_features": C1_META_FEATURES,
                                "policy_score": (
                                    "branch-local empirical percentile against "
                                    "c2_output_reference_scores_sorted"
                                ),
                                "negative_weight": neg_weight,
                                "finite_branch_feature_mode": (
                                    args.finite_branch_feature_mode
                                    if branch_flag == 0 else "not_applicable"
                                ),
                            },
                        )
                    )
                del final_c2, X2_train_all, X2_test
                gc.collect()

            if np.any(~np.isfinite(c2_test_policy)):
                bad = int(np.sum(~np.isfinite(c2_test_policy)))
                raise RuntimeError(
                    f"outer={test_seed} route={mode}: {bad} nonfinite branch C2 test scores"
                )

            route_points = choose_branch_3d_points(
                y_tr,
                z_tr,
                c1_train_policy,
                c2_oof_policy,
                y_train_all,
                validity_train,
                args.policy_grid_max,
                route_name,
                criteria,
            )
            route_outputs[mode] = {
                "route_name": route_name,
                "points": route_points,
                "c2_test_policy": c2_test_policy,
                "c2_oof_policy": c2_oof_policy,
                "branch_model_meta": branch_model_meta,
            }

        # Score audit for this held-out seed.
        score_frame = pd.DataFrame(
            {
                "global_all_index": df.loc[test_all_idx, "global_all_index"].to_numpy(),
                "seed": test_seed,
                "sample_id": df.loc[test_all_idx, "sample_id"].astype(str).to_numpy(),
                "y_true_final": y_test_all,
                "validity_alarm": validity_test,
                "eligible": (validity_test == 0).astype(int),
                "internal_nonfinite_seen": as01(
                    df.loc[test_all_idx, NONFINITE_ROUTE]
                ).to_numpy(int),
                "c1_raw_score": np.nan,
                "c1_policy_percentile": np.nan,
            }
        )
        score_frame.loc[eligible_test_positions, "c1_raw_score"] = outer_test_c1_raw
        score_frame.loc[
            eligible_test_positions, "c1_policy_percentile"
        ] = c1_test_policy
        for mode, info in route_outputs.items():
            col = f"branch_c2_policy_percentile__{mode}"
            score_frame[col] = np.nan
            score_frame.loc[eligible_test_positions, col] = info["c2_test_policy"]
        score_rows.append(score_frame)

        # Fold-local policy selection and outer-test application.
        for criterion in criteria:
            c1p = dict(c1_points[criterion])
            c1p.update({"system": "strict_repair_c1_only", "selected_route": "C1_only"})

            best_c2: Optional[Dict[str, Any]] = None
            best_mode: Optional[str] = None
            for mode, info in route_outputs.items():
                p = dict(info["points"][criterion])
                p.update(
                    {
                        "system": "strict_repair_branch_c1c2_best",
                        "selected_route": info["route_name"],
                        "mode": mode,
                    }
                )
                candidate_rows.append(
                    {
                        "outer_test_seed": test_seed,
                        "criterion": criterion,
                        "candidate_system": "C1+branch_C2",
                        "candidate_route": info["route_name"],
                        **{
                            f"cal_{k}": v
                            for k, v in p.items()
                            if k not in {"system", "selected_route", "mode"}
                        },
                    }
                )
                if best_c2 is None:
                    best_c2, best_mode = p, mode
                else:
                    chosen = better_point(best_c2, p, criterion)
                    if chosen is p:
                        best_c2, best_mode = p, mode
            assert best_c2 is not None and best_mode is not None

            selected = better_point(c1p, best_c2, criterion)
            selected_system = (
                "C1_only"
                if selected is c1p
                else route_outputs[best_mode]["route_name"]
            )

            pred_c1 = validity_test.astype(bool).copy()
            learned_c1 = c1_test_policy >= float(c1p["c1_threshold"])
            pred_c1[eligible_test_positions] |= learned_c1
            pred_store[("strict_repair_c1_only", criterion)][test_all_idx] = pred_c1

            best_info = route_outputs[best_mode]
            pred_c2 = validity_test.astype(bool).copy()
            branch_c2_threshold = np.where(
                z_te == 1,
                float(best_c2["c2_threshold_nonfinite"]),
                float(best_c2["c2_threshold_finite"]),
            )
            learned_c2 = (
                (c1_test_policy >= float(best_c2["c1_threshold"]))
                & (best_info["c2_test_policy"] >= branch_c2_threshold)
            )
            pred_c2[eligible_test_positions] |= learned_c2
            pred_store[("strict_repair_branch_c1c2_best", criterion)][test_all_idx] = pred_c2

            pred_sel = pred_c1 if selected is c1p else pred_c2
            learned_sel = learned_c1 if selected is c1p else learned_c2
            pred_store[("strict_repair_selected_best", criterion)][test_all_idx] = pred_sel

            m_c1_test = metrics(y_test_all, pred_c1)
            m_c2_test = metrics(y_test_all, pred_c2)
            m_sel_test = metrics(y_test_all, pred_sel)

            for system_name, point, test_m, route_label in [
                ("strict_repair_c1_only", c1p, m_c1_test, "C1_only"),
                (
                    "strict_repair_branch_c1c2_best",
                    best_c2,
                    m_c2_test,
                    best_info["route_name"],
                ),
                (
                    "strict_repair_selected_best",
                    selected,
                    m_sel_test,
                    selected_system,
                ),
            ]:
                outer_rows.append(
                    {
                        "outer_test_seed": test_seed,
                        "criterion": criterion,
                        "system": system_name,
                        "selected_route": route_label,
                        "calibration_feasible": int(
                            point.get("calibration_feasible", 0)
                        ),
                        "cal_recall": point.get("recall"),
                        "cal_far": point.get("far"),
                        "cal_precision": point.get("precision"),
                        "c1_threshold": point.get("c1_threshold"),
                        "c2_threshold": point.get("c2_threshold"),
                        "c2_threshold_finite": point.get("c2_threshold_finite"),
                        "c2_threshold_nonfinite": point.get("c2_threshold_nonfinite"),
                        **{f"test_{k}": v for k, v in test_m.items()},
                    }
                )

            for system_name, learned_pred, route_label in [
                ("strict_repair_c1_only", learned_c1, "C1_only"),
                (
                    "strict_repair_branch_c1c2_best",
                    learned_c2,
                    best_info["route_name"],
                ),
                ("strict_repair_selected_best", learned_sel, selected_system),
            ]:
                for branch_flag in [0, 1]:
                    bm = z_te == branch_flag
                    bm_metrics = metrics(y_te[bm], learned_pred[bm])
                    outer_branch_rows.append(
                        {
                            "outer_test_seed": test_seed,
                            "criterion": criterion,
                            "system": system_name,
                            "selected_route": route_label,
                            "branch": branch_name(branch_flag),
                            "branch_rows": int(bm.sum()),
                            "branch_positive": int(y_te[bm].sum()),
                            **bm_metrics,
                        }
                    )

        del outer_c1
        gc.collect()

    # Aggregate held-out binary predictions.
    y_all_global = as01(df[OVERALL_LABEL]).to_numpy(int)
    summary_rows: List[Dict[str, Any]] = []
    prediction_cols: Dict[str, Any] = {
        "global_all_index": df["global_all_index"].to_numpy(),
        "seed": df["seed"].astype(str).to_numpy(),
        "sample_id": df["sample_id"].astype(str).to_numpy(),
        "y_true_final": y_all_global,
        "validity_alarm": as01(df[VALIDITY]).to_numpy(int),
        "eligible": as01(df[ELIGIBLE]).to_numpy(int),
        "internal_nonfinite_seen": as01(df[NONFINITE_ROUTE]).to_numpy(int),
    }
    aggregate_branch_rows: List[Dict[str, Any]] = []
    work_global_idx = work["global_all_index"].to_numpy(int)

    for (system, criterion), pred in pred_store.items():
        m = metrics(y_all_global, pred)
        summary_rows.append(
            {
                "system": system,
                "criterion": criterion,
                "aggregate_target_met": criterion_target_met(m, criterion),
                **m,
            }
        )
        prediction_cols[f"pred__{system}__{criterion}"] = pred.astype(np.int8)

        learned_pred = pred[work_global_idx]
        for branch_flag in [0, 1]:
            bm = z == branch_flag
            bm_metrics = metrics(y[bm], learned_pred[bm])
            aggregate_branch_rows.append(
                {
                    "system": system,
                    "criterion": criterion,
                    "branch": branch_name(branch_flag),
                    "branch_rows": int(bm.sum()),
                    "branch_positive": int(y[bm].sum()),
                    "branch_negative": int(np.sum(y[bm] == 0)),
                    **bm_metrics,
                }
            )
        all_metrics = metrics(y, learned_pred)
        aggregate_branch_rows.append(
            {
                "system": system,
                "criterion": criterion,
                "branch": "all_eligible",
                "branch_rows": len(y),
                "branch_positive": int(y.sum()),
                "branch_negative": int(np.sum(y == 0)),
                **all_metrics,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "strict_nested_final_metrics_v4_4_6.csv", index=False)
    pd.DataFrame(aggregate_branch_rows).to_csv(
        out_dir / "strict_nested_branch_metrics_v4_4_6.csv", index=False
    )
    pd.DataFrame(outer_rows).to_csv(
        out_dir / "outer_fold_policy_and_test_metrics_v4_4_6.csv", index=False
    )
    pd.DataFrame(outer_branch_rows).to_csv(
        out_dir / "outer_fold_branch_metrics_v4_4_6.csv", index=False
    )
    pd.DataFrame(candidate_rows).to_csv(
        out_dir / "outer_fold_branch_c2_candidate_calibration_v4_4_6.csv",
        index=False,
    )
    pd.DataFrame(route_audit_rows).to_csv(
        out_dir / "outer_fold_branch_route_audit_v4_4_6.csv", index=False
    )
    pd.concat(score_rows, ignore_index=True).to_csv(
        out_dir / "outer_test_branch_calibrated_scores_v4_4_6.csv", index=False
    )
    pd.DataFrame(prediction_cols).to_csv(
        out_dir / "strict_nested_binary_predictions_v4_4_6.csv", index=False
    )
    if bundle_rows:
        pd.DataFrame(bundle_rows).to_csv(
            out_dir / "model_bundle_manifest_v4_4_6.csv", index=False
        )

    outer_df = pd.DataFrame(outer_rows)
    route_counts = (
        outer_df.query("system == 'strict_repair_branch_c1c2_best'")
        .groupby(["criterion", "selected_route"], dropna=False)
        .size()
        .reset_index(name="outer_fold_count")
    )
    route_counts.to_csv(
        out_dir / "selected_branch_c2_route_counts_v4_4_6.csv", index=False
    )

    log("===== STRICT NESTED v4.4.6 REPAIR-AWARE FINAL METRICS =====")
    show = summary.copy()
    show["recall_pct"] = show["recall"] * 100
    show["far_pct"] = show["far"] * 100
    print(
        show[
            [
                "system", "criterion", "aggregate_target_met", "recall_pct",
                "far_pct", "tp", "fn", "fp", "tn", "precision", "pred_pos",
            ]
        ].to_string(index=False),
        flush=True,
    )
    log(f"OUT_DIR={out_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--c2-route-modes", default="recall98,quantile45")
    ap.add_argument(
        "--finite-branch-feature-mode",
        choices=["exclude_nonfinite", "all"],
        default="exclude_nonfinite",
    )
    ap.add_argument("--c2-finite-neg-weight", type=float, default=2.0)
    ap.add_argument("--c2-nonfinite-neg-weight", type=float, default=2.0)
    ap.add_argument("--c2-min-pool-rows-finite", type=int, default=200)
    ap.add_argument("--c2-min-pool-rows-nonfinite", type=int, default=100)
    ap.add_argument("--c2-min-inner-branch-positives", type=int, default=3)
    ap.add_argument("--c2-min-total-branch-positives", type=int, default=8)
    ap.add_argument("--random-seed", type=int, default=773003)
    ap.add_argument("--policy-grid-max", type=int, default=100)
    ap.add_argument("--far-budgets", default="0.02")
    ap.add_argument("--recall-targets", default="0.90,0.95,0.99")
    ap.add_argument("--expected-rows-per-seed", type=int, default=6019)
    ap.add_argument("--repair-budget-max-default", type=int, default=4)
    ap.add_argument("--feature-contract-json", default="")
    ap.add_argument("--excluded-activation-features-file", default="")
    ap.add_argument("--expected-extended-source-feature-count", type=int, default=4780)
    ap.add_argument("--expected-runtime-summary-feature-count", type=int, default=669)
    ap.add_argument("--expected-activation-feature-count", type=int, default=5449)
    ap.add_argument(
        "--save-models", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--preflight-only", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
