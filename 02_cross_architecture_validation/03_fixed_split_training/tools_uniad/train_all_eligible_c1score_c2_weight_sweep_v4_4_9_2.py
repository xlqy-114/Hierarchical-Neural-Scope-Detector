#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UniAD2 v4.4.9.2: all-eligible C1-score-assisted C2 with weight sweep.

Design contract
---------------
1. Internal non-finite rows are intercepted by layer 1 and are excluded before any
   learned-model fitting, thresholding, or learned-detector evaluation.
2. Among rows with no internal non-finite observation, deterministic final-output
   validity alarms are handled directly. The learned branch is trained only on
   finite/legal/gate-accepted rows.
3. No OOF, no nested CV, no LOSO, no dual route, and no irreversible C1 hard gate.
   C1 produces a score for every learned-eligible row; C2 scores every eligible row.
4. C2 training uses all learned positives plus repair-triggered safe negatives,
   C1-hard negatives, and a reproducible random background-negative sample.
5. A validation-only C2 positive-weight sweep is performed. Model variant and
   threshold are selected only on validation; test is evaluated once with those choices.
6. All normal controls and planned-but-ineffective injections remain as negatives.
7. No import from any earlier experiment package. This file is self-contained apart
   from standard scientific Python dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time

# Prevent uncontrolled OpenMP/BLAS oversubscription. Users can override with
# V4492_NUM_THREADS before launching the process. This must be set before numpy/sklearn import.
_default_threads = os.environ.get("V4492_NUM_THREADS", "8")
for _env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_name, _default_threads)

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

VERSION = "v4.4.9.2"
DEFAULT_ALL_SEEDS = [
    959101, 959102, 959103, 959104, 959105, 959106, 959107,
    959113, 959114, 959115, 959116, 959117, 959118, 959119,
    959120, 959121, 959122, 959123, 959124,
]
DEFAULT_TRAIN_SEEDS = [
    959101, 959102, 959103, 959104, 959105,
    959113, 959114, 959115, 959116, 959117,
    959118, 959119, 959120, 959121, 959122,
]
DEFAULT_VAL_SEEDS = [959106, 959123]
DEFAULT_TEST_SEEDS = [959107, 959124]

INTERNAL_NF_COL = "feature__act_nonfinite_count_sample_sum"
FINAL_RISK_COL = "label__post_repair_unsafe_1p0_or_gate"
OUTPUT_ILLEGAL_COL = "label__repaired_output_illegal_or_missing"
GATE_REJECT_COL = "label__gate_reject"
GATE_ACCEPT_COL = "label__gate_accept"
OUTPUT_NONFINITE_COL = "label__repaired_output_nonfinite"
SEVERE_COL = "label__repaired_severe_l2max_gt_1p0_or_illegal"
REQUIRED_NONFEATURE_COLUMNS = [
    "sample_i", "sample_id", INTERNAL_NF_COL, FINAL_RISK_COL,
    OUTPUT_ILLEGAL_COL, GATE_REJECT_COL, GATE_ACCEPT_COL,
    OUTPUT_NONFINITE_COL, SEVERE_COL,
]

FORBIDDEN_MODEL_TOKENS = (
    "label__", "debug__", "clean_", "traj_delta", "harmful", "severe",
    "fault_", "target_", "bit_", "event_id", "ground_truth", "gt_",
)

GROUPS = [
    "U1_img_backbone_neck", "U2_bev_encoder", "U3_track_det_head",
    "U4_map_seg_head", "U5_motion_head", "U6_occ_head",
    "U7_planning_head", "U8_final_output", "U9_online_track_memory", "unknown",
]
ROLES = ["attention", "transformer_layer", "planning_head", "ffn", "projection_or_head", "generic", "unknown"]
ACTIONS = ["standard_fallback", "zero_tensor", "input_passthrough", "clip_median", "none", "other"]
REASON_BUCKETS = [
    "budget_exhausted", "finite_extremes_clip_median", "zero_tensor_containment",
    "input_passthrough", "detect_only", "no_modification", "other",
]
NUMERIC_EVENT_FIELDS = [
    "trace_order", "call_count", "monitor_sample_budget", "repair_modified_count_so_far",
    "pre_repair_finite_ratio_sample", "pre_repair_absmax_finite_sample",
    "pre_repair_nonfinite_count_sample", "pre_repair_sample_numel",
    "sample_numel_total", "nonfinite_count_sample", "huge_count_sample",
    "abnormal_count_sample", "nonfinite_ratio_sample", "huge_ratio_sample",
    "abnormal_ratio_sample", "absmax_ratio_sample", "finite_ratio_full",
    "absmax_finite_full", "nonfinite_count_full", "huge_count_full",
    "abnormal_count_full", "nonfinite_ratio_full", "huge_ratio_full",
    "abnormal_ratio_full", "absmax_ratio_full", "post_repair_finite_ratio_full",
    "post_repair_absmax_finite_full", "post_repair_nonfinite_count_full",
    "post_repair_huge_count_full", "post_repair_abnormal_count_full",
    "post_repair_abnormal_ratio_full", "post_repair_absmax_ratio_full",
    "repair_idx", "threshold_absmax", "replacement_value",
]
NUMERIC_AGGS = ["sum", "mean", "max", "last"]


def parse_seed_list(text: str | Sequence[int]) -> List[int]:
    if isinstance(text, (list, tuple)):
        return [int(x) for x in text]
    out = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("Seed list is empty")
    if len(out) != len(set(out)):
        raise ValueError(f"Duplicate seeds: {out}")
    return out


def parse_float_list(text: str | Sequence[float]) -> List[float]:
    if isinstance(text, (list, tuple)):
        vals = [float(x) for x in text]
    else:
        vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Float list is empty")
    if any((not math.isfinite(x)) or x <= 0 for x in vals):
        raise ValueError(f"All values must be finite and > 0: {vals}")
    if len(vals) != len(set(vals)):
        raise ValueError(f"Duplicate values: {vals}")
    return vals


def as01(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0).to_numpy(dtype=np.float64) > 0


def safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)


def sha256_text_lines(lines: Sequence[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(str(line).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def file_sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def iter_string_lists(obj: Any, path: str = "$") -> Iterator[Tuple[str, List[str]]]:
    if isinstance(obj, list):
        if obj and all(isinstance(x, str) for x in obj):
            yield path, list(obj)
        for i, x in enumerate(obj):
            yield from iter_string_lists(x, f"{path}[{i}]")
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        if keys and all(isinstance(k, str) and k.startswith("feature__") for k in keys):
            yield path + ".<dict_keys>", keys
        for k, v in obj.items():
            yield from iter_string_lists(v, f"{path}.{k}")


def resolve_activation_contract(
    path: Path,
    expected_total: int,
    expected_reviewed: int,
    expected_runtime: int,
    expected_legacy_repair: int,
    legacy_repair_prefix: str,
) -> Tuple[List[str], Dict[str, Any]]:
    """Resolve the exact activation-only contract without fuzzy matching.

    Accepted authoritative layouts are deliberately narrow:
    1. one exact activation-only list of ``expected_total`` columns;
    2. one unique disjoint ``expected_reviewed + expected_runtime`` pair; or
    3. one combined previous-model-input list containing exactly
       ``expected_total`` activation columns and exactly
       ``expected_legacy_repair`` legacy repair telemetry columns, where every
       legacy repair column belongs to the explicit namespace
       ``legacy_repair_prefix``.

    Layout (3) matches the formal v4.4.6.2 frozen contract, whose top-level
    ``features`` list contains 5,651 total inputs = 5,449 activation + 202
    repair telemetry. The old telemetry is excluded from the activation source
    contract and is independently rematerialized by this self-contained version.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Frozen activation contract not found: {path}")
    if expected_legacy_repair < 0:
        raise ValueError("expected_legacy_repair must be >= 0")
    if not legacy_repair_prefix.startswith("feature__"):
        raise ValueError("legacy_repair_prefix must be an explicit feature namespace")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    candidates: List[Tuple[str, List[str]]] = []
    for p, vals in iter_string_lists(obj):
        uniq = list(dict.fromkeys(vals))
        if len(uniq) == len(vals) and vals and all(x.startswith("feature__") for x in vals):
            candidates.append((p, vals))

    exact = [(p, v) for p, v in candidates if len(v) == expected_total]
    mode = ""
    source_paths: List[str] = []
    legacy_repair_cols: List[str] = []
    combined_feature_count = 0

    if len(exact) == 1:
        cols = exact[0][1]
        mode = "single_exact_activation_list"
        source_paths = [exact[0][0]]
    elif len(exact) > 1:
        normalized = {tuple(v) for _, v in exact}
        if len(normalized) != 1:
            raise RuntimeError(
                f"Multiple different {expected_total}-feature activation lists in contract: "
                f"{[p for p, _ in exact]}"
            )
        cols = exact[0][1]
        mode = "multiple_identical_exact_activation_lists"
        source_paths = [p for p, _ in exact]
    else:
        # Formal v4.4.6.2 layout: $.features has 5651 total model inputs.
        combined_expected = expected_total + expected_legacy_repair
        combined_matches: List[Tuple[str, List[str], List[str], List[str]]] = []
        if expected_legacy_repair > 0:
            for p, vals in candidates:
                if len(vals) != combined_expected:
                    continue
                repairs = [c for c in vals if c.startswith(legacy_repair_prefix)]
                activations = [c for c in vals if not c.startswith(legacy_repair_prefix)]
                if len(repairs) != expected_legacy_repair:
                    continue
                if len(activations) != expected_total:
                    continue
                if set(repairs).intersection(activations):
                    continue
                combined_matches.append((p, vals, activations, repairs))

        unique_combined: Dict[Tuple[str, ...], Tuple[str, List[str], List[str], List[str]]] = {}
        for item in combined_matches:
            unique_combined.setdefault(tuple(item[2]), item)

        if len(unique_combined) == 1:
            p0, vals0, cols, legacy_repair_cols = next(iter(unique_combined.values()))
            mode = "combined_model_input_split_by_exact_legacy_repair_namespace"
            source_paths = [p0]
            combined_feature_count = len(vals0)
        elif len(unique_combined) > 1:
            raise RuntimeError(
                "Multiple different combined model-input contracts resolve to different "
                f"activation lists: {[x[0] for x in unique_combined.values()]}"
            )
        else:
            reviewed = [(p, v) for p, v in candidates if len(v) == expected_reviewed]
            runtime = [(p, v) for p, v in candidates if len(v) == expected_runtime]
            combos: List[Tuple[str, str, List[str]]] = []
            for p1, a in reviewed:
                for p2, b in runtime:
                    if set(a).isdisjoint(set(b)) and len(a) + len(b) == expected_total:
                        combos.append((p1, p2, a + b))
            unique_combos: Dict[Tuple[str, ...], Tuple[str, str, List[str]]] = {}
            for c in combos:
                unique_combos.setdefault(tuple(c[2]), c)
            if len(unique_combos) != 1:
                report = {
                    "candidate_paths_and_lengths": [(p, len(v)) for p, v in candidates],
                    "expected_activation_only": expected_total,
                    "expected_reviewed_plus_runtime": [expected_reviewed, expected_runtime],
                    "expected_combined_total": combined_expected,
                    "legacy_repair_prefix": legacy_repair_prefix,
                    "expected_legacy_repair_count": expected_legacy_repair,
                }
                raise RuntimeError(
                    "Cannot resolve exact activation contract. Expected exactly one "
                    f"activation-only list of {expected_total}, one unique disjoint "
                    f"{expected_reviewed}+{expected_runtime} pair, or one combined list "
                    f"of {combined_expected} that splits exactly into {expected_total} "
                    f"activation and {expected_legacy_repair} '{legacy_repair_prefix}*' "
                    f"legacy repair features. Observed: {report}"
                )
            p1, p2, cols = next(iter(unique_combos.values()))
            mode = "reviewed_plus_runtime_exact_union"
            source_paths = [p1, p2]

    if len(cols) != expected_total or len(set(cols)) != expected_total:
        raise RuntimeError("Resolved activation contract has wrong count or duplicate columns")
    if any(c.startswith(legacy_repair_prefix) for c in cols):
        raise RuntimeError(
            "Resolved activation contract still contains legacy repair telemetry namespace: "
            f"{[c for c in cols if c.startswith(legacy_repair_prefix)][:20]}"
        )
    if INTERNAL_NF_COL not in cols:
        raise RuntimeError(f"Required online nonfinite route feature absent from contract: {INTERNAL_NF_COL}")
    forbidden_hits = [c for c in cols if any(tok in c.lower() for tok in FORBIDDEN_MODEL_TOKENS)]
    if forbidden_hits:
        raise RuntimeError(f"Forbidden feature names in activation contract: {forbidden_hits[:20]}")

    audit = {
        "contract_path": str(path),
        "contract_file_sha256": file_sha256(path),
        "resolution_mode": mode,
        "source_paths": source_paths,
        "feature_count": len(cols),
        "feature_name_sha256": sha256_text_lines(cols),
        "combined_source_feature_count": int(combined_feature_count),
        "legacy_repair_prefix": legacy_repair_prefix,
        "legacy_repair_feature_count_excluded": len(legacy_repair_cols),
        "legacy_repair_feature_name_sha256": sha256_text_lines(legacy_repair_cols) if legacy_repair_cols else "",
        "legacy_repair_features_excluded": legacy_repair_cols,
    }
    return cols, audit


def one_seed_dir(data_root: Path, seed: int) -> Path:
    matches = sorted(data_root.glob(f"all_prefix_6019_seed{seed}_*_extv431"))
    if len(matches) != 1:
        raise RuntimeError(f"seed={seed}: expected exactly one v4.3.1 directory, found={len(matches)}: {matches}")
    return matches[0]


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_no}")
            yield obj


def role_bucket(v: Any) -> str:
    s = str(v or "").strip()
    return s if s in ROLES else "unknown"


def group_bucket(v: Any) -> str:
    s = str(v or "").strip()
    return s if s in GROUPS else "unknown"


def action_bucket(event: Mapping[str, Any]) -> str:
    s = str(event.get("high_contam_action") or "").strip().lower()
    reason = str(event.get("repair_reason") or "").strip().lower()
    method = str(event.get("repair_method") or "").strip().lower()
    if s in {"standard_fallback", "zero_tensor", "input_passthrough"}:
        return s
    if "clip" in s or "median" in s or "clip" in reason or "median" in reason or "clip" in method:
        return "clip_median"
    if not s and not reason and not method:
        return "none"
    return "other"


def reason_bucket(v: Any) -> str:
    s = str(v or "").lower()
    if "budget" in s and "exhaust" in s:
        return "budget_exhausted"
    if "clip" in s or "median" in s:
        return "finite_extremes_clip_median"
    if "zero_tensor" in s:
        return "zero_tensor_containment"
    if "passthrough" in s:
        return "input_passthrough"
    if "detect" in s and "no_mod" in s:
        return "detect_only"
    if "no_mod" in s or "not_modified" in s:
        return "no_modification"
    return "other"


def empty_event_state() -> Dict[str, Any]:
    return {
        "events": 0,
        "triggered": 0,
        "modified": 0,
        "nonfinite_events": 0,
        "huge_events": 0,
        "budget_exhausted": 0,
        "replace_fail": 0,
        "full_check": 0,
        "high_contam": 0,
        "post_stats_error": 0,
        "modules": set(),
        "groups": set(),
        "roles": set(),
        "group_counts": Counter(),
        "role_counts": Counter(),
        "action_counts": Counter(),
        "reason_counts": Counter(),
        "numeric": defaultdict(list),
        "first_group": "unknown",
        "last_group": "unknown",
        "first_role": "unknown",
        "last_role": "unknown",
        "first_trace": float("nan"),
        "last_trace": float("nan"),
        "first_modified_trace": float("nan"),
        "last_modified_trace": float("nan"),
    }


def telemetry_feature_names() -> List[str]:
    names = [
        "feature__repair_event_count", "feature__repair_trigger_count", "feature__repair_modified_count",
        "feature__repair_nonfinite_event_count", "feature__repair_huge_event_count",
        "feature__repair_budget_exhausted_count", "feature__repair_replace_output_fail_count",
        "feature__repair_full_check_count", "feature__repair_high_contam_count",
        "feature__repair_post_stats_error_count", "feature__repair_unique_module_count",
        "feature__repair_unique_group_count", "feature__repair_unique_role_count",
        "feature__repair_recurrence_count", "feature__repair_trace_span",
        "feature__repair_modified_trace_span",
    ]
    for g in GROUPS:
        tok = g.lower()
        names += [f"feature__repair_group_count__{tok}", f"feature__repair_first_group__{tok}", f"feature__repair_last_group__{tok}"]
    for r in ROLES:
        names += [f"feature__repair_role_count__{r}", f"feature__repair_first_role__{r}", f"feature__repair_last_role__{r}"]
    for a in ACTIONS:
        names.append(f"feature__repair_action_count__{a}")
    for r in REASON_BUCKETS:
        names.append(f"feature__repair_reason_count__{r}")
    for field in NUMERIC_EVENT_FIELDS:
        for agg in NUMERIC_AGGS:
            names.append(f"feature__repair_event_{field}__{agg}")
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate telemetry feature names")
    return names

TELEMETRY_FEATURES = telemetry_feature_names()


def finalize_event_state(st: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {k: 0.0 for k in TELEMETRY_FEATURES}
    out.update({
        "feature__repair_event_count": float(st["events"]),
        "feature__repair_trigger_count": float(st["triggered"]),
        "feature__repair_modified_count": float(st["modified"]),
        "feature__repair_nonfinite_event_count": float(st["nonfinite_events"]),
        "feature__repair_huge_event_count": float(st["huge_events"]),
        "feature__repair_budget_exhausted_count": float(st["budget_exhausted"]),
        "feature__repair_replace_output_fail_count": float(st["replace_fail"]),
        "feature__repair_full_check_count": float(st["full_check"]),
        "feature__repair_high_contam_count": float(st["high_contam"]),
        "feature__repair_post_stats_error_count": float(st["post_stats_error"]),
        "feature__repair_unique_module_count": float(len(st["modules"])),
        "feature__repair_unique_group_count": float(len(st["groups"])),
        "feature__repair_unique_role_count": float(len(st["roles"])),
        "feature__repair_recurrence_count": float(max(0, st["events"] - len(st["modules"]))),
    })
    ft, lt = safe_float(st["first_trace"]), safe_float(st["last_trace"])
    fmt, lmt = safe_float(st["first_modified_trace"]), safe_float(st["last_modified_trace"])
    out["feature__repair_trace_span"] = float(lt - ft) if math.isfinite(ft) and math.isfinite(lt) else 0.0
    out["feature__repair_modified_trace_span"] = float(lmt - fmt) if math.isfinite(fmt) and math.isfinite(lmt) else 0.0
    for g in GROUPS:
        tok = g.lower()
        out[f"feature__repair_group_count__{tok}"] = float(st["group_counts"][g])
        out[f"feature__repair_first_group__{tok}"] = float(st["first_group"] == g)
        out[f"feature__repair_last_group__{tok}"] = float(st["last_group"] == g)
    for r in ROLES:
        out[f"feature__repair_role_count__{r}"] = float(st["role_counts"][r])
        out[f"feature__repair_first_role__{r}"] = float(st["first_role"] == r)
        out[f"feature__repair_last_role__{r}"] = float(st["last_role"] == r)
    for a in ACTIONS:
        out[f"feature__repair_action_count__{a}"] = float(st["action_counts"][a])
    for r in REASON_BUCKETS:
        out[f"feature__repair_reason_count__{r}"] = float(st["reason_counts"][r])
    for field in NUMERIC_EVENT_FIELDS:
        vals = np.asarray(st["numeric"].get(field, []), dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        for agg in NUMERIC_AGGS:
            key = f"feature__repair_event_{field}__{agg}"
            if vals.size == 0:
                out[key] = float("nan")
            elif agg == "sum":
                out[key] = float(vals.sum())
            elif agg == "mean":
                out[key] = float(vals.mean())
            elif agg == "max":
                out[key] = float(vals.max())
            elif agg == "last":
                out[key] = float(vals[-1])
    return out


def materialize_repair_telemetry(seed_dir: Path, cache_dir: Path, seed: int, expected_sample_i: Sequence[int]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"repair_telemetry_seed{seed}_{VERSION.replace('.', '_')}.npz"
    event_path = seed_dir / "debug_repair_events_v4_3_1.jsonl"
    if not event_path.is_file():
        raise FileNotFoundError(event_path)
    sample_i_arr = np.asarray(expected_sample_i, dtype=np.int64)
    if len(np.unique(sample_i_arr)) != len(sample_i_arr):
        raise RuntimeError(f"seed={seed}: duplicate sample_i in activation CSV")
    if cache_path.is_file() and cache_path.stat().st_mtime >= event_path.stat().st_mtime:
        z = np.load(cache_path, allow_pickle=False)
        cached_i = z["sample_i"].astype(np.int64)
        cached_names = [str(x) for x in z["feature_names"].tolist()]
        if np.array_equal(cached_i, sample_i_arr) and cached_names == TELEMETRY_FEATURES:
            mat = z["X"].astype(np.float32, copy=False)
            return pd.DataFrame(mat, index=cached_i, columns=TELEMETRY_FEATURES), {
                "seed": seed, "cache_used": 1, "event_path": str(event_path), "event_rows": int(z["event_rows"][0])
            }
    valid = set(int(x) for x in sample_i_arr.tolist())
    states: Dict[int, Dict[str, Any]] = {}
    event_rows = 0
    unknown_sample_events = 0
    for ev in read_jsonl(event_path):
        event_rows += 1
        sample_i = safe_int(ev.get("sample_i"), -1)
        if sample_i not in valid:
            unknown_sample_events += 1
            continue
        st = states.setdefault(sample_i, empty_event_state())
        st["events"] += 1
        trig = int(safe_int(ev.get("repair_triggered"), 0) > 0)
        mod = int(safe_int(ev.get("repair_modified"), 0) > 0)
        st["triggered"] += trig
        st["modified"] += mod
        st["nonfinite_events"] += int(safe_int(ev.get("nonfinite_count_sample"), 0) > 0 or safe_int(ev.get("pre_repair_nonfinite_count_sample"), 0) > 0 or safe_int(ev.get("nonfinite_count_full"), 0) > 0)
        st["huge_events"] += int(safe_int(ev.get("huge_count_sample"), 0) > 0 or safe_int(ev.get("huge_count_full"), 0) > 0)
        rb = reason_bucket(ev.get("repair_reason"))
        st["reason_counts"][rb] += 1
        st["budget_exhausted"] += int(rb == "budget_exhausted")
        st["replace_fail"] += int(safe_int(ev.get("replace_output_ok"), 1) == 0)
        st["full_check"] += int(safe_int(ev.get("full_check_on_trigger"), 0) > 0)
        st["high_contam"] += int(safe_int(ev.get("high_contam_policy_flag", ev.get("high_contam_flag", 0)), 0) > 0)
        st["post_stats_error"] += int(bool(str(ev.get("post_repair_stats_error") or "").strip()))
        module = str(ev.get("module_name") or "").strip()
        if module:
            st["modules"].add(module)
        group = group_bucket(ev.get("uniad_group"))
        role = role_bucket(ev.get("module_aware_role"))
        action = action_bucket(ev)
        st["groups"].add(group)
        st["roles"].add(role)
        st["group_counts"][group] += 1
        st["role_counts"][role] += 1
        st["action_counts"][action] += 1
        st["first_group"] = group if st["events"] == 1 else st["first_group"]
        st["last_group"] = group
        st["first_role"] = role if st["events"] == 1 else st["first_role"]
        st["last_role"] = role
        tr = safe_float(ev.get("trace_order"))
        if st["events"] == 1:
            st["first_trace"] = tr
        st["last_trace"] = tr
        if mod:
            if not math.isfinite(safe_float(st["first_modified_trace"])):
                st["first_modified_trace"] = tr
            st["last_modified_trace"] = tr
        for field in NUMERIC_EVENT_FIELDS:
            x = safe_float(ev.get(field))
            if math.isfinite(x):
                st["numeric"][field].append(x)
    mat = np.empty((len(sample_i_arr), len(TELEMETRY_FEATURES)), dtype=np.float32)
    for row_idx, sample_i in enumerate(sample_i_arr.tolist()):
        feats = finalize_event_state(states.get(int(sample_i), empty_event_state()))
        mat[row_idx] = np.asarray([feats[k] for k in TELEMETRY_FEATURES], dtype=np.float32)
    np.savez_compressed(
        cache_path, sample_i=sample_i_arr, feature_names=np.asarray(TELEMETRY_FEATURES),
        X=mat, event_rows=np.asarray([event_rows], dtype=np.int64),
    )
    return pd.DataFrame(mat, index=sample_i_arr, columns=TELEMETRY_FEATURES), {
        "seed": seed, "cache_used": 0, "event_path": str(event_path),
        "event_rows": event_rows, "unknown_sample_events": unknown_sample_events,
        "samples_with_events": len(states),
    }


def aggregate_debug_metadata(seed_dir: Path, sample_i_values: Sequence[int]) -> pd.DataFrame:
    path = seed_dir / "debug_metadata_v4_3_1.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    valid = set(int(x) for x in sample_i_values)
    state: Dict[int, Dict[str, int]] = defaultdict(lambda: {"target": 0, "applied": 0, "effective": 0, "rows": 0})
    for obj in read_jsonl(path):
        i = safe_int(obj.get("sample_i"), -1)
        if i not in valid:
            continue
        st = state[i]
        st["rows"] += 1
        st["target"] = max(st["target"], int(safe_int(obj.get("debug__target_present"), 0) > 0))
        st["applied"] = max(st["applied"], int(safe_int(obj.get("debug__fault_applied_count"), 0) > 0))
        st["effective"] = max(st["effective"], int(safe_int(obj.get("debug__fault_effective_count"), 0) > 0))
    rows = []
    for i in sample_i_values:
        st = state.get(int(i), {"target": 0, "applied": 0, "effective": 0, "rows": 0})
        rows.append({
            "sample_i": int(i), "fault_plan_target_present": st["target"],
            "fault_applied": st["applied"], "fault_effective": st["effective"],
            "debug_metadata_row_count": st["rows"],
        })
    return pd.DataFrame(rows).set_index("sample_i")


@dataclass
class SplitData:
    name: str
    seeds: List[int]
    X: np.ndarray
    meta: pd.DataFrame


def load_seed_frame(seed_dir: Path, seed: int, activation_features: Sequence[str], cache_dir: Path, max_rows: int = 0) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    csv_path = seed_dir / "detector_train_activation_only_v4_3_1.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [c for c in list(activation_features) + REQUIRED_NONFEATURE_COLUMNS if c not in header]
    if missing:
        raise RuntimeError(f"seed={seed}: missing required columns ({len(missing)}): {missing[:30]}")
    usecols = list(dict.fromkeys(list(REQUIRED_NONFEATURE_COLUMNS) + list(activation_features)))
    nrows = int(max_rows) if int(max_rows) > 0 else None
    dtype_map: Dict[str, Any] = {c: np.float32 for c in activation_features}
    for c in [FINAL_RISK_COL, OUTPUT_ILLEGAL_COL, GATE_REJECT_COL, GATE_ACCEPT_COL, OUTPUT_NONFINITE_COL, SEVERE_COL]:
        dtype_map[c] = np.float32
    dtype_map["sample_i"] = np.int64
    dtype_map["sample_id"] = str
    try:
        df = pd.read_csv(
            csv_path, usecols=usecols, nrows=nrows, low_memory=False, dtype=dtype_map,
            na_values=["", "nan", "NaN", "None", "null"], keep_default_na=True,
        )
    except Exception as exc:
        raise RuntimeError(f"seed={seed}: strict numeric CSV load failed for {csv_path}: {exc}") from exc
    if df.empty:
        raise RuntimeError(f"seed={seed}: no rows loaded from {csv_path}")
    df["sample_i"] = pd.to_numeric(df["sample_i"], errors="raise").astype(np.int64)
    if df["sample_i"].duplicated().any():
        raise RuntimeError(f"seed={seed}: duplicate sample_i in training CSV")
    sample_i = df["sample_i"].to_numpy(dtype=np.int64)
    telemetry, tele_audit = materialize_repair_telemetry(seed_dir, cache_dir, seed, sample_i)
    telemetry = telemetry.loc[sample_i]
    debug = aggregate_debug_metadata(seed_dir, sample_i).loc[sample_i]
    X_act = df[list(activation_features)].to_numpy(dtype=np.float32, copy=True)
    X_rep = telemetry.to_numpy(dtype=np.float32, copy=True)
    activation_inf_count = int(np.isinf(X_act).sum())
    repair_inf_count = int(np.isinf(X_rep).sum())
    if activation_inf_count:
        X_act[np.isinf(X_act)] = np.nan
    if repair_inf_count:
        X_rep[np.isinf(X_rep)] = np.nan
    X = np.concatenate([X_act, X_rep], axis=1)
    internal_nf = as01(df[INTERNAL_NF_COL])
    final_risk = as01(df[FINAL_RISK_COL])
    output_illegal = as01(df[OUTPUT_ILLEGAL_COL])
    gate_reject = as01(df[GATE_REJECT_COL])
    gate_accept = as01(df[GATE_ACCEPT_COL])
    output_nonfinite = as01(df[OUTPUT_NONFINITE_COL])
    severe = as01(df[SEVERE_COL])
    validity_alarm = output_illegal | gate_reject
    no_internal_nf = ~internal_nf
    learned_eligible = no_internal_nf & (~validity_alarm)
    # On learned-eligible rows, final risk must equal severe label and gate_accept must be true.
    mismatch = learned_eligible & (final_risk != severe)
    if int(mismatch.sum()) != 0:
        raise RuntimeError(f"seed={seed}: learned target mismatch final_risk vs severe on accepted rows: {int(mismatch.sum())}")
    if int((learned_eligible & (~gate_accept)).sum()) != 0:
        raise RuntimeError(f"seed={seed}: learned eligible rows with gate_accept=0")
    meta = pd.DataFrame({
        "seed": int(seed),
        "sample_i": sample_i,
        "sample_id": df["sample_id"].astype(str).to_numpy(),
        "internal_nonfinite_seen": internal_nf.astype(np.int8),
        "no_internal_nonfinite": no_internal_nf.astype(np.int8),
        "validity_alarm": validity_alarm.astype(np.int8),
        "learned_eligible": learned_eligible.astype(np.int8),
        "final_risk_1p0": final_risk.astype(np.int8),
        "learned_y_1p0": (final_risk & learned_eligible).astype(np.int8),
        "gate_accept": gate_accept.astype(np.int8),
        "output_illegal": output_illegal.astype(np.int8),
        "output_nonfinite": output_nonfinite.astype(np.int8),
        "fault_plan_target_present": debug["fault_plan_target_present"].to_numpy(dtype=np.int8),
        "fault_applied": debug["fault_applied"].to_numpy(dtype=np.int8),
        "fault_effective": debug["fault_effective"].to_numpy(dtype=np.int8),
    })
    meta["system_y"] = ((meta["internal_nonfinite_seen"] > 0) | (meta["final_risk_1p0"] > 0)).astype(np.int8)
    audit = {
        "seed": seed, "seed_dir": str(seed_dir), "csv_path": str(csv_path),
        "rows": len(df), "activation_features": len(activation_features),
        "repair_features": len(TELEMETRY_FEATURES), "total_features": X.shape[1],
        "activation_inf_replaced_with_nan": activation_inf_count,
        "repair_inf_replaced_with_nan": repair_inf_count,
        "internal_nonfinite": int(internal_nf.sum()),
        "no_internal_nonfinite": int(no_internal_nf.sum()),
        "validity_alarm_no_nf": int((no_internal_nf & validity_alarm).sum()),
        "learned_eligible": int(learned_eligible.sum()),
        "learned_positive": int((learned_eligible & final_risk).sum()),
        "final_risk_no_nf": int((no_internal_nf & final_risk).sum()),
        "fault_effective": int(meta["fault_effective"].sum()),
        "planned_but_ineffective": int(((meta["fault_plan_target_present"] > 0) & (meta["fault_effective"] == 0)).sum()),
        "normal_control": int((meta["fault_plan_target_present"] == 0).sum()),
        "telemetry": tele_audit,
    }
    return X, meta, audit


def load_split(name: str, seeds: Sequence[int], data_root: Path, activation_features: Sequence[str], cache_dir: Path, max_rows_per_seed: int = 0) -> Tuple[SplitData, List[Dict[str, Any]]]:
    xs: List[np.ndarray] = []
    metas: List[pd.DataFrame] = []
    audits: List[Dict[str, Any]] = []
    for idx, seed in enumerate(seeds, 1):
        print(f"[{VERSION}] load {name} seed {idx}/{len(seeds)}: {seed}", flush=True)
        seed_dir = one_seed_dir(data_root, int(seed))
        X, meta, audit = load_seed_frame(seed_dir, int(seed), activation_features, cache_dir, max_rows=max_rows_per_seed)
        xs.append(X)
        metas.append(meta)
        audits.append(audit)
    return SplitData(name=name, seeds=[int(x) for x in seeds], X=np.concatenate(xs, axis=0), meta=pd.concat(metas, ignore_index=True)), audits


def positive_proba(model: HistGradientBoostingClassifier, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        raise RuntimeError(f"Model has no positive class: {classes}")
    return proba[:, classes.index(1)].astype(np.float64)


def metrics_from_counts(tp: int, fn: int, fp: int, tn: int) -> Dict[str, Any]:
    return {
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "far": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "positive_total": int(tp + fn), "negative_total": int(fp + tn),
    }


def prefix_metrics(prefix: str, values: Mapping[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in values.items()}


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    yb = np.asarray(y, dtype=bool)
    pb = np.asarray(pred, dtype=bool)
    tp = int(np.sum(yb & pb)); fn = int(np.sum(yb & (~pb)))
    fp = int(np.sum((~yb) & pb)); tn = int(np.sum((~yb) & (~pb)))
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "far": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "positive_total": tp + fn, "negative_total": fp + tn,
    }


def score_threshold_curve(meta: pd.DataFrame, learned_score: np.ndarray, model_variant: str, positive_weight: float) -> pd.DataFrame:
    if len(meta) != len(learned_score):
        raise ValueError("Length mismatch in threshold curve")
    y_system = meta["system_y"].to_numpy(dtype=bool)
    first_alarm = meta["internal_nonfinite_seen"].to_numpy(dtype=bool)
    validity = meta["validity_alarm"].to_numpy(dtype=bool)
    eligible = meta["learned_eligible"].to_numpy(dtype=bool)
    y_final = meta["final_risk_1p0"].to_numpy(dtype=bool)
    no_nf = meta["no_internal_nonfinite"].to_numpy(dtype=bool)

    base_alarm = first_alarm | validity
    sys_base = binary_metrics(y_system, base_alarm)
    second_base = binary_metrics(y_final[no_nf], validity[no_nf])
    learned_pos_total = int((y_final & eligible).sum())
    learned_neg_total = int(((~y_final) & eligible).sum())

    def make_row(threshold: float, cum_pos: int, cum_neg: int, learned_alarm_count: int) -> Dict[str, Any]:
        sys_m = metrics_from_counts(
            sys_base["tp"] + cum_pos,
            sys_base["fn"] - cum_pos,
            sys_base["fp"] + cum_neg,
            sys_base["tn"] - cum_neg,
        )
        second_m = metrics_from_counts(
            second_base["tp"] + cum_pos,
            second_base["fn"] - cum_pos,
            second_base["fp"] + cum_neg,
            second_base["tn"] - cum_neg,
        )
        learned_m = metrics_from_counts(
            cum_pos, learned_pos_total - cum_pos,
            cum_neg, learned_neg_total - cum_neg,
        )
        return {
            "model_variant": model_variant,
            "positive_weight": float(positive_weight),
            "threshold": float(threshold),
            "learned_alarm_count": int(learned_alarm_count),
            **prefix_metrics("system", sys_m),
            **prefix_metrics("second_layer", second_m),
            **prefix_metrics("learned_branch", learned_m),
        }

    active_idx = np.where(eligible)[0]
    order = active_idx[np.argsort(-learned_score[active_idx], kind="mergesort")]
    rows = [make_row(float("inf"), 0, 0, 0)]
    cum_pos = 0
    cum_neg = 0
    pos = 0
    while pos < len(order):
        score_val = learned_score[order[pos]]
        end = pos + 1
        while end < len(order) and learned_score[order[end]] == score_val:
            end += 1
        idx = order[pos:end]
        group_pos = int(y_final[idx].sum())
        group_neg = int(len(idx) - group_pos)
        cum_pos += group_pos
        cum_neg += group_neg
        rows.append(make_row(float(score_val), cum_pos, cum_neg, cum_pos + cum_neg))
        pos = end
    return pd.DataFrame(rows)


def criterion_spec(criterion: str) -> Dict[str, Any]:
    if criterion == "system_target_recall_ge_0.99_far_le_0.001":
        return {"scope": "system", "mode": "joint", "recall": 0.99, "far": 0.001}
    m = re.fullmatch(r"best_(system|learned)_recall_at_far_le_(0\.\d+)", criterion)
    if m:
        scope = "system" if m.group(1) == "system" else "learned_branch"
        return {"scope": scope, "mode": "far_budget", "far": float(m.group(2))}
    m = re.fullmatch(r"min_(system|learned)_far_at_recall_ge_(0\.\d+)", criterion)
    if m:
        scope = "system" if m.group(1) == "system" else "learned_branch"
        return {"scope": scope, "mode": "recall_target", "recall": float(m.group(2))}
    raise ValueError(f"Unknown criterion: {criterion}")


def criterion_met(metrics: Mapping[str, Any], criterion: str) -> int:
    spec = criterion_spec(criterion)
    scope = spec["scope"]
    recall = float(metrics[f"{scope}_recall"])
    far = float(metrics[f"{scope}_far"])
    if spec["mode"] == "joint":
        return int(recall >= spec["recall"] - 1e-15 and far <= spec["far"] + 1e-15)
    if spec["mode"] == "far_budget":
        return int(far <= spec["far"] + 1e-15)
    return int(recall >= spec["recall"] - 1e-15)


def choose_curve_point(curve: pd.DataFrame, criterion: str) -> Dict[str, Any]:
    spec = criterion_spec(criterion)
    scope = spec["scope"]
    rcol = f"{scope}_recall"
    fcol = f"{scope}_far"
    c = curve.copy()
    if spec["mode"] == "joint":
        feasible = c[(c[rcol] >= spec["recall"] - 1e-15) & (c[fcol] <= spec["far"] + 1e-15)]
        met = not feasible.empty
        if met:
            chosen = feasible.sort_values([fcol, rcol, "threshold"], ascending=[True, False, False]).iloc[0]
        else:
            under_far = c[c[fcol] <= spec["far"] + 1e-15]
            chosen = (under_far.sort_values([rcol, fcol, "threshold"], ascending=[False, True, False]).iloc[0]
                      if not under_far.empty else c.sort_values([fcol, rcol], ascending=[True, False]).iloc[0])
    elif spec["mode"] == "far_budget":
        feasible = c[c[fcol] <= spec["far"] + 1e-15]
        met = not feasible.empty
        chosen = (feasible.sort_values([rcol, fcol, "threshold"], ascending=[False, True, False]).iloc[0]
                  if met else c.sort_values([fcol, rcol], ascending=[True, False]).iloc[0])
    else:
        feasible = c[c[rcol] >= spec["recall"] - 1e-15]
        met = not feasible.empty
        chosen = (feasible.sort_values([fcol, rcol, "threshold"], ascending=[True, False, False]).iloc[0]
                  if met else c.sort_values([rcol, fcol], ascending=[False, True]).iloc[0])
    out = chosen.to_dict()
    out["criterion"] = criterion
    out["target_met_on_validation"] = int(met)
    return out


def evaluate_threshold(meta: pd.DataFrame, score: np.ndarray, threshold: float) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any], Dict[str, Any]]:
    first_alarm = meta["internal_nonfinite_seen"].to_numpy(dtype=bool)
    validity = meta["validity_alarm"].to_numpy(dtype=bool)
    eligible = meta["learned_eligible"].to_numpy(dtype=bool)
    learned_alarm = eligible & (score >= threshold)
    system_alarm = first_alarm | validity | learned_alarm
    y_system = meta["system_y"].to_numpy(dtype=bool)
    system_m = binary_metrics(y_system, system_alarm)
    no_nf = meta["no_internal_nonfinite"].to_numpy(dtype=bool)
    second_m = binary_metrics(meta.loc[no_nf, "final_risk_1p0"].to_numpy(dtype=bool), (validity | learned_alarm)[no_nf])
    learned_m = binary_metrics(meta.loc[eligible, "final_risk_1p0"].to_numpy(dtype=bool), learned_alarm[eligible])
    return system_m, system_alarm, second_m, learned_m


def select_c2_training_indices(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c1_score: np.ndarray,
    feature_names: Sequence[str],
    hard_negative_count: int,
    random_negative_count: int,
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if hard_negative_count < 0 or random_negative_count < 0:
        raise ValueError("C2 negative counts must be >= 0")
    pos_idx = np.where(y_train > 0)[0]
    neg_idx = np.where(y_train == 0)[0]
    if pos_idx.size == 0 or neg_idx.size == 0:
        raise RuntimeError("C2 selection requires both classes")
    trigger_name = "feature__repair_trigger_count"
    if trigger_name not in feature_names:
        raise RuntimeError(f"Required repair feature missing: {trigger_name}")
    trigger_col = list(feature_names).index(trigger_name)
    repair_neg = neg_idx[X_train[neg_idx, trigger_col] > 0]
    used = set(int(x) for x in repair_neg)
    remaining = np.asarray([int(x) for x in neg_idx if int(x) not in used], dtype=np.int64)
    hard_n = min(int(hard_negative_count), int(remaining.size))
    if hard_n > 0:
        order = remaining[np.argsort(-c1_score[remaining], kind="mergesort")]
        hard_neg = order[:hard_n]
    else:
        hard_neg = np.empty(0, dtype=np.int64)
    used.update(int(x) for x in hard_neg)
    remaining2 = np.asarray([int(x) for x in remaining if int(x) not in used], dtype=np.int64)
    random_n = min(int(random_negative_count), int(remaining2.size))
    rng = np.random.default_rng(int(random_state))
    random_neg = (rng.choice(remaining2, size=random_n, replace=False).astype(np.int64)
                  if random_n > 0 else np.empty(0, dtype=np.int64))
    selected = np.concatenate([pos_idx.astype(np.int64), repair_neg.astype(np.int64), hard_neg, random_neg])
    selected = selected[rng.permutation(len(selected))]
    audit = {
        "all_positive": int(pos_idx.size),
        "all_negative": int(neg_idx.size),
        "repair_triggered_safe_negative": int(repair_neg.size),
        "c1_hard_negative": int(hard_neg.size),
        "random_background_negative": int(random_neg.size),
        "selected_rows": int(selected.size),
        "selected_positive": int((y_train[selected] > 0).sum()),
        "selected_negative": int((y_train[selected] == 0).sum()),
        "hard_negative_requested": int(hard_negative_count),
        "random_negative_requested": int(random_negative_count),
        "selection_random_state": int(random_state),
    }
    return selected, audit


def build_model(max_iter: int, learning_rate: float, max_leaf_nodes: int, min_samples_leaf: int, l2: float, random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=float(learning_rate), max_iter=int(max_iter),
        max_leaf_nodes=int(max_leaf_nodes), min_samples_leaf=int(min_samples_leaf),
        l2_regularization=float(l2), max_bins=255, early_stopping=False,
        random_state=int(random_state), verbose=0,
    )


def fit_with_optional_positive_weight(model: HistGradientBoostingClassifier, X: np.ndarray, y: np.ndarray, positive_weight: float) -> HistGradientBoostingClassifier:
    if len(np.unique(y)) != 2:
        raise RuntimeError(f"Training labels must contain both classes, got {np.unique(y)}")
    if positive_weight <= 0:
        raise ValueError("positive_weight must be > 0")
    sample_weight = None
    if abs(float(positive_weight) - 1.0) > 1e-12:
        sample_weight = np.where(y > 0, float(positive_weight), 1.0).astype(np.float64)
    return model.fit(X, y, sample_weight=sample_weight)


def population_summary(split: SplitData) -> Dict[str, Any]:
    m = split.meta
    no_nf = m["no_internal_nonfinite"] > 0
    elig = m["learned_eligible"] > 0
    return {
        "split": split.name, "seeds": split.seeds, "rows": len(m),
        "internal_nonfinite": int((m["internal_nonfinite_seen"] > 0).sum()),
        "no_internal_nonfinite": int(no_nf.sum()),
        "no_nf_final_risk": int(((m["final_risk_1p0"] > 0) & no_nf).sum()),
        "no_nf_final_safe": int(((m["final_risk_1p0"] == 0) & no_nf).sum()),
        "no_nf_validity_alarm": int(((m["validity_alarm"] > 0) & no_nf).sum()),
        "learned_eligible": int(elig.sum()),
        "learned_positive": int(((m["final_risk_1p0"] > 0) & elig).sum()),
        "learned_negative": int(((m["final_risk_1p0"] == 0) & elig).sum()),
        "fault_effective": int((m["fault_effective"] > 0).sum()),
        "planned_but_ineffective": int(((m["fault_plan_target_present"] > 0) & (m["fault_effective"] == 0)).sum()),
        "normal_control": int((m["fault_plan_target_present"] == 0).sum()),
        "system_positive": int((m["system_y"] > 0).sum()),
        "system_negative": int((m["system_y"] == 0).sum()),
    }


def preflight(args: argparse.Namespace, activation_features: Sequence[str], contract_audit: Mapping[str, Any], out_dir: Path) -> None:
    all_seeds = args.train_seeds + args.val_seeds + args.test_seeds
    rows = []
    for seed in all_seeds:
        d = one_seed_dir(args.data_root, seed)
        csv_path = d / "detector_train_activation_only_v4_3_1.csv"
        event_path = d / "debug_repair_events_v4_3_1.jsonl"
        debug_path = d / "debug_metadata_v4_3_1.jsonl"
        if not csv_path.is_file() or not event_path.is_file() or not debug_path.is_file():
            raise FileNotFoundError(f"seed={seed}: required material missing in {d}")
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
        missing = [c for c in list(activation_features) + REQUIRED_NONFEATURE_COLUMNS if c not in header]
        rows.append({
            "seed": seed, "seed_dir": str(d), "header_columns": len(header),
            "required_missing_count": len(missing), "missing_preview": ";".join(missing[:20]),
            "csv_size_bytes": csv_path.stat().st_size, "event_size_bytes": event_path.stat().st_size,
            "debug_size_bytes": debug_path.stat().st_size,
        })
        if missing:
            raise RuntimeError(f"seed={seed}: missing columns: {missing[:30]}")
    pd.DataFrame(rows).to_csv(out_dir / "preflight_seed_material_audit_v4_4_9_2.csv", index=False)
    write_json(out_dir / "preflight_contract_audit_v4_4_9_2.json", dict(contract_audit))
    write_json(out_dir / "preflight_status_v4_4_9_2.json", {"status": "PASS", "version": VERSION, "seed_count": len(all_seeds)})
    print(f"[{VERSION}] PREFLIGHT PASS: seeds={len(all_seeds)} activation_features={len(activation_features)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--frozen-contract-json", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-seeds", type=parse_seed_list, default=DEFAULT_TRAIN_SEEDS)
    p.add_argument("--val-seeds", type=parse_seed_list, default=DEFAULT_VAL_SEEDS)
    p.add_argument("--test-seeds", type=parse_seed_list, default=DEFAULT_TEST_SEEDS)
    p.add_argument("--expected-activation-features", type=int, default=5449)
    p.add_argument("--expected-reviewed-features", type=int, default=4780)
    p.add_argument("--expected-runtime-features", type=int, default=669)
    p.add_argument("--expected-legacy-repair-features", type=int, default=202)
    p.add_argument("--legacy-repair-prefix", type=str, default="feature__repair_")
    p.add_argument("--c1-positive-weight", type=float, default=1.0)
    p.add_argument("--c2-positive-weights", type=parse_float_list, default=[1.0, 2.0, 3.0, 5.0, 10.0])
    p.add_argument("--c2-hard-negative-count", type=int, default=8000)
    p.add_argument("--c2-random-negative-count", type=int, default=8000)
    p.add_argument("--random-state", type=int, default=4492)
    p.add_argument("--c1-max-iter", type=int, default=220)
    p.add_argument("--c2-max-iter", type=int, default=180)
    p.add_argument("--learning-rate", type=float, default=0.06)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--min-samples-leaf", type=int, default=20)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--max-rows-per-seed", type=int, default=0)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()

    args.data_root = args.data_root.resolve()
    args.frozen_contract_json = args.frozen_contract_json.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = args.out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    flat = [x for s0 in [args.train_seeds, args.val_seeds, args.test_seeds] for x in s0]
    if len(flat) != len(set(flat)):
        raise RuntimeError(f"Seed splits overlap: train={args.train_seeds} val={args.val_seeds} test={args.test_seeds}")

    activation_features, contract_audit = resolve_activation_contract(
        args.frozen_contract_json, args.expected_activation_features,
        args.expected_reviewed_features, args.expected_runtime_features,
        args.expected_legacy_repair_features, args.legacy_repair_prefix,
    )
    model_activation_features = [x for x in activation_features if x != INTERNAL_NF_COL]
    all_model_features = model_activation_features + TELEMETRY_FEATURES
    if len(all_model_features) != len(set(all_model_features)):
        raise RuntimeError("Duplicate model feature names after activation+repair merge")
    feature_contract = {
        "version": VERSION,
        "source_activation_feature_count": len(activation_features),
        "internal_nonfinite_route_feature_removed_from_model": INTERNAL_NF_COL,
        "model_activation_feature_count": len(model_activation_features),
        "repair_telemetry_feature_count": len(TELEMETRY_FEATURES),
        "total_model_feature_count": len(all_model_features),
        "activation_feature_sha256": sha256_text_lines(activation_features),
        "model_feature_sha256": sha256_text_lines(all_model_features),
        "activation_features": activation_features,
        "repair_telemetry_features": TELEMETRY_FEATURES,
        "model_features": all_model_features,
        "source_contract_audit": contract_audit,
    }
    write_json(args.out_dir / "fixed_exact_feature_contract_v4_4_9_2.json", feature_contract)
    pd.DataFrame({"feature": TELEMETRY_FEATURES}).to_csv(args.out_dir / "repair_telemetry_feature_contract_v4_4_9_2.csv", index=False)

    preflight(args, activation_features, contract_audit, args.out_dir)
    if args.preflight_only:
        print(f"OUT_DIR={args.out_dir}", flush=True)
        return

    started = time.time()
    train, audit_train = load_split("train", args.train_seeds, args.data_root, activation_features, cache_dir, args.max_rows_per_seed)
    val, audit_val = load_split("validation", args.val_seeds, args.data_root, activation_features, cache_dir, args.max_rows_per_seed)
    test, audit_test = load_split("test", args.test_seeds, args.data_root, activation_features, cache_dir, args.max_rows_per_seed)
    route_idx = activation_features.index(INTERNAL_NF_COL)
    train.X = np.delete(train.X, route_idx, axis=1)
    val.X = np.delete(val.X, route_idx, axis=1)
    test.X = np.delete(test.X, route_idx, axis=1)
    if train.X.shape[1] != len(all_model_features):
        raise RuntimeError(f"Feature shape mismatch: X={train.X.shape[1]} contract={len(all_model_features)}")

    audits = audit_train + audit_val + audit_test
    pd.json_normalize(audits).to_csv(args.out_dir / "per_seed_data_audit_v4_4_9_2.csv", index=False)
    pop = [population_summary(x) for x in [train, val, test]]
    pd.DataFrame(pop).to_csv(args.out_dir / "population_summary_v4_4_9_2.csv", index=False)
    write_json(args.out_dir / "population_summary_v4_4_9_2.json", pop)

    for split in [train, val, test]:
        bad = split.meta["learned_eligible"].to_numpy(dtype=bool) & split.meta["internal_nonfinite_seen"].to_numpy(dtype=bool)
        if int(bad.sum()) != 0:
            raise RuntimeError(f"{split.name}: internal-nonfinite rows leaked into learned eligibility: {int(bad.sum())}")

    train_eligible = train.meta["learned_eligible"].to_numpy(dtype=bool)
    val_eligible = val.meta["learned_eligible"].to_numpy(dtype=bool)
    test_eligible = test.meta["learned_eligible"].to_numpy(dtype=bool)
    X_train = train.X[train_eligible]
    y_train = train.meta.loc[train_eligible, "final_risk_1p0"].to_numpy(dtype=np.int8)

    print(f"[{VERSION}] fit C1 all-eligible rows={len(y_train)} pos={int(y_train.sum())} neg={int((y_train==0).sum())} features={X_train.shape[1]} pw={args.c1_positive_weight}", flush=True)
    c1 = build_model(args.c1_max_iter, args.learning_rate, args.max_leaf_nodes, args.min_samples_leaf, args.l2, args.random_state)
    fit_with_optional_positive_weight(c1, X_train, y_train, args.c1_positive_weight)
    joblib.dump(c1, args.out_dir / "c1_model_v4_4_9_2.joblib", compress=3)

    def c1_scores(split: SplitData) -> np.ndarray:
        eligible = split.meta["learned_eligible"].to_numpy(dtype=bool)
        out = np.zeros(len(split.meta), dtype=np.float64)
        if int(eligible.sum()) > 0:
            out[eligible] = positive_proba(c1, split.X[eligible])
        return out

    c1_train_e = positive_proba(c1, X_train)
    c1_val = c1_scores(val)
    c1_test = c1_scores(test)

    c2_idx, c2_subset_audit = select_c2_training_indices(
        X_train, y_train, c1_train_e, all_model_features,
        args.c2_hard_negative_count, args.c2_random_negative_count,
        args.random_state + 20,
    )
    X_c2_train = np.concatenate([X_train[c2_idx], c1_train_e[c2_idx, None].astype(np.float32)], axis=1)
    y_c2_train = y_train[c2_idx]
    pd.DataFrame([c2_subset_audit]).to_csv(args.out_dir / "c2_training_subset_audit_v4_4_9_2.csv", index=False)
    write_json(args.out_dir / "c2_training_subset_audit_v4_4_9_2.json", c2_subset_audit)
    print(f"[{VERSION}] C2 subset rows={len(y_c2_train)} pos={int(y_c2_train.sum())} neg={int((y_c2_train==0).sum())}", flush=True)

    def score_all_eligible(model: HistGradientBoostingClassifier, split: SplitData, c1_score: np.ndarray) -> np.ndarray:
        eligible = split.meta["learned_eligible"].to_numpy(dtype=bool)
        out = np.zeros(len(split.meta), dtype=np.float64)
        if int(eligible.sum()) > 0:
            X2 = np.concatenate([split.X[eligible], c1_score[eligible, None].astype(np.float32)], axis=1)
            out[eligible] = positive_proba(model, X2)
        return out

    criteria = [
        "system_target_recall_ge_0.99_far_le_0.001",
        "best_system_recall_at_far_le_0.001",
        "best_system_recall_at_far_le_0.002",
        "best_system_recall_at_far_le_0.005",
        "best_system_recall_at_far_le_0.01",
        "best_system_recall_at_far_le_0.02",
        "min_system_far_at_recall_ge_0.99",
        "best_learned_recall_at_far_le_0.001",
        "best_learned_recall_at_far_le_0.002",
        "best_learned_recall_at_far_le_0.005",
        "best_learned_recall_at_far_le_0.01",
        "best_learned_recall_at_far_le_0.02",
        "min_learned_far_at_recall_ge_0.80",
        "min_learned_far_at_recall_ge_0.8272",
        "min_learned_far_at_recall_ge_0.85",
        "min_learned_far_at_recall_ge_0.90",
    ]

    c1_val_curve = score_threshold_curve(val.meta, c1_val, "c1_only", args.c1_positive_weight)
    c1_val_curve.to_csv(args.out_dir / "validation_c1_only_threshold_curve_v4_4_9_2.csv", index=False)
    c1_selected = [choose_curve_point(c1_val_curve, c) for c in criteria]
    pd.DataFrame(c1_selected).to_csv(args.out_dir / "validation_c1_only_selected_thresholds_v4_4_9_2.csv", index=False)

    val_score_by_variant: Dict[str, np.ndarray] = {}
    test_score_by_variant: Dict[str, np.ndarray] = {}
    c2_model_files: Dict[str, str] = {}
    val_curves: List[pd.DataFrame] = []
    for wi, weight in enumerate(args.c2_positive_weights):
        tag = str(weight).replace(".", "p")
        variant = f"c2_pw{tag}"
        print(f"[{VERSION}] fit {variant} rows={len(y_c2_train)} pos_weight={weight}", flush=True)
        c2 = build_model(args.c2_max_iter, args.learning_rate, args.max_leaf_nodes, args.min_samples_leaf, args.l2, args.random_state + 100)
        fit_with_optional_positive_weight(c2, X_c2_train, y_c2_train, weight)
        model_file = f"c2_model_{variant}_v4_4_9_2.joblib"
        joblib.dump(c2, args.out_dir / model_file, compress=3)
        c2_model_files[variant] = model_file
        val_score = score_all_eligible(c2, val, c1_val)
        test_score = score_all_eligible(c2, test, c1_test)
        val_score_by_variant[variant] = val_score
        test_score_by_variant[variant] = test_score
        curve = score_threshold_curve(val.meta, val_score, variant, weight)
        val_curves.append(curve)
        del c2

    all_val_curves = pd.concat(val_curves, ignore_index=True)
    all_val_curves.to_csv(args.out_dir / "validation_c2_weight_sweep_threshold_curves_v4_4_9_2.csv", index=False)

    per_variant_validation: List[Dict[str, Any]] = []
    per_variant_test: List[Dict[str, Any]] = []
    for variant, curve in all_val_curves.groupby("model_variant", sort=True):
        test_score = test_score_by_variant[str(variant)]
        for criterion in criteria:
            row = choose_curve_point(curve, criterion)
            threshold = float(row["threshold"])
            vm, _, v2m, vlm = evaluate_threshold(val.meta, val_score_by_variant[str(variant)], threshold)
            tm, _, t2m, tlm = evaluate_threshold(test.meta, test_score, threshold)
            base = {"model_variant": str(variant), "positive_weight": float(row["positive_weight"]), "criterion": criterion, "threshold_selected_on_validation": threshold}
            per_variant_validation.append({**base, "validation_target_met": int(row["target_met_on_validation"]), **prefix_metrics("system", vm), **prefix_metrics("second_layer", v2m), **prefix_metrics("learned_branch", vlm)})
            test_payload = {**prefix_metrics("system", tm), **prefix_metrics("second_layer", t2m), **prefix_metrics("learned_branch", tlm)}
            per_variant_test.append({**base, "validation_target_met": int(row["target_met_on_validation"]), "test_target_met": criterion_met(test_payload, criterion), **test_payload})
    pd.DataFrame(per_variant_validation).to_csv(args.out_dir / "c2_weight_sweep_validation_metrics_v4_4_9_2.csv", index=False)
    pd.DataFrame(per_variant_test).to_csv(args.out_dir / "c2_weight_sweep_test_metrics_v4_4_9_2.csv", index=False)

    selected = [choose_curve_point(all_val_curves, c) for c in criteria]
    pd.DataFrame(selected).to_csv(args.out_dir / "validation_selected_c2_policies_v4_4_9_2.csv", index=False)

    validation_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    val_pred = val.meta.copy()
    test_pred = test.meta.copy()
    val_pred["c1_score"] = c1_val
    test_pred["c1_score"] = c1_test
    val_pred["c2_scored"] = val_eligible.astype(np.int8)
    test_pred["c2_scored"] = test_eligible.astype(np.int8)
    for variant, score in val_score_by_variant.items():
        val_pred[f"score__{variant}"] = score
    for variant, score in test_score_by_variant.items():
        test_pred[f"score__{variant}"] = score

    for row in selected:
        criterion = str(row["criterion"])
        variant = str(row["model_variant"])
        threshold = float(row["threshold"])
        vm, val_alarm, v2m, vlm = evaluate_threshold(val.meta, val_score_by_variant[variant], threshold)
        tm, test_alarm, t2m, tlm = evaluate_threshold(test.meta, test_score_by_variant[variant], threshold)
        validation_rows.append({
            "criterion": criterion, "selected_model_variant": variant,
            "selected_positive_weight": float(row["positive_weight"]),
            "threshold_selected_on_validation": threshold,
            "validation_target_met": int(row["target_met_on_validation"]),
            **prefix_metrics("system", vm), **prefix_metrics("second_layer", v2m), **prefix_metrics("learned_branch", vlm),
        })
        test_payload = {**prefix_metrics("system", tm), **prefix_metrics("second_layer", t2m), **prefix_metrics("learned_branch", tlm)}
        test_rows.append({
            "criterion": criterion, "selected_model_variant": variant,
            "selected_positive_weight": float(row["positive_weight"]),
            "threshold_selected_on_validation": threshold,
            "validation_target_met": int(row["target_met_on_validation"]),
            "test_target_met": criterion_met(test_payload, criterion),
            **test_payload,
        })
        safe = criterion.replace(".", "p")
        val_pred[f"alarm__{safe}"] = val_alarm.astype(np.int8)
        test_pred[f"alarm__{safe}"] = test_alarm.astype(np.int8)

    pd.DataFrame(validation_rows).to_csv(args.out_dir / "fixed_split_validation_metrics_v4_4_9_2.csv", index=False)
    pd.DataFrame(test_rows).to_csv(args.out_dir / "fixed_split_test_metrics_v4_4_9_2.csv", index=False)
    val_pred.to_csv(args.out_dir / "fixed_split_validation_predictions_v4_4_9_2.csv", index=False)
    test_pred.to_csv(args.out_dir / "fixed_split_test_predictions_v4_4_9_2.csv", index=False)

    c1_test_rows: List[Dict[str, Any]] = []
    for row in c1_selected:
        criterion = str(row["criterion"])
        threshold = float(row["threshold"])
        tm, _, t2m, tlm = evaluate_threshold(test.meta, c1_test, threshold)
        payload = {**prefix_metrics("system", tm), **prefix_metrics("second_layer", t2m), **prefix_metrics("learned_branch", tlm)}
        c1_test_rows.append({
            "criterion": criterion, "threshold_selected_on_validation": threshold,
            "validation_target_met": int(row["target_met_on_validation"]),
            "test_target_met": criterion_met(payload, criterion), **payload,
        })
    pd.DataFrame(c1_test_rows).to_csv(args.out_dir / "c1_only_test_metrics_v4_4_9_2.csv", index=False)

    model_manifest = {
        "version": VERSION,
        "no_hard_gate": True,
        "all_learned_eligible_scored_by_c2": True,
        "feature_count_c1": len(all_model_features),
        "feature_count_c2": len(all_model_features) + 1,
        "c1_model": "c1_model_v4_4_9_2.joblib",
        "c1_positive_weight": args.c1_positive_weight,
        "c2_models": c2_model_files,
        "c2_positive_weights": args.c2_positive_weights,
        "learned_training_rows": len(y_train),
        "learned_training_positive": int(y_train.sum()),
        "c2_training_subset": c2_subset_audit,
    }
    write_json(args.out_dir / "model_manifest_v4_4_9_2.json", model_manifest)
    run_config = {
        "version": VERSION,
        "task_definition": {
            "layer1": "internal_nonfinite_seen -> intercepted and excluded from learned detector regardless final safety",
            "layer2_validity": "no_internal_nonfinite and deterministic invalid/gate reject -> direct alarm",
            "layer2_learned": "all finite/legal/gate-accepted no-NF rows -> C1 score -> C2 final score; no hard gate",
            "learned_label": FINAL_RISK_COL,
        },
        "no_oof": True, "no_nested_cv": True, "no_hard_gate": True,
        "validation_selects_model_weight_and_threshold": True,
        "test_used_for_selection": False,
        "train_seeds": args.train_seeds, "validation_seeds": args.val_seeds, "test_seeds": args.test_seeds,
        "parameters": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
        "paths": {"data_root": str(args.data_root), "frozen_contract_json": str(args.frozen_contract_json), "out_dir": str(args.out_dir)},
        "elapsed_sec": time.time() - started,
        "feature_contract_sha256": sha256_text_lines(all_model_features),
    }
    write_json(args.out_dir / "run_config_v4_4_9_2.json", run_config)
    summary = {
        "status": "DONE", "version": VERSION, "out_dir": str(args.out_dir),
        "population": pop, "c2_training_subset": c2_subset_audit,
        "validation_selected_policies": validation_rows, "test_metrics": test_rows,
        "elapsed_sec": time.time() - started,
    }
    write_json(args.out_dir / "summary_v4_4_9_2.json", summary)
    print(f"[{VERSION}] DONE elapsed={time.time()-started:.1f}s", flush=True)
    print(f"OUT_DIR={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
