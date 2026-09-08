#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build strictly online repair telemetry features for UniAD2 v4.4.6.

Only fields produced during the current inference/repair execution are read.
No clean-reference, GT, trajectory delta, gate decision, injected fault plan,
bit metadata, or post-hoc safety label is used.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EVENTS_NAME = "debug_repair_events_v4_3_1.jsonl"
DEBUG_SUMMARY_NAME = "debug_case_summaries_v4_3_1.csv"
PREFIX = "feature__repair_"

GROUPS = [
    "U1_img_backbone_neck",
    "U2_bev_encoder",
    "U3_track_det_head",
    "U4_map_seg_head",
    "U5_motion_head",
    "U6_occ_head",
    "U7_planning_head",
    "U9_online_track_memory",
    "unknown",
]
ROLES = [
    "attention",
    "transformer_layer",
    "planning_head",
    "ffn",
    "projection_or_head",
    "generic",
]
ACTIONS = [
    "clip_median",
    "finite_clip",
    "zero_tensor",
    "input_passthrough",
    "standard_repair",
    "budget_exhausted",
    "no_modification",
    "other",
]
BANDS = ["none", "low", "medium", "high", "unknown"]

# Exact fields read from event JSONL. The file may contain forbidden/debug fields,
# but this module never reads or transforms them.
EVENT_FIELD_ALLOWLIST = {
    "sample_i",
    "sample_id",
    "trace_order",
    "module_name",
    "module_class",
    "uniad_group",
    "repair_triggered",
    "repair_modified",
    "repair_reason",
    "repair_method",
    "max_repair_count_per_sample",
    "repair_modified_count_so_far",
    "repair_idx",
    "pre_repair_finite_ratio_sample",
    "pre_repair_absmax_finite_sample",
    "pre_repair_nonfinite_count_sample",
    "pre_repair_sample_numel",
    "nonfinite_count_sample",
    "huge_count_sample",
    "abnormal_count_sample",
    "nonfinite_ratio_sample",
    "huge_ratio_sample",
    "abnormal_ratio_sample",
    "absmax_ratio_sample",
    "contamination_band_sample",
    "full_check_on_trigger",
    "finite_ratio_full",
    "absmax_finite_full",
    "nonfinite_count_full",
    "huge_count_full",
    "abnormal_count_full",
    "nonfinite_ratio_full",
    "huge_ratio_full",
    "abnormal_ratio_full",
    "absmax_ratio_full",
    "contamination_band_full",
    "post_repair_finite_ratio_full",
    "post_repair_absmax_finite_full",
    "post_repair_nonfinite_count_full",
    "post_repair_huge_count_full",
    "post_repair_abnormal_count_full",
    "post_repair_abnormal_ratio_full",
    "post_repair_absmax_ratio_full",
    "post_repair_contamination_band_full",
    "replace_output_ok",
    "high_contam_policy_flag",
    "high_contam_action",
    "module_aware_role",
}

DEBUG_SUMMARY_FIELD_ALLOWLIST = {
    "sample_i",
    "sample_id",
    "final_trace_order",
    "max_repair_count_per_sample",
}

FORBIDDEN_SOURCE_TOKENS = (
    "target_",
    "fault_",
    "bit_",
    "clean_",
    "delta_",
    "harmful",
    "severe",
    "label_",
    "gate_",
    "true_fault",
    "detection_distance",
)


def _slug(text: str) -> str:
    x = re.sub(r"[^0-9a-zA-Z]+", "_", str(text).strip()).strip("_").lower()
    return x or "unknown"


def _finite_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _int01(v: Any) -> int:
    try:
        return int(float(v) != 0.0)
    except Exception:
        return 0


def _nonneg_int(v: Any, default: int = 0) -> int:
    try:
        return max(0, int(float(v)))
    except Exception:
        return default


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) > 0 else 0.0


def canonical_group(v: Any) -> str:
    s = str(v or "").strip()
    return s if s in GROUPS else "unknown"


def repair_role(group: Any, module_class: Any, module_name: Any, explicit: Any = "") -> str:
    ex = str(explicit or "").strip()
    if ex in ROLES:
        return ex
    text = " ".join([str(group or ""), str(module_class or ""), str(module_name or "")]).lower()
    if "planning" in text or str(group) == "U7_planning_head":
        return "planning_head"
    if "multiheadattention" in text or "multi_head_attention" in text or "attention" in text or "attn" in text:
        return "attention"
    if "ffn" in text or "feedforward" in text or "feed_forward" in text:
        return "ffn"
    if (
        "detrtransformerdecoderlayer" in text
        or "decoderlayer" in text
        or "bevformerlayer" in text
        or "transformerlayers" in text
        or "transformerlayer" in text
    ):
        return "transformer_layer"
    if "linear" in text or "conv" in text or "mlp" in text or "projection" in text:
        return "projection_or_head"
    return "generic"


def action_category(reason: Any, high_action: Any = "") -> str:
    text = (str(reason or "") + " " + str(high_action or "")).lower()
    if "budget_exhausted" in text:
        return "budget_exhausted"
    if "input_passthrough" in text or "rollback" in text:
        return "input_passthrough"
    if "zero_tensor" in text or "zero_containment" in text or "zero_contribution" in text:
        return "zero_tensor"
    if "imputed_with_tensor_median" in text or ("median" in text and "clip" in text):
        return "clip_median"
    if "finite_extremes_clipped" in text or "finite_clip" in text:
        return "finite_clip"
    if "standard_repair" in text or "standard_fallback" in text:
        return "standard_repair"
    if (
        "no_modification" in text
        or "not_modified" in text
        or "cannot_restore" in text
        or "not_repairable" in text
        or "no_violation" in text
    ):
        return "no_modification"
    return "other"


def canonical_band(v: Any) -> str:
    s = str(v or "").strip().lower()
    return s if s in BANDS else "unknown"


def repair_feature_names() -> List[str]:
    names: List[str] = [
        "event_count",
        "triggered_count",
        "modified_count",
        "nonfinite_event_count",
        "huge_event_count",
        "any_event",
        "any_triggered",
        "any_modified",
        "no_modify_event_count",
        "trigger_no_modify_count",
        "unique_module_count",
        "unique_class_count",
        "unique_group_count",
        "unique_role_count",
        "first_trace_order",
        "last_trace_order",
        "trace_span",
        "first_trace_fraction",
        "last_trace_fraction",
        "trace_span_fraction",
        "budget_max",
        "budget_used_ratio",
        "budget_saturated",
        "budget_exhausted_event_count",
        "events_after_first_modified_count",
        "modified_after_first_modified_count",
        "post_stats_event_count",
        "post_nonfinite_remaining_event_count",
        "nonfinite_eliminated_event_count",
        "post_huge_remaining_event_count",
        "trigger_per_event",
        "modified_per_event",
        "modified_per_trigger",
        "nonfinite_per_event",
        "huge_per_event",
        "no_modify_per_event",
        "pre_finite_ratio_sample_min",
        "pre_finite_ratio_sample_mean",
        "pre_abnormal_ratio_sample_max",
        "pre_abnormal_ratio_sample_mean",
        "pre_nonfinite_ratio_sample_max",
        "pre_nonfinite_ratio_sample_mean",
        "pre_huge_ratio_sample_max",
        "pre_huge_ratio_sample_mean",
        "pre_absmax_ratio_sample_max",
        "pre_absmax_ratio_sample_mean",
        "pre_finite_ratio_full_min",
        "pre_finite_ratio_full_mean",
        "pre_abnormal_ratio_full_max",
        "pre_abnormal_ratio_full_mean",
        "pre_nonfinite_ratio_full_max",
        "pre_nonfinite_ratio_full_mean",
        "pre_huge_ratio_full_max",
        "pre_huge_ratio_full_mean",
        "pre_absmax_ratio_full_max",
        "pre_absmax_ratio_full_mean",
        "post_finite_ratio_full_min",
        "post_finite_ratio_full_mean",
        "post_abnormal_ratio_full_max",
        "post_abnormal_ratio_full_mean",
        "post_absmax_ratio_full_max",
        "post_absmax_ratio_full_mean",
        "post_nonfinite_count_full_sum",
        "post_nonfinite_count_full_max",
        "post_huge_count_full_sum",
        "post_huge_count_full_max",
        "finite_ratio_gain_full_mean",
        "finite_ratio_gain_full_max",
        "abnormal_ratio_reduction_full_mean",
        "abnormal_ratio_reduction_full_max",
        "absmax_ratio_reduction_full_mean",
        "absmax_ratio_reduction_full_max",
        "nonfinite_count_reduction_full_sum",
        "nonfinite_count_reduction_full_max",
        "huge_count_reduction_full_sum",
        "huge_count_reduction_full_max",
        "replace_output_failed_count",
        "high_contam_event_count",
    ]
    for g in GROUPS:
        s = _slug(g)
        names.extend([
            "group_%s_event_count" % s,
            "group_%s_modified_count" % s,
            "group_%s_event_ratio" % s,
            "group_%s_modified_ratio" % s,
            "first_group_%s" % s,
            "last_group_%s" % s,
        ])
    for r in ROLES:
        s = _slug(r)
        names.extend([
            "role_%s_event_count" % s,
            "role_%s_modified_count" % s,
            "role_%s_event_ratio" % s,
            "role_%s_modified_ratio" % s,
            "first_role_%s" % s,
            "last_role_%s" % s,
        ])
    for a in ACTIONS:
        s = _slug(a)
        names.extend([
            "action_%s_count" % s,
            "action_%s_ratio" % s,
            "first_action_%s" % s,
        ])
    for scope in ["sample", "full"]:
        for b in BANDS:
            names.append("first_contamination_%s_%s" % (scope, _slug(b)))
    return [PREFIX + n for n in names]


REPAIR_FEATURES = repair_feature_names()


def _list_stat(vals: Sequence[float], mode: str, default: float = 0.0) -> float:
    arr = np.asarray([x for x in vals if math.isfinite(float(x))], dtype=np.float64)
    if len(arr) == 0:
        return float(default)
    if mode == "min":
        return float(np.min(arr))
    if mode == "max":
        return float(np.max(arr))
    if mode == "mean":
        return float(np.mean(arr))
    if mode == "sum":
        return float(np.sum(arr))
    raise ValueError(mode)


def _new_state() -> Dict[str, Any]:
    return {
        "events": [],
        "groups": Counter(),
        "groups_modified": Counter(),
        "roles": Counter(),
        "roles_modified": Counter(),
        "actions": Counter(),
        "modules": set(),
        "classes": set(),
        "pre": defaultdict(list),
        "post": defaultdict(list),
        "improve": defaultdict(list),
    }


def _event_subset(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: raw.get(k) for k in EVENT_FIELD_ALLOWLIST}


def _read_summary_context(path: Path) -> Dict[int, Dict[str, Any]]:
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in ["sample_i", "sample_id"] if c not in header]
    if missing:
        raise RuntimeError("repair debug summary missing keys %s: %s" % (missing, path))
    use = [c for c in DEBUG_SUMMARY_FIELD_ALLOWLIST if c in header]
    df = pd.read_csv(path, usecols=use, low_memory=False)
    if df["sample_i"].duplicated().any():
        raise RuntimeError("duplicate sample_i in %s" % path)
    out: Dict[int, Dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        i = int(row["sample_i"])
        out[i] = {k: row.get(k) for k in use}
    return out


def build_repair_telemetry(
    events_path: Path,
    debug_summary_path: Path,
    sample_keys: pd.DataFrame,
    default_budget_max: int = 4,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate event-level repair telemetry into one row per sample."""
    if not events_path.is_file():
        raise FileNotFoundError(events_path)
    if not debug_summary_path.is_file():
        raise FileNotFoundError(debug_summary_path)
    if "sample_i" not in sample_keys.columns or "sample_id" not in sample_keys.columns:
        raise ValueError("sample_keys requires sample_i and sample_id")

    context = _read_summary_context(debug_summary_path)
    states: Dict[int, Dict[str, Any]] = {}
    event_rows = 0
    unknown_sample_events = 0
    sample_index = set(int(x) for x in sample_keys["sample_i"].tolist())

    with events_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except Exception as exc:
                raise RuntimeError("invalid JSON at %s:%d: %s" % (events_path, line_no, exc))
            try:
                sample_i = int(raw.get("sample_i"))
            except Exception:
                raise RuntimeError("event missing numeric sample_i at %s:%d" % (events_path, line_no))
            if sample_i not in sample_index:
                unknown_sample_events += 1
                continue
            ev = _event_subset(raw)
            st = states.setdefault(sample_i, _new_state())
            st["events"].append(ev)
            event_rows += 1

            group = canonical_group(ev.get("uniad_group"))
            role = repair_role(group, ev.get("module_class"), ev.get("module_name"), ev.get("module_aware_role"))
            action = action_category(ev.get("repair_reason"), ev.get("high_contam_action"))
            modified = _int01(ev.get("repair_modified"))
            st["groups"][group] += 1
            st["roles"][role] += 1
            st["actions"][action] += 1
            if modified:
                st["groups_modified"][group] += 1
                st["roles_modified"][role] += 1
            name = str(ev.get("module_name") or "").strip()
            cls = str(ev.get("module_class") or "").strip()
            if name:
                st["modules"].add(name)
            if cls:
                st["classes"].add(cls)

            pre_pairs = {
                "finite_ratio_sample": ev.get("pre_repair_finite_ratio_sample"),
                "abnormal_ratio_sample": ev.get("abnormal_ratio_sample"),
                "nonfinite_ratio_sample": ev.get("nonfinite_ratio_sample"),
                "huge_ratio_sample": ev.get("huge_ratio_sample"),
                "absmax_ratio_sample": ev.get("absmax_ratio_sample"),
                "finite_ratio_full": ev.get("finite_ratio_full"),
                "abnormal_ratio_full": ev.get("abnormal_ratio_full"),
                "nonfinite_ratio_full": ev.get("nonfinite_ratio_full"),
                "huge_ratio_full": ev.get("huge_ratio_full"),
                "absmax_ratio_full": ev.get("absmax_ratio_full"),
                "nonfinite_count_full": ev.get("nonfinite_count_full"),
                "huge_count_full": ev.get("huge_count_full"),
            }
            for key, value in pre_pairs.items():
                x = _finite_float(value)
                if math.isfinite(x):
                    st["pre"][key].append(x)

            post_pairs = {
                "finite_ratio_full": ev.get("post_repair_finite_ratio_full"),
                "abnormal_ratio_full": ev.get("post_repair_abnormal_ratio_full"),
                "absmax_ratio_full": ev.get("post_repair_absmax_ratio_full"),
                "nonfinite_count_full": ev.get("post_repair_nonfinite_count_full"),
                "huge_count_full": ev.get("post_repair_huge_count_full"),
            }
            has_post = False
            for key, value in post_pairs.items():
                x = _finite_float(value)
                if math.isfinite(x):
                    st["post"][key].append(x)
                    has_post = True
            if has_post:
                for key, pre_key, post_key, sign in [
                    ("finite_ratio_gain_full", "finite_ratio_full", "finite_ratio_full", 1.0),
                    ("abnormal_ratio_reduction_full", "abnormal_ratio_full", "abnormal_ratio_full", -1.0),
                    ("absmax_ratio_reduction_full", "absmax_ratio_full", "absmax_ratio_full", -1.0),
                    ("nonfinite_count_reduction_full", "nonfinite_count_full", "nonfinite_count_full", -1.0),
                    ("huge_count_reduction_full", "huge_count_full", "huge_count_full", -1.0),
                ]:
                    pre_x = _finite_float(ev.get(pre_key if pre_key.startswith("post_") else pre_key))
                    # Event keys use no pre_ prefix for full-check statistics.
                    pre_x = _finite_float(ev.get(pre_key))
                    post_x = _finite_float(ev.get("post_repair_" + post_key))
                    if math.isfinite(pre_x) and math.isfinite(post_x):
                        delta = post_x - pre_x if sign > 0 else pre_x - post_x
                        st["improve"][key].append(delta)

    rows: List[Dict[str, Any]] = []
    for key_row in sample_keys[["sample_i", "sample_id"]].to_dict(orient="records"):
        sample_i = int(key_row["sample_i"])
        sample_id = str(key_row["sample_id"])
        st = states.get(sample_i, _new_state())
        events = sorted(st["events"], key=lambda e: _nonneg_int(e.get("trace_order"), 0))
        ctx = context.get(sample_i, {})
        final_trace = _nonneg_int(ctx.get("final_trace_order"), 0)
        budget_values = [_nonneg_int(e.get("max_repair_count_per_sample"), 0) for e in events]
        budget_values = [x for x in budget_values if x > 0]
        budget_max = budget_values[0] if budget_values else _nonneg_int(ctx.get("max_repair_count_per_sample"), default_budget_max)
        if budget_max <= 0:
            budget_max = int(default_budget_max)

        n = len(events)
        triggered = sum(_int01(e.get("repair_triggered")) for e in events)
        modified = sum(_int01(e.get("repair_modified")) for e in events)
        nonfinite_n = sum(_nonneg_int(e.get("nonfinite_count_sample"), 0) > 0 for e in events)
        huge_n = sum(_nonneg_int(e.get("huge_count_sample"), 0) > 0 for e in events)
        no_modify = sum(_int01(e.get("repair_modified")) == 0 for e in events)
        trigger_no_modify = sum(
            _int01(e.get("repair_triggered")) == 1 and _int01(e.get("repair_modified")) == 0
            for e in events
        )
        first = events[0] if events else {}
        last = events[-1] if events else {}
        first_trace = _nonneg_int(first.get("trace_order"), 0) if events else 0
        last_trace = _nonneg_int(last.get("trace_order"), 0) if events else 0
        first_modified_pos: Optional[int] = None
        for pos, e in enumerate(events):
            if _int01(e.get("repair_modified")):
                first_modified_pos = pos
                break
        after_events = 0 if first_modified_pos is None else max(0, n - first_modified_pos - 1)
        after_modified = 0 if first_modified_pos is None else sum(
            _int01(e.get("repair_modified")) for e in events[first_modified_pos + 1 :]
        )
        post_stats_n = max((len(v) for v in st["post"].values()), default=0)
        post_nf = st["post"].get("nonfinite_count_full", [])
        post_huge = st["post"].get("huge_count_full", [])
        pre_nf = st["pre"].get("nonfinite_count_full", [])
        eliminated = 0
        for e in events:
            a = _finite_float(e.get("nonfinite_count_full"))
            b = _finite_float(e.get("post_repair_nonfinite_count_full"))
            if math.isfinite(a) and math.isfinite(b) and a > 0 and b <= 0:
                eliminated += 1

        vals: Dict[str, float] = {name: 0.0 for name in REPAIR_FEATURES}
        def put(short: str, value: Any) -> None:
            vals[PREFIX + short] = float(value)

        put("event_count", n)
        put("triggered_count", triggered)
        put("modified_count", modified)
        put("nonfinite_event_count", nonfinite_n)
        put("huge_event_count", huge_n)
        put("any_event", int(n > 0))
        put("any_triggered", int(triggered > 0))
        put("any_modified", int(modified > 0))
        put("no_modify_event_count", no_modify)
        put("trigger_no_modify_count", trigger_no_modify)
        put("unique_module_count", len(st["modules"]))
        put("unique_class_count", len(st["classes"]))
        put("unique_group_count", len([g for g, c in st["groups"].items() if c > 0]))
        put("unique_role_count", len([r for r, c in st["roles"].items() if c > 0]))
        put("first_trace_order", first_trace)
        put("last_trace_order", last_trace)
        put("trace_span", max(0, last_trace - first_trace))
        put("first_trace_fraction", _safe_div(first_trace, final_trace))
        put("last_trace_fraction", _safe_div(last_trace, final_trace))
        put("trace_span_fraction", _safe_div(max(0, last_trace - first_trace), final_trace))
        put("budget_max", budget_max)
        put("budget_used_ratio", _safe_div(modified, budget_max))
        put("budget_saturated", int(modified >= budget_max and budget_max > 0))
        put("budget_exhausted_event_count", st["actions"].get("budget_exhausted", 0))
        put("events_after_first_modified_count", after_events)
        put("modified_after_first_modified_count", after_modified)
        put("post_stats_event_count", post_stats_n)
        put("post_nonfinite_remaining_event_count", sum(float(x) > 0 for x in post_nf))
        put("nonfinite_eliminated_event_count", eliminated)
        put("post_huge_remaining_event_count", sum(float(x) > 0 for x in post_huge))
        put("trigger_per_event", _safe_div(triggered, n))
        put("modified_per_event", _safe_div(modified, n))
        put("modified_per_trigger", _safe_div(modified, triggered))
        put("nonfinite_per_event", _safe_div(nonfinite_n, n))
        put("huge_per_event", _safe_div(huge_n, n))
        put("no_modify_per_event", _safe_div(no_modify, n))
        put("replace_output_failed_count", sum(_int01(e.get("replace_output_ok")) == 0 for e in events))
        put("high_contam_event_count", sum(_int01(e.get("high_contam_policy_flag")) for e in events))

        for short, key, mode in [
            ("pre_finite_ratio_sample_min", "finite_ratio_sample", "min"),
            ("pre_finite_ratio_sample_mean", "finite_ratio_sample", "mean"),
            ("pre_abnormal_ratio_sample_max", "abnormal_ratio_sample", "max"),
            ("pre_abnormal_ratio_sample_mean", "abnormal_ratio_sample", "mean"),
            ("pre_nonfinite_ratio_sample_max", "nonfinite_ratio_sample", "max"),
            ("pre_nonfinite_ratio_sample_mean", "nonfinite_ratio_sample", "mean"),
            ("pre_huge_ratio_sample_max", "huge_ratio_sample", "max"),
            ("pre_huge_ratio_sample_mean", "huge_ratio_sample", "mean"),
            ("pre_absmax_ratio_sample_max", "absmax_ratio_sample", "max"),
            ("pre_absmax_ratio_sample_mean", "absmax_ratio_sample", "mean"),
            ("pre_finite_ratio_full_min", "finite_ratio_full", "min"),
            ("pre_finite_ratio_full_mean", "finite_ratio_full", "mean"),
            ("pre_abnormal_ratio_full_max", "abnormal_ratio_full", "max"),
            ("pre_abnormal_ratio_full_mean", "abnormal_ratio_full", "mean"),
            ("pre_nonfinite_ratio_full_max", "nonfinite_ratio_full", "max"),
            ("pre_nonfinite_ratio_full_mean", "nonfinite_ratio_full", "mean"),
            ("pre_huge_ratio_full_max", "huge_ratio_full", "max"),
            ("pre_huge_ratio_full_mean", "huge_ratio_full", "mean"),
            ("pre_absmax_ratio_full_max", "absmax_ratio_full", "max"),
            ("pre_absmax_ratio_full_mean", "absmax_ratio_full", "mean"),
        ]:
            put(short, _list_stat(st["pre"].get(key, []), mode, 0.0))

        for short, key, mode in [
            ("post_finite_ratio_full_min", "finite_ratio_full", "min"),
            ("post_finite_ratio_full_mean", "finite_ratio_full", "mean"),
            ("post_abnormal_ratio_full_max", "abnormal_ratio_full", "max"),
            ("post_abnormal_ratio_full_mean", "abnormal_ratio_full", "mean"),
            ("post_absmax_ratio_full_max", "absmax_ratio_full", "max"),
            ("post_absmax_ratio_full_mean", "absmax_ratio_full", "mean"),
            ("post_nonfinite_count_full_sum", "nonfinite_count_full", "sum"),
            ("post_nonfinite_count_full_max", "nonfinite_count_full", "max"),
            ("post_huge_count_full_sum", "huge_count_full", "sum"),
            ("post_huge_count_full_max", "huge_count_full", "max"),
        ]:
            put(short, _list_stat(st["post"].get(key, []), mode, 0.0))

        for short, key, mode in [
            ("finite_ratio_gain_full_mean", "finite_ratio_gain_full", "mean"),
            ("finite_ratio_gain_full_max", "finite_ratio_gain_full", "max"),
            ("abnormal_ratio_reduction_full_mean", "abnormal_ratio_reduction_full", "mean"),
            ("abnormal_ratio_reduction_full_max", "abnormal_ratio_reduction_full", "max"),
            ("absmax_ratio_reduction_full_mean", "absmax_ratio_reduction_full", "mean"),
            ("absmax_ratio_reduction_full_max", "absmax_ratio_reduction_full", "max"),
            ("nonfinite_count_reduction_full_sum", "nonfinite_count_reduction_full", "sum"),
            ("nonfinite_count_reduction_full_max", "nonfinite_count_reduction_full", "max"),
            ("huge_count_reduction_full_sum", "huge_count_reduction_full", "sum"),
            ("huge_count_reduction_full_max", "huge_count_reduction_full", "max"),
        ]:
            put(short, _list_stat(st["improve"].get(key, []), mode, 0.0))

        first_group = canonical_group(first.get("uniad_group")) if events else "unknown"
        last_group = canonical_group(last.get("uniad_group")) if events else "unknown"
        first_role = repair_role(first_group, first.get("module_class"), first.get("module_name"), first.get("module_aware_role")) if events else "generic"
        last_role = repair_role(last_group, last.get("module_class"), last.get("module_name"), last.get("module_aware_role")) if events else "generic"
        first_action = action_category(first.get("repair_reason"), first.get("high_contam_action")) if events else "other"

        for g in GROUPS:
            s = _slug(g)
            gc = int(st["groups"].get(g, 0))
            gm = int(st["groups_modified"].get(g, 0))
            put("group_%s_event_count" % s, gc)
            put("group_%s_modified_count" % s, gm)
            put("group_%s_event_ratio" % s, _safe_div(gc, n))
            put("group_%s_modified_ratio" % s, _safe_div(gm, modified))
            put("first_group_%s" % s, int(n > 0 and first_group == g))
            put("last_group_%s" % s, int(n > 0 and last_group == g))

        for r in ROLES:
            s = _slug(r)
            rc = int(st["roles"].get(r, 0))
            rm = int(st["roles_modified"].get(r, 0))
            put("role_%s_event_count" % s, rc)
            put("role_%s_modified_count" % s, rm)
            put("role_%s_event_ratio" % s, _safe_div(rc, n))
            put("role_%s_modified_ratio" % s, _safe_div(rm, modified))
            put("first_role_%s" % s, int(n > 0 and first_role == r))
            put("last_role_%s" % s, int(n > 0 and last_role == r))

        for a in ACTIONS:
            s = _slug(a)
            ac = int(st["actions"].get(a, 0))
            put("action_%s_count" % s, ac)
            put("action_%s_ratio" % s, _safe_div(ac, n))
            put("first_action_%s" % s, int(n > 0 and first_action == a))

        first_band_sample = canonical_band(first.get("contamination_band_sample")) if events else "unknown"
        first_band_full = canonical_band(first.get("contamination_band_full")) if events else "unknown"
        for b in BANDS:
            put("first_contamination_sample_%s" % _slug(b), int(n > 0 and first_band_sample == b))
            put("first_contamination_full_%s" % _slug(b), int(n > 0 and first_band_full == b))

        row = {"sample_i": sample_i, "sample_id": sample_id}
        row.update(vals)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out["sample_i"].duplicated().any():
        raise RuntimeError("duplicate sample_i after repair telemetry aggregation")
    missing_features = [c for c in REPAIR_FEATURES if c not in out.columns]
    if missing_features:
        raise RuntimeError("missing generated repair features: %s" % missing_features[:10])
    if out[REPAIR_FEATURES].isna().any().any():
        bad = out[REPAIR_FEATURES].columns[out[REPAIR_FEATURES].isna().any()].tolist()
        raise RuntimeError("NaN in generated repair features: %s" % bad[:20])

    audit = {
        "event_file": str(events_path),
        "debug_summary_file": str(debug_summary_path),
        "sample_rows": int(len(out)),
        "event_rows_consumed": int(event_rows),
        "samples_with_event": int((out[PREFIX + "any_event"] > 0).sum()),
        "samples_with_modified": int((out[PREFIX + "any_modified"] > 0).sum()),
        "unknown_sample_events_ignored": int(unknown_sample_events),
        "feature_count": int(len(REPAIR_FEATURES)),
        "event_field_allowlist": sorted(EVENT_FIELD_ALLOWLIST),
        "debug_summary_field_allowlist": sorted(DEBUG_SUMMARY_FIELD_ALLOWLIST),
        "forbidden_source_tokens": list(FORBIDDEN_SOURCE_TOKENS),
        "uses_clean_reference": 0,
        "uses_gt_or_delta": 0,
        "uses_gate_decision": 0,
        "uses_fault_or_bit_metadata": 0,
        "uses_posthoc_safety_label": 0,
    }
    return out, audit


def validate_source_allowlists() -> None:
    all_fields = sorted(EVENT_FIELD_ALLOWLIST | DEBUG_SUMMARY_FIELD_ALLOWLIST)
    bad = [f for f in all_fields if any(tok in f.lower() for tok in FORBIDDEN_SOURCE_TOKENS)]
    if bad:
        raise RuntimeError("forbidden field in repair telemetry allowlist: %s" % bad)
    if len(REPAIR_FEATURES) != len(set(REPAIR_FEATURES)):
        dup = sorted({x for x in REPAIR_FEATURES if REPAIR_FEATURES.count(x) > 1})
        raise RuntimeError("duplicate repair telemetry feature names: %s" % dup)
