#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v73.8E no-detector multi-stage retrain/eval.

Purpose
-------
Train and evaluate an independent online-available DriveVLA detector pipeline:
  C1: the only main classifier; it uses online raw/derived/consistency features only.
  C2: a second-stage veto/correction model; it may use this round's C1 OOF/current outputs
      plus the same allowed online features.

Forbidden as features
---------------------
- Any previous detector/classifier output: score__, alarm__, threshold__, hgbmax, lineA*.
- Any posthoc/evaluation/leakage columns: harmful labels, delta L2/collision, GT/clean/fault-vs-clean.
- Any fault metadata or sample identity: seed, sample_id, exact_layer, target_group, fault_scope, etc.

Confirmed defaults
------------------
Feature matrix:
/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260520_v63_decode_token_feature_formal8_956302_956309_analysis/v59_targeted_feature_audit/v59_targeted_feature_matrix_sanitized_full_v63_clean.csv

Label table:
/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/20260520_v63_decode_token_feature_formal8_956302_956309_analysis/v54_online_exports/v54_2_posthoc_label_table_multi_policy.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_MATRIX = Path(
    "/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/"
    "20260520_v63_decode_token_feature_formal8_956302_956309_analysis/"
    "v59_targeted_feature_audit/v59_targeted_feature_matrix_sanitized_full_v63_clean.csv"
)
DEFAULT_LABELS = Path(
    "/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/"
    "20260520_v63_decode_token_feature_formal8_956302_956309_analysis/"
    "v54_online_exports/v54_2_posthoc_label_table_multi_policy.csv"
)
DEFAULT_OLD_EXACT = Path(
    "/root/autodl-tmp/DriveVLA/output/OpenDriveVLA-0.5B/"
    "20260520_v63_decode_token_feature_formal8_956302_956309_analysis/"
    "v54_online_exports/v54_online_deployable_feature_matrix.csv"
)
DEFAULT_OUT = Path("/root/autodl-tmp/DriveVLA/pack_output_analysis/v73_8e_fullfeature_splitused")
DEFAULT_EXACT_FEATURE_POLICY = None  # supplied by run script from package resources

LABELS_DEFAULT = ["harmful", "severe_harmful", "v31_speed_alter_harmful_compatible"]
KEY_COLS = ["seed", "sample_id"]

# v73.8E feature policy is intentionally exact-list / exact-namespace based.
# It does NOT use semantic fuzzy matching, feature importance, SHAP, correlation,
# or substring-based model-output inference to decide training columns.
#
# Source policy:
#   v63_full: this exact matrix was pre-checked; detector-output-like columns = 0,
#             label/leakage-like column = exact_layer only. Therefore allowed columns
#             are exactly all header columns except exact identity/audit columns below.
#   old_exact: only the exact old deployable raw feature namespace `feat__*` is eligible.
#              Even inside `feat__*`, exact report/alarm/score components are excluded.
#   C2 internal: only the exact names in C2_INTERNAL_FEATURES are allowed.
#
# If a future column falls outside these exact namespaces, it is excluded and written
# to unrecognized review files; it is never silently used.

V63_FULL_EXACT_DROP_COLUMNS = {
    "seed",
    "sample_id",
    "exact_layer",
}

OLD_EXACT_EXACT_DROP_COLUMNS = {
    "seed",
    "sample_id",
    "raw_sample_id",
    "row_source",
    "sample_token",
    "row_id",
    "sample_index",
    "exact_layer",
    "target_group",
    "fault_scope",
    "scope_group",
    "fault_type",
    "bit_idx",
    "bit_index",
    "target_name",
    "target_module",
    "applied",
    "effective",
    # old exact legacy report/rule outputs observed in v73.4/v73.5 logs
    "e_output_sanity_alarm",
    "raw_input_alarm",
    "v19_output_first_harm_alarm",
}

# Exact feature namespace roots, not fuzzy matching.
V63_FULL_ALLOWED_PREFIXES = (
    "clip__feat__",
    "slog1p__feat__",
    "miss__feat__",
    "extreme__feat__",
    "safe_raw__feat__",
    "prop__",
    "v73gen__",
)

OLD_EXACT_ALLOWED_PREFIXES = (
    "feat__",
)

# Exact detector/model-output or legacy report components that must not enter C1/C2.
# These are checked as exact `__` components or exact prefixes/suffixes, not fuzzy tokens.
FORBIDDEN_EXACT_PREFIXES = (
    "score__",
    "alarm__",
    "threshold__",
)
FORBIDDEN_EXACT_NAMES = {
    "hgbmax",
}
FORBIDDEN_EXACT_COMPONENTS = {
    "alarm",
    "strong_alarm",
    "soft_score",
    "detector_score",
    "detector_alarm",
    "detector_threshold",
}
FORBIDDEN_EXACT_SUFFIXES = (
    "__alarm",
    "__strong_alarm",
    "__soft_score",
    "__margin",
    "_alarm",
    "_strong_alarm",
    "_soft_score",
)

# Exact leakage/evaluation columns. Labels are joined later as y only and never included
# in feature_cols. This set is deliberately exact for the current confirmed label table.
LABEL_OR_EVAL_EXACT_COLUMNS = {
    "harmful",
    "severe_harmful",
    "v31_old_harmful_compatible",
    "v31_heading_harmful_compatible",
    "v31_heading_expanded_harmful_compatible",
    "v31_speed_harmful_compatible",
    "v31_speed_alter_harmful_compatible",
    "v31_heading_speed_alter_harmful_compatible",
    "delta_l2_avg",
    "delta_collision_avg",
}

C2_INTERNAL_FEATURES = [
    "v73_c1_oof_score",
    "v73_c1_oof_logit",
    "v73_c1_oof_rank",
    "v73_c1_oof_score_band",
    "v73_c1_oof_borderline_flag",
]


class SafeFiniteClipper(BaseEstimator, TransformerMixin):
    """Make numeric inputs finite and bounded while preserving column order.

    ExtraTrees internally casts input arrays to float32. Some legacy old-exact
    raw features can contain very large finite values; those values trigger
    overflow during the float32 cast even after NaN imputation. This transformer
    performs a deterministic deployment-safe numeric clamp before imputation and
    model fitting. The cap is deliberately the same order as the v59 sanitized
    matrix cap used in earlier probes, so it is not a learned feature filter.
    """
    def __init__(self, abs_cap: float = 1_000_000.0):
        self.abs_cap = float(abs_cap)
        self.n_features_in_ = None

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise RuntimeError(f"SafeFiniteClipper expects 2D input, got shape={arr.shape}")
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        if self.n_features_in_ is None:
            raise RuntimeError("SafeFiniteClipper is not fitted")
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.n_features_in_:
            raise RuntimeError(
                f"SafeFiniteClipper feature width mismatch: expected {self.n_features_in_}, got {arr.shape}"
            )
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return np.clip(arr, -self.abs_cap, self.abs_cap)


class SafeMedianImputer(BaseEstimator, TransformerMixin):
    """Median imputer that keeps every column and maps all-NaN columns to 0.

    sklearn SimpleImputer(strategy='median') can drop all-empty columns in some
    versions. That changes feature-index semantics and can break exact split-used
    audits. This transformer preserves the exact feature order/width.
    """
    def __init__(self, fill_value: float = 0.0, abs_cap: float = 1_000_000.0):
        self.fill_value = float(fill_value)
        self.abs_cap = float(abs_cap)
        self.statistics_ = None
        self.all_nan_mask_ = None

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        arr = np.clip(arr, -self.abs_cap, self.abs_cap)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(arr, axis=0)
        all_nan = ~np.isfinite(med)
        med[all_nan] = self.fill_value
        med = np.clip(med, -self.abs_cap, self.abs_cap)
        self.statistics_ = med.astype(np.float64)
        self.all_nan_mask_ = all_nan.astype(bool)
        return self

    def transform(self, X):
        if self.statistics_ is None:
            raise RuntimeError("SafeMedianImputer is not fitted")
        arr = np.asarray(X, dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        arr = np.clip(arr, -self.abs_cap, self.abs_cap)
        out = np.where(np.isfinite(arr), arr, self.statistics_)
        return np.clip(out, -self.abs_cap, self.abs_cap)

@dataclass
class FeatureDecision:
    column: str
    source: str
    role: str
    allowed_for_c1: int
    allowed_for_c2_raw: int
    drop_reason: str
    family: str


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def lower(s: str) -> str:
    return str(s).lower()


def exact_components(col: str) -> List[str]:
    """Return exact name components for policy checks.

    This is used only for exact component membership such as `alarm` or
    `soft_score`; it is not semantic/fuzzy matching.
    """
    c = str(col)
    parts: List[str] = []
    parts.extend([x for x in c.split("__") if x])
    parts.extend([x for x in c.split("_") if x])
    return parts


def has_exact_prefix(col: str, prefixes: Sequence[str]) -> bool:
    return any(str(col).startswith(p) for p in prefixes)


def has_exact_suffix(col: str, suffixes: Sequence[str]) -> bool:
    return any(str(col).endswith(s) for s in suffixes)


def is_forbidden_exact_detector_output(col: str) -> bool:
    c = str(col)
    if c in FORBIDDEN_EXACT_NAMES:
        return True
    if has_exact_prefix(c, FORBIDDEN_EXACT_PREFIXES):
        return True
    if has_exact_suffix(c, FORBIDDEN_EXACT_SUFFIXES):
        return True
    comps = set(exact_components(c))
    if comps & FORBIDDEN_EXACT_COMPONENTS:
        return True
    # Exact historical line names are previous detector identities.
    if c.startswith("lineA_old") or c.startswith("lineA_speed") or c.startswith("linea_old") or c.startswith("linea_speed"):
        return True
    return False


def is_v63_allowed_exact(col: str) -> Tuple[bool, str]:
    c = str(col)
    if c in V63_FULL_EXACT_DROP_COLUMNS:
        return False, "exact_v63_identity_or_audit_column"
    if c in LABEL_OR_EVAL_EXACT_COLUMNS:
        return False, "exact_label_or_eval_column"
    if is_forbidden_exact_detector_output(c):
        return False, "exact_detector_or_classifier_output"
    if has_exact_prefix(c, V63_FULL_ALLOWED_PREFIXES):
        return True, ""
    return False, "v63_not_in_exact_online_namespace"


def is_old_exact_allowed_exact(col: str) -> Tuple[bool, str]:
    c = str(col)
    if c in OLD_EXACT_EXACT_DROP_COLUMNS:
        return False, "exact_old_metadata_or_known_report_column"
    if c in LABEL_OR_EVAL_EXACT_COLUMNS:
        return False, "exact_label_or_eval_column"
    if is_forbidden_exact_detector_output(c):
        return False, "exact_detector_or_classifier_output"
    if has_exact_prefix(c, OLD_EXACT_ALLOWED_PREFIXES):
        return True, ""
    return False, "old_exact_not_in_exact_feat_namespace"


def infer_family(col: str) -> str:
    c = lower(col)
    if "decode_token" in c or "output_last" in c or "top1_prob" in c or "top5" in c or "entropy" in c:
        return "decode_token"
    if "d_llm" in c:
        return "D_LLM"
    if "dout" in c or "d_out" in c or "lm_head" in c:
        return "D_out"
    if "s4" in c:
        return "S4"
    if "s3" in c:
        return "S3"
    if "s2" in c:
        return "S2"
    if "s1" in c:
        return "S1"
    if "s0" in c or "can_bus" in c or "canbus" in c:
        return "S0_canbus"
    if "mm_projector" in c or "projector" in c or "vision" in c:
        return "C"
    if c.startswith("prop__"):
        return "propagation_existing"
    if c.startswith("v73gen__"):
        return "v73_generated_consistency"
    return "other_online"


def classify_column(col: str, source: str = "v63_full") -> FeatureDecision:
    """Classify one column with exact source-specific policy.

    No fuzzy matching is used for inclusion. v63_full columns are allowed only
    by exact namespace membership after exact drops; old_exact columns are
    allowed only by the exact `feat__` raw-feature namespace after exact
    detector/report/audit exclusions. Every column is still written to the
    manifest so the decision can be audited line by line.
    """
    c = str(col)
    if source == "old_exact":
        ok, reason = is_old_exact_allowed_exact(c)
        if ok:
            return FeatureDecision(c, source, "allowed_old_exact_online_feature", 1, 1, "", infer_family(c))
        role = "forbidden_detector_or_classifier_output" if reason == "exact_detector_or_classifier_output" else (
            "forbidden_label_or_evaluation_leakage" if reason == "exact_label_or_eval_column" else (
                "audit_or_fault_metadata" if reason == "exact_old_metadata_or_known_report_column" else "unrecognized_excluded_for_review"
            )
        )
        return FeatureDecision(c, source, role, 0, 0, reason, infer_family(c))

    if source == "v63_full":
        ok, reason = is_v63_allowed_exact(c)
        if ok:
            return FeatureDecision(c, source, "allowed_online_feature", 1, 1, "", infer_family(c))
        role = "forbidden_detector_or_classifier_output" if reason == "exact_detector_or_classifier_output" else (
            "forbidden_label_or_evaluation_leakage" if reason == "exact_label_or_eval_column" else (
                "audit_or_fault_metadata" if reason == "exact_v63_identity_or_audit_column" else "unrecognized_excluded_for_review"
            )
        )
        return FeatureDecision(c, source, role, 0, 0, reason, infer_family(c))

    if source == "generated_consistency":
        if c.startswith("v73gen__"):
            return FeatureDecision(c, source, "allowed_generated_online_consistency_feature", 1, 1, "", infer_family(c))
        return FeatureDecision(c, source, "unrecognized_excluded_for_review", 0, 0, "generated_feature_without_v73gen_prefix", infer_family(c))

    if c in C2_INTERNAL_FEATURES:
        return FeatureDecision(c, source, "allowed_this_round_c1_output_for_c2_only", 0, 1, "", infer_family(c))

    return FeatureDecision(c, source, "unrecognized_excluded_for_review", 0, 0, "unknown_source", infer_family(c))

def read_header(path: Path) -> List[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def build_feature_manifest(source_to_cols: Dict[str, Sequence[str]], out_dir: Path) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    rows = []
    for source, cols in source_to_cols.items():
        rows.extend([asdict(classify_column(c, source=source)) for c in cols])
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "v73_8e_feature_manifest_all_columns.csv", index=False)
    for (source, role), sub in manifest.groupby(["source", "role"]):
        safe_source = re.sub(r"[^A-Za-z0-9_]+", "_", str(source))
        safe_role = re.sub(r"[^A-Za-z0-9_]+", "_", str(role))
        sub.to_csv(out_dir / f"v73_8e_feature_manifest__{safe_source}__role__{safe_role}.csv", index=False)

    allowed_by_source: Dict[str, List[str]] = {}
    for source, sub in manifest.groupby("source"):
        allowed_by_source[str(source)] = sub.loc[sub["allowed_for_c1"] == 1, "column"].tolist()

    allowed_all = []
    for source, cols in allowed_by_source.items():
        for c in cols:
            allowed_all.append({"source": source, "feature": c})
    pd.DataFrame(allowed_all).to_csv(out_dir / "v73_8e_allowed_online_features.csv", index=False)
    manifest.loc[manifest["allowed_for_c1"] == 0].to_csv(out_dir / "v73_8e_dropped_features_with_reasons.csv", index=False)
    manifest.loc[manifest["role"].str.contains("forbidden", na=False)].to_csv(out_dir / "v73_8e_forbidden_detector_or_leakage_columns.csv", index=False)
    manifest.loc[manifest["role"] == "unrecognized_excluded_for_review"].to_csv(out_dir / "v73_8e_unrecognized_excluded_columns_for_review.csv", index=False)
    manifest.loc[manifest["family"].isin(["C", "D_LLM", "D_out", "decode_token", "S4", "v73_generated_consistency"])].to_csv(
        out_dir / "v73_8e_c_d_llm_dout_decode_s4_feature_candidates.csv", index=False
    )
    return manifest, allowed_by_source


def load_exact_allowed_feature_policy(policy_csv: Path, source_to_cols: Dict[str, Sequence[str]], manifest: pd.DataFrame, out_dir: Path) -> Dict[str, List[str]]:
    """Load a frozen exact feature allowlist.

    This is the v73.8E strict path: training columns are not selected by runtime
    keyword/fuzzy logic. They must appear exactly in the package-owned policy CSV
    and exactly in the source matrix header.
    """
    if policy_csv is None or not Path(policy_csv).exists():
        raise RuntimeError(f"Exact feature policy CSV not found: {policy_csv}")
    pol = pd.read_csv(policy_csv)
    required = {"source", "feature"}
    if not required.issubset(pol.columns):
        raise RuntimeError(f"Exact feature policy must contain columns {sorted(required)}; got {list(pol.columns)}")
    pol["source"] = pol["source"].astype(str)
    pol["feature"] = pol["feature"].astype(str)
    if pol.duplicated(["source", "feature"]).any():
        dup = pol.loc[pol.duplicated(["source", "feature"], keep=False)].head(100)
        dup.to_csv(out_dir / "v73_8e_exact_policy_duplicate_entries_sample.csv", index=False)
        raise RuntimeError("Exact feature policy has duplicate source+feature entries")

    allowed: Dict[str, List[str]] = {}
    missing_rows = []
    disallowed_rows = []
    header_sets = {src: set(map(str, cols)) for src, cols in source_to_cols.items()}
    manifest_key = {(str(r.source), str(r.column)): r for r in manifest.itertuples(index=False)}
    for src, sub in pol.groupby("source", sort=True):
        if src not in header_sets:
            raise RuntimeError(f"Exact feature policy source not available in current run: {src}")
        feats = sub["feature"].tolist()
        for f in feats:
            if f not in header_sets[src]:
                missing_rows.append({"source": src, "feature": f, "reason": "not_in_current_header"})
                continue
            dec = manifest_key.get((src, f))
            if dec is None or int(getattr(dec, "allowed_for_c1")) != 1:
                disallowed_rows.append({"source": src, "feature": f, "reason": "not_allowed_by_current_exact_manifest"})
        allowed[src] = feats

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(out_dir / "v73_8e_exact_policy_missing_features.csv", index=False)
        raise RuntimeError(f"Exact feature policy has missing source columns: {len(missing_rows)}")
    if disallowed_rows:
        pd.DataFrame(disallowed_rows).to_csv(out_dir / "v73_8e_exact_policy_disallowed_features.csv", index=False)
        raise RuntimeError(f"Exact feature policy contains features disallowed by current manifest: {len(disallowed_rows)}")

    pd.DataFrame([{"source": src, "feature": f} for src, feats in allowed.items() for f in feats]).to_csv(
        out_dir / "v73_8e_exact_allowed_features_used_for_training.csv", index=False
    )
    pd.DataFrame([{
        "source": src,
        "feature_count": len(feats),
        "policy_csv": str(policy_csv),
    } for src, feats in allowed.items()]).to_csv(out_dir / "v73_8e_exact_policy_summary.csv", index=False)
    return allowed


def assert_no_forbidden_features(features: Sequence[str], context: str, allow_c2_internal: bool = True) -> None:
    bad = []
    for f in features:
        if allow_c2_internal and f in C2_INTERNAL_FEATURES:
            continue
        # At this stage features have already been selected from the exact allowed
        # manifest. This final guard checks only exact detector/model-output,
        # exact labels/eval columns, and exact metadata names. It deliberately
        # does not re-run a source-specific namespace classifier, because the
        # same feature list may contain both v63_full and old_exact `feat__*` names.
        if is_forbidden_exact_detector_output(f):
            bad.append((f, "exact_detector_or_classifier_output"))
        elif f in LABEL_OR_EVAL_EXACT_COLUMNS:
            bad.append((f, "exact_label_or_eval_column"))
        elif f in V63_FULL_EXACT_DROP_COLUMNS or f in OLD_EXACT_EXACT_DROP_COLUMNS:
            bad.append((f, "exact_metadata_or_identity_column"))
    if bad:
        sample = bad[:50]
        msg = json.dumps(sample, ensure_ascii=False, indent=2)
        raise RuntimeError(f"Forbidden/non-online features found in {context}: first bad entries:\n{msg}")

def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    denom = max(1, len(x) - 1)
    return order / denom


def score_band(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # Fixed online-deployable banding over score in [0, 1].
    return np.digitize(x, bins=[0.1, 0.25, 0.5, 0.75, 0.9]).astype(float)


def borderline_flag(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return ((x >= 0.25) & (x <= 0.9)).astype(float)


def load_matrices_and_labels(
    matrix_csv: Path,
    old_exact_csv: Optional[Path],
    label_csv: Path,
    allowed_by_source: Dict[str, List[str]],
    labels: List[str],
    quick_rows: int,
    out_dir: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    main_allowed = list(allowed_by_source.get("v63_full", []))
    old_allowed_raw = list(allowed_by_source.get("old_exact", []))
    # Avoid duplicate feature names; keep v63/full feature when same name exists.
    old_allowed = [c for c in old_allowed_raw if c not in set(KEY_COLS + main_allowed)]
    skipped_old_duplicates = sorted(set(old_allowed_raw) - set(old_allowed))
    pd.DataFrame({"old_exact_duplicate_skipped": skipped_old_duplicates}).to_csv(
        out_dir / "v73_8e_old_exact_duplicate_columns_skipped.csv", index=False
    )

    usecols = KEY_COLS + main_allowed
    log(f"Loading v63/full matrix usecols: keys={KEY_COLS}, allowed_features={len(main_allowed)}")
    if quick_rows and quick_rows > 0:
        X = pd.read_csv(matrix_csv, usecols=usecols, nrows=quick_rows)
    else:
        X = pd.read_csv(matrix_csv, usecols=usecols)
    for c in KEY_COLS:
        X[c] = X[c].astype(str)
    log(f"Loaded v63/full matrix subset: shape={X.shape}")

    if old_exact_csv is not None and str(old_exact_csv) and old_exact_csv.exists() and old_allowed:
        old_usecols = KEY_COLS + old_allowed
        log(f"Loading old-exact deployable raw matrix usecols: old_allowed_features={len(old_allowed)}")
        X_old = pd.read_csv(old_exact_csv, usecols=old_usecols)
        for c in KEY_COLS:
            X_old[c] = X_old[c].astype(str)
        if X_old.duplicated(KEY_COLS).any():
            X_old.loc[X_old.duplicated(KEY_COLS, keep=False), KEY_COLS].head(50).to_csv(
                out_dir / "v73_8e_old_exact_duplicate_keys_sample.csv", index=False
            )
            raise RuntimeError("old_exact matrix has duplicate seed+sample_id keys")
        before_cols = X.shape[1]
        X = X.merge(X_old, on=KEY_COLS, how="left", validate="one_to_one")
        missing_old_rows = int(X[old_allowed].isna().all(axis=1).sum()) if old_allowed else 0
        log(f"Merged old-exact features: shape={X.shape}, added={X.shape[1]-before_cols}, all-old-missing-rows={missing_old_rows}")
        pd.DataFrame({
            "old_exact_csv": [str(old_exact_csv)],
            "old_allowed_raw": [len(old_allowed_raw)],
            "old_allowed_used": [len(old_allowed)],
            "old_duplicate_skipped": [len(skipped_old_duplicates)],
            "all_old_missing_rows_after_join": [missing_old_rows],
        }).to_csv(out_dir / "v73_8e_old_exact_merge_summary.csv", index=False)
    else:
        log("No old-exact matrix loaded; continuing with v63/full only")

    label_usecols = KEY_COLS + labels
    label_header = read_header(label_csv)
    missing_y = [c for c in label_usecols if c not in label_header]
    if missing_y:
        raise RuntimeError(f"Label CSV missing required columns: {missing_y}")
    Y = pd.read_csv(label_csv, usecols=label_usecols)
    log(f"Loaded labels: shape={Y.shape}")

    for c in KEY_COLS:
        Y[c] = Y[c].astype(str)
    if Y.duplicated(KEY_COLS).any():
        dup = Y.loc[Y.duplicated(KEY_COLS, keep=False), KEY_COLS].head(20)
        dup.to_csv(out_dir / "v73_8e_duplicate_label_keys_sample.csv", index=False)
        raise RuntimeError("Label table has duplicate seed+sample_id keys; wrote sample CSV.")
    M = X.merge(Y, on=KEY_COLS, how="left", validate="one_to_one")
    miss = {lab: int(M[lab].isna().sum()) for lab in labels}
    if any(v > 0 for v in miss.values()):
        raise RuntimeError(f"Missing labels after seed+sample_id join: {miss}")
    for lab in labels:
        M[lab] = M[lab].astype(int)
    feature_cols = [c for c in M.columns if c not in set(KEY_COLS + labels)]
    log(f"Merged features+labels: shape={M.shape}, feature_cols={len(feature_cols)}")
    return M, feature_cols


def numeric_clean_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in feature_cols:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def make_generated_consistency_features(X: pd.DataFrame, out_dir: Path, max_cols_per_family: int = 250) -> pd.DataFrame:
    """Generate online-computable row-level consistency features.

    These do NOT use previous detector/classifier outputs, labels, GT, clean reference, or fault metadata.
    They are deterministic functions of allowed online feature columns.
    """
    log("Generating v73 online consistency features from allowed raw/sanitized columns")
    families: Dict[str, List[str]] = {fam: [] for fam in ["D_LLM", "D_out", "decode_token", "S4", "S3", "C"]}
    for c in X.columns:
        fam = infer_family(c)
        if fam in families:
            families[fam].append(c)
    # Limit each family deterministically to avoid large row-wise aggregation cost.
    selected: Dict[str, List[str]] = {}
    for fam, cols in families.items():
        # Prefer clipped/slog/safe raw numeric summaries; avoid miss/extreme flags dominating aggregate.
        pref = [c for c in cols if (c.startswith("clip__") or c.startswith("slog1p__") or c.startswith("safe_raw__"))]
        if not pref:
            pref = cols
        selected[fam] = pref[:max_cols_per_family]

    audit_rows = []
    gen = {}
    for fam, cols in selected.items():
        audit_rows.append({"family": fam, "candidate_cols": len(families.get(fam, [])), "used_cols": len(cols), "used_cols_preview": "|".join(cols[:20])})
        if not cols:
            continue
        arr = X[cols].to_numpy(dtype=float, copy=False)
        abs_arr = np.abs(arr)
        # Pure row-wise summaries: online-computable without calibration labels.
        gen[f"v73gen__{fam}__row_mean_abs"] = np.nanmean(abs_arr, axis=1)
        gen[f"v73gen__{fam}__row_p95_abs"] = np.nanpercentile(abs_arr, 95, axis=1)
        gen[f"v73gen__{fam}__row_std"] = np.nanstd(arr, axis=1)
        gen[f"v73gen__{fam}__row_nan_rate"] = np.mean(~np.isfinite(arr), axis=1)

    # Cross-family consistency differences/ratios based on row summaries.
    pairs = [
        ("D_LLM", "decode_token"),
        ("D_out", "decode_token"),
        ("D_LLM", "D_out"),
        ("S4", "D_LLM"),
        ("S4", "decode_token"),
        ("S3", "S4"),
        ("C", "D_LLM"),
    ]
    eps = 1e-6
    for a, b in pairs:
        ak = f"v73gen__{a}__row_mean_abs"
        bk = f"v73gen__{b}__row_mean_abs"
        if ak in gen and bk in gen:
            av = gen[ak]
            bv = gen[bk]
            gen[f"v73gen__consistency__{a}_minus_{b}__mean_abs"] = av - bv
            gen[f"v73gen__consistency__{a}_over_{b}__mean_abs_logratio"] = np.log((av + eps) / (bv + eps))
        ak = f"v73gen__{a}__row_p95_abs"
        bk = f"v73gen__{b}__row_p95_abs"
        if ak in gen and bk in gen:
            av = gen[ak]
            bv = gen[bk]
            gen[f"v73gen__consistency__{a}_minus_{b}__p95_abs"] = av - bv
            gen[f"v73gen__consistency__{a}_over_{b}__p95_abs_logratio"] = np.log((av + eps) / (bv + eps))

    pd.DataFrame(audit_rows).to_csv(out_dir / "v73_8e_generated_consistency_family_audit.csv", index=False)
    source_rows = []
    for fam, cols in selected.items():
        for c in cols:
            source_rows.append({"generated_family": fam, "source_feature": c})
    pd.DataFrame(source_rows).to_csv(out_dir / "v73_8e_generated_consistency_source_columns.csv", index=False)
    if not gen:
        return pd.DataFrame(index=X.index)
    G = pd.DataFrame(gen, index=X.index)
    G = G.replace([np.inf, -np.inf], np.nan)
    # Audit generated columns and assert they are in allowed namespace.
    for c in G.columns:
        if not c.startswith("v73gen__"):
            raise RuntimeError(f"Generated consistency feature has invalid prefix: {c}")
    pd.DataFrame({"feature": list(G.columns)}).to_csv(out_dir / "v73_8e_generated_consistency_features.csv", index=False)
    log(f"Generated consistency features: {G.shape[1]}")
    return G


def select_k_by_f_classif(X_train: pd.DataFrame, y_train: np.ndarray, k: int) -> List[str]:
    k = min(k, X_train.shape[1])
    if k <= 0:
        raise RuntimeError("No features available for selection")
    # Simple median imputation for feature scoring only.
    arr = X_train.to_numpy(dtype=float, copy=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(arr, axis=0)
    med[~np.isfinite(med)] = 0.0
    arr2 = np.where(np.isfinite(arr), arr, med)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, _ = f_classif(arr2, y_train)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    order = np.argsort(scores)[::-1][:k]
    return [X_train.columns[i] for i in order]


def make_model(model_name: str, seed: int) -> object:
    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.055,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
    if model_name == "hgb_shallow":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=260,
            max_depth=7,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def fit_predict_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, object]:
    # Use imputer for all models for consistent behavior and safe deployment plan.
    estimator = make_model(model_name, seed)
    pipe = Pipeline([
        ("clipper", SafeFiniteClipper(abs_cap=1_000_000.0)),
        ("imputer", SafeMedianImputer(fill_value=0.0, abs_cap=1_000_000.0)),
        ("model", estimator),
    ])
    if sample_weight is not None:
        pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
    else:
        pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    return proba, pipe


def metrics_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    pred = score >= threshold
    pos = y == 1
    neg = ~pos
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & neg))
    fn = int(np.sum((~pred) & pos))
    tn = int(np.sum((~pred) & neg))
    recall = tp / max(1, int(np.sum(pos)))
    far = fp / max(1, int(np.sum(neg)))
    precision = tp / max(1, tp + fp)
    return {"threshold": float(threshold), "recall": float(recall), "far": float(far), "precision": float(precision), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def threshold_grid(scores: np.ndarray, max_points: int = 1200) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.array([0.5])
    uniq = np.unique(s)
    if len(uniq) <= max_points:
        grid = uniq
    else:
        qs = np.linspace(0, 1, max_points)
        grid = np.unique(np.quantile(s, qs))
    # Add boundaries.
    grid = np.unique(np.concatenate(([0.0, 1e-12], grid, [1.0, 1.0 + 1e-12])))
    return grid


def summarize_score_operating_points(y: np.ndarray, score: np.ndarray, route: str, label: str, model_config: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    grid = threshold_grid(score)
    rows = []
    for t in grid:
        r = metrics_at_threshold(y, score, float(t))
        r.update({"route": route, "label": label, "model_config": model_config})
        rows.append(r)
    df = pd.DataFrame(rows)
    # Max recall, tie min FAR then max precision.
    max_recall_row = df.sort_values(["recall", "far", "precision"], ascending=[False, True, False]).iloc[0].to_dict()
    # Lowest FAR with at least one alarm? include all thresholds; tie max recall.
    min_far_row = df.sort_values(["far", "recall", "precision"], ascending=[True, False, False]).iloc[0].to_dict()
    ge99 = df[df["recall"] >= 0.99]
    if len(ge99):
        ge99_row = ge99.sort_values(["far", "recall", "precision"], ascending=[True, False, False]).iloc[0].to_dict()
    else:
        ge99_row = {k: np.nan for k in df.columns}
        ge99_row.update({"route": route, "label": label, "model_config": model_config, "note": "no threshold reached recall>=0.99"})
    best = {
        "label": label,
        "route": route,
        "model_config": model_config,
        "max_recall": max_recall_row,
        "min_far": min_far_row,
        "recall_ge_99_min_far": ge99_row,
    }
    return df, best


def summarize_c1_c2_policy(y: np.ndarray, c1: np.ndarray, c2: np.ndarray, route: str, label: str, model_config: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    # Policy: C1 is the only main classifier. C2 can only veto/correct within C1-routed samples.
    # Alarm iff C1 score >= t1 AND C2 keep score >= t2.
    t1_grid = threshold_grid(c1, max_points=260)
    t2_grid = threshold_grid(c2, max_points=260)
    # To keep cost bounded, use quantile-subset if grid too large.
    if len(t1_grid) > 180:
        t1_grid = np.unique(np.quantile(t1_grid, np.linspace(0, 1, 180)))
    if len(t2_grid) > 180:
        t2_grid = np.unique(np.quantile(t2_grid, np.linspace(0, 1, 180)))
    rows = []
    y = np.asarray(y).astype(int)
    pos = y == 1
    neg = ~pos
    for t1 in t1_grid:
        routed = c1 >= t1
        if not np.any(routed):
            continue
        for t2 in t2_grid:
            pred = routed & (c2 >= t2)
            tp = int(np.sum(pred & pos))
            fp = int(np.sum(pred & neg))
            fn = int(np.sum((~pred) & pos))
            tn = int(np.sum((~pred) & neg))
            recall = tp / max(1, int(np.sum(pos)))
            far = fp / max(1, int(np.sum(neg)))
            precision = tp / max(1, tp + fp)
            rows.append({
                "label": label,
                "route": route,
                "model_config": model_config,
                "c1_threshold": float(t1),
                "c2_threshold": float(t2),
                "recall": float(recall),
                "far": float(far),
                "precision": float(precision),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        empty_best = {"label": label, "route": route, "model_config": model_config, "note": "empty_policy_grid"}
        return df, {"label": label, "route": route, "model_config": model_config, "max_recall": empty_best, "min_far": empty_best, "recall_ge_99_min_far": empty_best}
    max_recall_row = df.sort_values(["recall", "far", "precision"], ascending=[False, True, False]).iloc[0].to_dict()
    min_far_row = df.sort_values(["far", "recall", "precision"], ascending=[True, False, False]).iloc[0].to_dict()
    ge99 = df[df["recall"] >= 0.99]
    if len(ge99):
        ge99_row = ge99.sort_values(["far", "recall", "precision"], ascending=[True, False, False]).iloc[0].to_dict()
    else:
        ge99_row = {"label": label, "route": route, "model_config": model_config, "note": "no policy reached recall>=0.99"}
    best = {"label": label, "route": route, "model_config": model_config, "max_recall": max_recall_row, "min_far": min_far_row, "recall_ge_99_min_far": ge99_row}
    return df, best


def auc_safe(y: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


def infer_split_used_features(pipe: object, selected_features: List[str]) -> List[str]:
    """Map tree split feature indices back to selected feature names where possible."""
    try:
        model = pipe.named_steps["model"] if isinstance(pipe, Pipeline) else pipe
        used_idx = set()
        if hasattr(model, "estimators_"):
            # ExtraTrees / RF-style
            for est in model.estimators_:
                feats = est.tree_.feature
                for idx in feats:
                    if idx >= 0:
                        used_idx.add(int(idx))
        elif hasattr(model, "_predictors"):
            # HistGradientBoosting internal TreePredictor nodes.
            for iter_preds in model._predictors:
                for pred in iter_preds:
                    nodes = getattr(pred, "nodes", None)
                    if nodes is None:
                        continue
                    # field names can differ across sklearn versions.
                    names = nodes.dtype.names or []
                    field = "feature_idx" if "feature_idx" in names else ("feature" if "feature" in names else None)
                    if field is None:
                        continue
                    for idx in nodes[field]:
                        try:
                            idx = int(idx)
                            if idx >= 0:
                                used_idx.add(idx)
                        except Exception:
                            pass
        if not used_idx:
            return []
        return [selected_features[i] for i in sorted(used_idx) if i < len(selected_features)]
    except Exception:
        return []


def threshold_for_target_recall(y: np.ndarray, score: np.ndarray, target_recall: float) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    grid = threshold_grid(score, max_points=1200)
    best_t = float(np.min(grid))
    best_far = 1e9
    for t in grid:
        m = metrics_at_threshold(y, score, float(t))
        if m["recall"] >= target_recall:
            if m["far"] < best_far:
                best_far = m["far"]
                best_t = float(t)
    return best_t


def route_mask_for_c2(mode: str, y_train: np.ndarray, train_scores: np.ndarray) -> Tuple[np.ndarray, float, str]:
    mode = str(mode)
    if mode.startswith("recall"):
        # e.g. recall99, recall995, recall98.
        val = mode.replace("recall", "")
        if val == "99":
            target = 0.99
        elif val == "995":
            target = 0.995
        elif val == "98":
            target = 0.98
        else:
            target = float("0." + val) if val.isdigit() else 0.99
        cut = threshold_for_target_recall(y_train, train_scores, target)
        return train_scores >= cut, cut, f"c1_train_threshold_for_recall_{target}"
    if mode.startswith("quantile"):
        q = float(mode.replace("quantile", "")) / 100.0
        cut = float(np.quantile(train_scores, q))
        return train_scores >= cut, cut, f"c1_train_score_quantile_{q}"
    # conservative default, broad route.
    cut = float(np.quantile(train_scores, 0.45))
    return train_scores >= cut, cut, "c1_train_score_quantile_0.45"



def export_error_slices_for_label(label_dir: Path, df: pd.DataFrame, label: str) -> None:
    """Export FN/FP slices for the best Recall>=99% operating point.

    This uses OOF scores only. It does not use metrics as model features.
    """
    best_path = label_dir / f"v73_8e_best_operating_points__{label}.csv"
    if not best_path.exists():
        return
    best = pd.read_csv(best_path)
    if "best_kind" not in best.columns or len(best) == 0:
        return
    cand = best[best["best_kind"] == "recall_ge_99_min_far"].copy()
    cand["recall_num"] = pd.to_numeric(cand.get("recall"), errors="coerce")
    cand["far_num"] = pd.to_numeric(cand.get("far"), errors="coerce")
    cand = cand[cand["recall_num"] >= 0.99]
    if cand.empty:
        return
    row = cand.sort_values(["far_num", "recall_num"], ascending=[True, False]).iloc[0].to_dict()
    model_config = str(row.get("model_config", ""))
    route = str(row.get("route", ""))
    base_model_config = model_config.split("__")[0]
    oof_path = label_dir / f"v73_8e_oof_scores__{label}__{base_model_config}.csv"
    if not oof_path.exists():
        return
    oof = pd.read_csv(oof_path)
    if route == "C1_only":
        thr = float(row.get("threshold"))
        pred = oof["c1_oof_score"].astype(float).to_numpy() >= thr
    else:
        c1_thr = float(row.get("c1_threshold"))
        c2_thr = float(row.get("c2_threshold"))
        # route looks like C1_plus_C2_veto__recall98
        c2_mode = route.split("C1_plus_C2_veto__", 1)[1] if "C1_plus_C2_veto__" in route else model_config.split("__")[-1]
        c2_col = f"c2_oof_score__{c2_mode}"
        if c2_col not in oof.columns:
            return
        pred = (oof["c1_oof_score"].astype(float).to_numpy() >= c1_thr) & (oof[c2_col].astype(float).to_numpy() >= c2_thr)
    y = oof["y"].astype(int).to_numpy()
    err = pd.DataFrame({
        "seed": oof["seed"].astype(str),
        "sample_id": oof["sample_id"].astype(str),
        "label": label,
        "model_config": model_config,
        "route": route,
        "y": y,
        "pred": pred.astype(int),
        "c1_oof_score": oof["c1_oof_score"].astype(float),
        "error_type": np.where((y == 1) & (~pred), "FN", np.where((y == 0) & pred, "FP", "correct")),
    })
    extra_cols = [c for c in oof.columns if c.startswith("c2_oof_score__")]
    for c in extra_cols:
        err[c] = oof[c]
    err[err["error_type"].isin(["FN", "FP"])].to_csv(label_dir / f"v73_8e_error_slice_best_recall_ge99__{label}.csv", index=False)
    pd.DataFrame([row]).to_csv(label_dir / f"v73_8e_error_slice_best_recall_ge99_policy__{label}.csv", index=False)



def safe_filename_part(x: object) -> str:
    s = str(x)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:180]


def save_model_bundle(
    bundle_root: Path,
    stage: str,
    label: str,
    model_config: str,
    fold_test_seed: str,
    pipe: object,
    feature_names: List[str],
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    stage_dir = bundle_root / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    fname = "__".join([
        safe_filename_part(label),
        safe_filename_part(stage),
        safe_filename_part(model_config),
        f"fold_seed_{safe_filename_part(fold_test_seed)}",
    ]) + ".joblib"
    path = stage_dir / fname
    payload = {
        "version": "v73_8e_saved_model_bundle_v1",
        "stage": stage,
        "label": label,
        "model_config": model_config,
        "fold_test_seed": str(fold_test_seed),
        "feature_names": list(feature_names),
        "feature_count": int(len(feature_names)),
        "pipe": pipe,
        "created_at": now(),
    }
    if extra:
        payload.update(extra)
    joblib.dump(payload, path, compress=3)
    row = {
        "label": label,
        "stage": stage,
        "model_config": model_config,
        "fold_test_seed": str(fold_test_seed),
        "feature_count": int(len(feature_names)),
        "bundle_path": str(path),
    }
    if extra:
        for k, v in extra.items():
            if k not in {"pipe", "feature_names"}:
                row[k] = v
    return row


def run_label_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    label: str,
    out_dir: Path,
    c1_models: List[str],
    c2_route_modes: List[str],
    c2_neg_weight: float,
    random_seed: int,
) -> Dict[str, object]:
    """Train/evaluate one label using the FULL exact-allowed feature pool and save models.

    v73.8E intentionally separates training from split-used extraction. This script trains
    C1/C2, writes OOF scores/metrics, and saves every fold model bundle. It does NOT
    introspect tree split nodes here, so extraction/contract bugs do not force retraining.
    """
    log(f"===== label={label} =====")
    y = df[label].to_numpy(dtype=int)
    seeds = sorted(df["seed"].astype(str).unique().tolist())
    X_all = numeric_clean_frame(df, feature_cols)
    assert_no_forbidden_features(feature_cols, f"C1 full feature_cols for {label}")

    label_dir = out_dir / f"label__{label}"
    label_dir.mkdir(parents=True, exist_ok=True)
    bundle_root = label_dir / "model_bundles"
    bundle_root.mkdir(parents=True, exist_ok=True)

    all_threshold_rows = []
    all_policy_rows = []
    all_best_rows = []
    all_oof_score_rows = []
    c2_pool_rows = []
    all_config_summaries = []
    bundle_rows = []

    iterator = c1_models
    if tqdm is not None:
        iterator = tqdm(c1_models, desc=f"v73.8E TRAIN-SAVE full-feature C1/C2 configs {label}")

    for config_idx, model_name in enumerate(iterator):
        model_config = f"{model_name}_full"
        log(f"Training C1 FULL-FEATURE config {model_config} for label={label}; features={len(feature_cols)}")
        c1_oof = np.full(len(df), np.nan, dtype=float)

        # First pass: C1 OOF. Full exact-allowed feature pool is used in every fold.
        for fold_idx, test_seed in enumerate(seeds):
            test_mask = df["seed"].astype(str).to_numpy() == str(test_seed)
            train_mask = ~test_mask
            y_train = y[train_mask]
            if len(np.unique(y_train)) < 2:
                raise RuntimeError(f"C1 train fold has single class for label={label}, seed={test_seed}")
            proba, pipe = fit_predict_model(
                model_name,
                X_all.loc[train_mask, feature_cols],
                y_train,
                X_all.loc[test_mask, feature_cols],
                random_seed + fold_idx,
            )
            c1_oof[test_mask] = proba
            bundle_rows.append(save_model_bundle(
                bundle_root=bundle_root,
                stage="C1",
                label=label,
                model_config=model_config,
                fold_test_seed=str(test_seed),
                pipe=pipe,
                feature_names=feature_cols,
                extra={
                    "model_name": model_name,
                    "fold_idx": int(fold_idx),
                    "train_rows": int(np.sum(train_mask)),
                    "test_rows": int(np.sum(test_mask)),
                    "feature_mode": "full_exact_allowed_pool_no_kbest",
                },
            ))
            del pipe
            gc.collect()

        if np.any(~np.isfinite(c1_oof)):
            raise RuntimeError(f"C1 OOF contains non-finite values for {label}/{model_config}")

        # C1-only metrics.
        c1_threshold_df, c1_best = summarize_score_operating_points(y, c1_oof, "C1_only", label, model_config)
        c1_threshold_df.to_csv(label_dir / f"v73_8e_thresholds__{label}__{model_config}__C1_only.csv", index=False)
        all_threshold_rows.append(c1_threshold_df)
        for kind, row in c1_best.items():
            if isinstance(row, dict):
                rr = dict(row)
                rr["best_kind"] = kind
                rr["label"] = label
                rr["route"] = "C1_only"
                rr["model_config"] = model_config
                all_best_rows.append(rr)

        # Build C1 internal features for C2. These are generated by this same C1.
        c1_internal = pd.DataFrame({
            "v73_c1_oof_score": c1_oof,
            "v73_c1_oof_logit": safe_logit(c1_oof),
            "v73_c1_oof_rank": rank01(c1_oof),
            "v73_c1_oof_score_band": score_band(c1_oof),
            "v73_c1_oof_borderline_flag": borderline_flag(c1_oof),
        }, index=df.index)

        c2_variant_scores: Dict[str, np.ndarray] = {}
        c2_variant_meta: List[Dict[str, object]] = []
        for c2_mode in c2_route_modes:
            c2_oof_variant = np.full(len(df), np.nan, dtype=float)
            log(f"Training C2 FULL-FEATURE veto/correction for {label}/{model_config}, c2_mode={c2_mode}; raw_features={len(feature_cols)}")
            for fold_idx, test_seed in enumerate(seeds):
                test_mask = df["seed"].astype(str).to_numpy() == str(test_seed)
                train_mask = ~test_mask
                y_train = y[train_mask]
                if len(np.unique(y_train)) < 2:
                    raise RuntimeError(f"C2 train fold has single class for label={label}, seed={test_seed}")
                train_scores = c1_oof[train_mask]
                route_train_mask_local, route_cut, route_desc = route_mask_for_c2(c2_mode, y_train, train_scores)

                routed_y = y_train[route_train_mask_local]
                if len(np.unique(routed_y)) < 2 or int(np.sum(route_train_mask_local)) < 200:
                    route_train_mask_local, route_cut, route_desc = route_mask_for_c2("quantile25", y_train, train_scores)
                    routed_y = y_train[route_train_mask_local]
                if len(np.unique(routed_y)) < 2:
                    route_train_mask_local = np.ones_like(y_train, dtype=bool)
                    routed_y = y_train
                    route_desc = "all_train_rows_fallback"
                    route_cut = float("nan")

                c2_pool_rows.append({
                    "label": label, "model_config": model_config, "c2_mode": c2_mode,
                    "fold_test_seed": test_seed, "route_desc": route_desc,
                    "route_cut": route_cut,
                    "train_rows_total": int(np.sum(train_mask)),
                    "route_train_rows": int(route_train_mask_local.sum()),
                    "route_train_pos": int(np.sum(routed_y == 1)),
                    "route_train_neg": int(np.sum(routed_y == 0)),
                    "full_raw_feature_count": int(len(feature_cols)),
                    "c2_internal_feature_count": int(len(C2_INTERNAL_FEATURES)),
                    "c2_neg_weight": float(c2_neg_weight),
                })

                X_train_full = X_all.loc[train_mask]
                assert_no_forbidden_features(feature_cols, f"C2 full raw features for {label}/{model_config}/{c2_mode}/seed{test_seed}")

                X2_train_raw = X_train_full.loc[route_train_mask_local, feature_cols].reset_index(drop=True)
                X2_train_c1 = c1_internal.loc[train_mask].loc[route_train_mask_local, C2_INTERNAL_FEATURES].reset_index(drop=True)
                X2_train = pd.concat([X2_train_raw, X2_train_c1], axis=1)
                y2_train = routed_y
                sw = np.where(y2_train == 1, 1.0, float(c2_neg_weight)).astype(float)

                X2_test = pd.concat([
                    X_all.loc[test_mask, feature_cols].reset_index(drop=True),
                    c1_internal.loc[test_mask, C2_INTERNAL_FEATURES].reset_index(drop=True),
                ], axis=1)
                c2_feature_names = feature_cols + C2_INTERNAL_FEATURES
                proba2, pipe2 = fit_predict_model(
                    "hgb_shallow", X2_train, y2_train, X2_test,
                    random_seed + 1000 + fold_idx, sample_weight=sw
                )
                c2_oof_variant[test_mask] = proba2
                bundle_rows.append(save_model_bundle(
                    bundle_root=bundle_root,
                    stage="C2",
                    label=label,
                    model_config=f"{model_config}__{c2_mode}",
                    fold_test_seed=str(test_seed),
                    pipe=pipe2,
                    feature_names=c2_feature_names,
                    extra={
                        "c1_model_config": model_config,
                        "c2_mode": c2_mode,
                        "model_name": "hgb_shallow",
                        "fold_idx": int(fold_idx),
                        "train_rows_total": int(np.sum(train_mask)),
                        "route_train_rows": int(route_train_mask_local.sum()),
                        "route_train_pos": int(np.sum(routed_y == 1)),
                        "route_train_neg": int(np.sum(routed_y == 0)),
                        "route_desc": route_desc,
                        "route_cut": float(route_cut) if np.isfinite(route_cut) else None,
                        "c2_neg_weight": float(c2_neg_weight),
                        "feature_mode": "full_exact_allowed_pool_plus_this_round_c1_outputs_no_kbest",
                    },
                ))
                c2_variant_meta.append({
                    "label": label, "model_config": model_config, "c2_mode": c2_mode, "fold_test_seed": test_seed,
                    "route_desc": route_desc, "route_cut": route_cut, "route_train_rows": int(route_train_mask_local.sum()),
                    "route_train_pos": int(np.sum(routed_y == 1)), "route_train_neg": int(np.sum(routed_y == 0)),
                    "c2_neg_weight": float(c2_neg_weight),
                    "c2_full_raw_feature_count": int(len(feature_cols)),
                    "c2_internal_feature_count": int(len(C2_INTERNAL_FEATURES)),
                })
                del pipe2
                gc.collect()
            if np.any(~np.isfinite(c2_oof_variant)):
                raise RuntimeError(f"C2 OOF contains non-finite values for {label}/{model_config}/{c2_mode}")
            c2_variant_scores[c2_mode] = c2_oof_variant

        pd.DataFrame(c2_variant_meta).to_csv(label_dir / f"v73_8e_c2_route_training_meta__{label}__{model_config}.csv", index=False)

        # Summarize C2 variants and policy grid.
        oof_df = pd.DataFrame({
            "seed": df["seed"].values,
            "sample_id": df["sample_id"].values,
            "label": label,
            "model_config": model_config,
            "y": y,
            "c1_oof_score": c1_oof,
        })
        for mode, c2score in c2_variant_scores.items():
            oof_df[f"c2_oof_score__{mode}"] = c2score
        oof_df.to_csv(label_dir / f"v73_8e_oof_scores__{label}__{model_config}.csv", index=False)
        all_oof_score_rows.append(oof_df)

        c1_auc = auc_safe(y, c1_oof)
        for c2_mode, c2_oof_score in c2_variant_scores.items():
            c2_auc_by_mode = auc_safe(y, c2_oof_score)
            c2_model_config = f"{model_config}__{c2_mode}"
            c2_policy_df, c2_best = summarize_c1_c2_policy(y, c1_oof, c2_oof_score, f"C1_plus_C2_veto__{c2_mode}", label, c2_model_config)
            c2_policy_df.to_csv(label_dir / f"v73_8e_policy_grid__{label}__{c2_model_config}__C1_plus_C2_veto.csv", index=False)
            all_policy_rows.append(c2_policy_df)
            for kind, row in c2_best.items():
                if isinstance(row, dict):
                    rr = dict(row)
                    rr["best_kind"] = kind
                    rr["label"] = label
                    rr["route"] = f"C1_plus_C2_veto__{c2_mode}"
                    rr["model_config"] = c2_model_config
                    all_best_rows.append(rr)
            all_config_summaries.append({
                "label": label,
                "model_config": c2_model_config,
                "c1_model_config": model_config,
                "c2_mode": c2_mode,
                "c1_auc": c1_auc,
                "c2_auc": c2_auc_by_mode,
                "c1_feature_mode": "full_exact_allowed_pool_no_kbest",
                "c2_feature_mode": "full_exact_allowed_pool_plus_this_round_c1_outputs_no_kbest",
                "raw_feature_count": len(feature_cols),
            })

    # Persist label-level outputs.
    if all_threshold_rows:
        pd.concat(all_threshold_rows, axis=0, ignore_index=True).to_csv(label_dir / f"v73_8e_threshold_grid_all__{label}.csv", index=False)
    if all_policy_rows:
        pd.concat(all_policy_rows, axis=0, ignore_index=True).to_csv(label_dir / f"v73_8e_policy_grid_all__{label}.csv", index=False)
    if all_best_rows:
        best_df = pd.DataFrame(all_best_rows)
        best_df.to_csv(label_dir / f"v73_8e_best_operating_points__{label}.csv", index=False)
    if all_oof_score_rows:
        pd.concat(all_oof_score_rows, axis=0, ignore_index=True).to_csv(label_dir / f"v73_8e_oof_scores_all__{label}.csv", index=False)
    if c2_pool_rows:
        pd.DataFrame(c2_pool_rows).to_csv(label_dir / f"v73_8e_c2_training_pool_summary__{label}.csv", index=False)
    if all_config_summaries:
        pd.DataFrame(all_config_summaries).to_csv(label_dir / f"v73_8e_config_summary__{label}.csv", index=False)
    if bundle_rows:
        pd.DataFrame(bundle_rows).to_csv(label_dir / f"v73_8e_model_bundle_manifest__{label}.csv", index=False)

    # Error slices for the best Recall>=99% operating point per route/config.
    try:
        export_error_slices_for_label(label_dir, df, label)
    except Exception as e:
        log(f"[WARN] error slice export failed for {label}: {repr(e)}")

    return {
        "label": label,
        "positive": int(np.sum(y == 1)),
        "negative": int(np.sum(y == 0)),
        "configs": [str(m) + "_full" for m in c1_models],
        "feature_mode": "full_exact_allowed_no_kbest_train_saved_models_only",
        "best_csv": str(label_dir / f"v73_8e_best_operating_points__{label}.csv"),
        "model_bundle_manifest_csv": str(label_dir / f"v73_8e_model_bundle_manifest__{label}.csv"),
        "model_bundle_count": int(len(bundle_rows)),
    }


# v73.8E deliberately keeps this function only for provenance comparison with v73.8B.
# It is not used for training.

def parse_kbest_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def flatten_best(out_dir: Path, labels: List[str]) -> pd.DataFrame:
    frames = []
    for lab in labels:
        p = out_dir / f"label__{lab}" / f"v73_8e_best_operating_points__{lab}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=0, ignore_index=True)
    df.to_csv(out_dir / "v73_8e_best_operating_points_all_labels.csv", index=False)

    # Compact summary per label/route/kind: choose global best among configs.
    compact = []
    for (label, route, kind), sub in df.groupby(["label", "route", "best_kind"], dropna=False):
        if kind == "max_recall":
            row = sub.sort_values(["recall", "far", "precision"], ascending=[False, True, False]).iloc[0]
        elif kind == "min_far":
            row = sub.sort_values(["far", "recall", "precision"], ascending=[True, False, False]).iloc[0]
        elif kind == "recall_ge_99_min_far":
            sub2 = sub[pd.to_numeric(sub.get("recall", np.nan), errors="coerce") >= 0.99]
            if len(sub2) == 0:
                continue
            row = sub2.sort_values(["far", "recall", "precision"], ascending=[True, False, False]).iloc[0]
        else:
            continue
        compact.append(row.to_dict())
    comp = pd.DataFrame(compact)
    comp.to_csv(out_dir / "v73_8e_best_summary_compact.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="v73.8E full-feature no-detector split-used retrain/eval")
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--old-exact-csv", type=Path, default=DEFAULT_OLD_EXACT, help="Optional old v35-era deployable raw feature matrix; detector outputs still forbidden")
    parser.add_argument("--exact-feature-policy-csv", type=Path, default=DEFAULT_EXACT_FEATURE_POLICY, help="Package-owned exact allowed source+feature list. Required for v73.8E strict mode.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--labels", type=str, default=",".join(LABELS_DEFAULT))
    parser.add_argument("--kbest-list", type=str, default="", help="Deprecated; ignored in v73.8E. Full exact-allowed features are used.")
    parser.add_argument("--c1-models", type=str, default="hgb,extra_trees")
    parser.add_argument("--c2-kbest", type=int, default=0, help="Deprecated; ignored in v73.8E. C2 uses full exact-allowed features plus this-round C1 outputs.")
    parser.add_argument("--c2-route-modes", type=str, default="recall99,recall98,quantile45", help="C2 training route modes tied to the same C1 output")
    parser.add_argument("--c2-neg-weight", type=float, default=2.0, help="Negative sample weight inside C2 routed training set")
    parser.add_argument("--quick-rows", type=int, default=0, help="Use first N rows for a fast smoke test; 0 means full data")
    parser.add_argument("--random-seed", type=int, default=773003)
    parser.add_argument("--max-consistency-cols-per-family", type=int, default=250)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log("v73.8E full-feature no-detector split-used retrain/eval started")
    log(f"matrix_csv={args.matrix_csv}")
    log(f"label_csv={args.label_csv}")
    log(f"old_exact_csv={args.old_exact_csv}")
    log(f"output_dir={out_dir}")

    if not args.matrix_csv.exists():
        raise SystemExit(f"[ERROR] matrix CSV not found: {args.matrix_csv}")
    if not args.label_csv.exists():
        raise SystemExit(f"[ERROR] label CSV not found: {args.label_csv}")
    if args.old_exact_csv and not args.old_exact_csv.exists():
        raise SystemExit(f"[ERROR] old-exact CSV not found: {args.old_exact_csv}")

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    c1_models = [x.strip() for x in args.c1_models.split(",") if x.strip()]
    if not c1_models:
        raise RuntimeError("No C1 models requested")
    # v73.8E full-feature mode: no KBest preselection. Each C1 model sees the full exact-allowed online feature pool.

    # 1) Full manifest audit before reading full matrix.
    cols = read_header(args.matrix_csv)
    source_to_cols = {"v63_full": cols}
    if args.old_exact_csv and args.old_exact_csv.exists():
        source_to_cols["old_exact"] = read_header(args.old_exact_csv)
    manifest, manifest_allowed_by_source = build_feature_manifest(source_to_cols, out_dir)
    allowed_by_source = load_exact_allowed_feature_policy(args.exact_feature_policy_csv, source_to_cols, manifest, out_dir)
    allowed_features = []
    for src in ["v63_full", "old_exact"]:
        allowed_features.extend(allowed_by_source.get(src, []))
    summary = {
        "version": "v73_8e_fullfeature_splitused",
        "policy": "C1 uses only online-available raw/derived/consistency features, now including v35-era old exact deployable raw features but no v35 detector outputs. C2 may use this round's C1 OOF/current outputs plus allowed online features. No previous detector/classifier outputs are allowed.",
        "matrix_csv": str(args.matrix_csv),
        "old_exact_csv": str(args.old_exact_csv) if args.old_exact_csv else "",
        "label_csv": str(args.label_csv),
        "exact_feature_policy_csv": str(args.exact_feature_policy_csv),
        "matrix_column_count": len(cols),
        "old_exact_column_count": len(source_to_cols.get("old_exact", [])),
        "allowed_online_feature_count_before_generated": int(sum(len(v) for v in allowed_by_source.values())),
        "labels": labels,
        "feature_selection_mode": "full_exact_allowed_pool_no_kbest",
        "c1_models": c1_models,
        "c2_feature_mode": "full_exact_allowed_pool_plus_this_round_c1_outputs_no_kbest",
        "c2_route_modes": args.c2_route_modes,
        "c2_neg_weight": args.c2_neg_weight,
        "quick_rows": args.quick_rows,
    }
    role_counts = manifest["role"].value_counts().to_dict()
    family_counts = manifest.loc[manifest["allowed_for_c1"] == 1, "family"].value_counts().to_dict()
    summary["manifest_role_counts"] = {str(k): int(v) for k, v in role_counts.items()}
    summary["allowed_family_counts"] = {str(k): int(v) for k, v in family_counts.items()}
    summary["allowed_count_by_source"] = {str(k): int(len(v)) for k, v in allowed_by_source.items()}

    assert_no_forbidden_features(allowed_features, "allowed_features_from_manifest")

    # 2) Load selected matrix columns from v63/full and old-exact raw matrix, then join labels.
    df, loaded_feature_cols = load_matrices_and_labels(args.matrix_csv, args.old_exact_csv, args.label_csv, allowed_by_source, labels, args.quick_rows, out_dir)
    allowed_features = loaded_feature_cols
    summary["merged_rows"] = int(len(df))
    summary["unique_seeds"] = sorted(df["seed"].astype(str).unique().tolist())
    summary["unique_seed_count"] = len(summary["unique_seeds"])
    summary["label_positive_counts"] = {lab: int(df[lab].sum()) for lab in labels}

    # 3) Online-computable generated consistency features.
    base_X = numeric_clean_frame(df, allowed_features)
    gen_X = make_generated_consistency_features(base_X, out_dir, max_cols_per_family=args.max_consistency_cols_per_family)
    for c in gen_X.columns:
        # Add generated features to df and manifest-like audit.
        df[c] = gen_X[c].values
    generated_cols = list(gen_X.columns)
    assert_no_forbidden_features(generated_cols, "generated_consistency_features")
    feature_cols = allowed_features + generated_cols
    summary["generated_consistency_feature_count"] = len(generated_cols)
    summary["total_c1_allowed_feature_count"] = len(feature_cols)
    pd.DataFrame({"feature": feature_cols}).to_csv(out_dir / "v73_8e_final_c1_allowed_feature_list.csv", index=False)

    # 4) Strong audit: no forbidden features in C1 list.
    assert_no_forbidden_features(feature_cols, "final C1 feature list")
    # C2 can additionally use only internally produced C1 outputs.
    assert_no_forbidden_features(feature_cols + C2_INTERNAL_FEATURES, "final C2 feature list including C1 internal outputs")

    independent_audit = {
        "version": "v73_8e_independent_online_model_audit",
        "c1_uses_previous_detector_or_classifier_outputs": False,
        "c1_uses_any_model_output": False,
        "c2_uses_previous_detector_or_classifier_outputs": False,
        "c2_uses_this_round_c1_outputs": True,
        "two_c1_main_models": False,
        "allowed_c2_internal_features": C2_INTERNAL_FEATURES,
        "forbidden_detector_or_leakage_columns_count": int((manifest["role"].str.contains("forbidden", na=False)).sum()),
        "c1_feature_count": len(feature_cols),
        "generated_consistency_feature_count": len(generated_cols),
        "exact_feature_policy_csv": str(args.exact_feature_policy_csv),
        "feature_selection_mode": "frozen_exact_source_feature_allowlist; full-feature training; no KBest; no runtime fuzzy matching",
        "preprocessing": "SafeFiniteClipper caps finite values to +/-1e6 before SafeMedianImputer; all-NaN train columns map to 0; feature width/order is preserved",
        "v35_method_ported": ["old_exact_raw_online_features", "seed_level_loso_oof", "high_recall_threshold_sweep", "HGB_and_ExtraTrees_retrained", "safe_missing_default_handling"],
        "v35_outputs_forbidden": ["score__", "alarm__", "threshold__", "hgbmax", "lineA_old", "lineA_speed"],
        "pass": True,
    }
    (out_dir / "v73_8e_independent_online_model_audit.json").write_text(json.dumps(independent_audit, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5) Train/evaluate each label.
    label_outputs = []
    for lab in labels:
        if lab not in df.columns:
            raise RuntimeError(f"Merged data missing label {lab}")
        if df[lab].nunique() < 2:
            raise RuntimeError(f"Label {lab} has a single class; cannot train")
        label_outputs.append(run_label_pipeline(
            df=df,
            feature_cols=feature_cols,
            label=lab,
            out_dir=out_dir,
            c1_models=c1_models,
            c2_route_modes=[x.strip() for x in args.c2_route_modes.split(",") if x.strip()],
            c2_neg_weight=args.c2_neg_weight,
            random_seed=args.random_seed,
        ))

    best_all = flatten_best(out_dir, labels)
    if not best_all.empty:
        log("Best operating point compact summary:")
        comp = pd.read_csv(out_dir / "v73_8e_best_summary_compact.csv")
        print(comp.to_string(index=False), flush=True)

    # 6) Save source map for the separate split-used extraction step.
    source_rows = []
    for src, feats in allowed_by_source.items():
        for f in feats:
            source_rows.append({"feature": str(f), "source": str(src), "kind": "raw_or_sanitized_allowed_online"})
    for f in generated_cols:
        source_rows.append({"feature": str(f), "source": "generated_consistency", "kind": "generated_online_consistency"})
    for f in C2_INTERNAL_FEATURES:
        source_rows.append({"feature": str(f), "source": "this_round_c1_internal", "kind": "c2_internal_from_this_round_c1"})
    pd.DataFrame(source_rows).drop_duplicates(["feature", "source"]).to_csv(out_dir / "v73_8e_feature_source_map.csv", index=False)

    # Split-used extraction is intentionally not performed here. Use:
    #   scripts/run_v73_8e_extract_splitused_contract.sh
    # This prevents extraction/contract bugs from forcing a retrain.
    summary["label_outputs"] = label_outputs
    summary["end_time"] = now()
    (out_dir / "v73_8e_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log("v73.8E full-feature training + model save completed successfully; run extraction step next")




# -----------------------------------------------------------------------------
# v73.16 selected-feature entry point
# -----------------------------------------------------------------------------
# The functions above are intentionally kept from v73.8E to preserve the exact
# training/evaluation method: same C1/C2 model configs, seed-level LOSO OOF,
# C2 routing, negative weighting, preprocessing, threshold sweep, and bundle format.
# Only the feature source is changed: instead of rebuilding a full v63+old_exact
# feature pool, this script consumes the v73 standalone inference model-ready CSV
# and uses the exact selected feature contract.

LABEL_ALIASES_V73_16 = {
    "harmful_l2_2s": "harmful",
    "severe_l2_2s": "severe_harmful",
}


def load_selected_contract_features(contract_csv: Path, df_columns: Sequence[str], include_aux: bool, out_dir: Path) -> Tuple[List[str], pd.DataFrame]:
    if not contract_csv.exists():
        raise RuntimeError(f"Feature contract CSV not found: {contract_csv}")
    contract = pd.read_csv(contract_csv)
    if "feature" not in contract.columns:
        raise RuntimeError(f"Feature contract CSV missing 'feature' column: {contract_csv}")
    features = contract["feature"].astype(str).drop_duplicates().tolist()
    if include_aux:
        aux = sorted([c for c in df_columns if str(c).startswith("v73aux__")])
        features.extend(aux)
    missing = [f for f in features if f not in set(map(str, df_columns))]
    if missing:
        pd.DataFrame({"missing_feature": missing}).to_csv(out_dir / "v73_16_exact_train_missing_features.csv", index=False)
        raise RuntimeError(f"Model-ready CSV missing {len(missing)} selected features; first={missing[:30]}")
    assert_no_forbidden_features(features, "v73.16 selected-feature exact contract")
    return features, contract


def ensure_requested_labels(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    df = df.copy()
    for lab in labels:
        if lab in df.columns:
            continue
        alias = LABEL_ALIASES_V73_16.get(lab)
        if alias and alias in df.columns:
            df[lab] = df[alias].astype(int)
            continue
        raise RuntimeError(f"Requested label {lab!r} missing. Available alias map={LABEL_ALIASES_V73_16}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="v73.16 selected-feature training using the exact v73.8E method")
    parser.add_argument("--model-ready-csv", type=Path, required=True, help="v73 standalone model-ready CSV from inference export")
    parser.add_argument("--feature-contract-csv", type=Path, required=True, help="Selected feature contract CSV, normally resources/v73_9_hgb_online_feature_contract.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=str, default="harmful_l2_2s,severe_l2_2s")
    parser.add_argument("--c1-models", type=str, default="hgb,extra_trees")
    parser.add_argument("--c2-route-modes", type=str, default="recall99,recall98,quantile45")
    parser.add_argument("--c2-neg-weight", type=float, default=2.0)
    parser.add_argument("--quick-rows", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=773003)
    parser.add_argument("--include-aux", action="store_true", help="Optional; changes the feature set by adding v73aux__ missingness columns. Off by default to match selected-feature contract.")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log("v73.16 selected-feature exact-v73.8E-method training started")
    log(f"model_ready_csv={args.model_ready_csv}")
    log(f"feature_contract_csv={args.feature_contract_csv}")
    log(f"output_dir={out_dir}")
    if not args.model_ready_csv.exists():
        raise SystemExit(f"[ERROR] model-ready CSV not found: {args.model_ready_csv}")
    if not args.feature_contract_csv.exists():
        raise SystemExit(f"[ERROR] feature contract CSV not found: {args.feature_contract_csv}")

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    c1_models = [x.strip() for x in args.c1_models.split(",") if x.strip()]
    if not c1_models:
        raise RuntimeError("No C1 models requested")

    # Load inference output. This file already contains the generated consistency
    # features produced by the standalone data contract. No KBest or feature
    # importance selection is performed here.
    df = pd.read_csv(args.model_ready_csv)
    if args.quick_rows and args.quick_rows > 0:
        df = df.head(args.quick_rows).copy()
    if "seed" not in df.columns or "sample_id" not in df.columns:
        raise RuntimeError("Model-ready CSV must contain seed and sample_id for seed-level LOSO OOF")
    df = ensure_requested_labels(df, labels)
    feature_cols, contract = load_selected_contract_features(args.feature_contract_csv, df.columns, args.include_aux, out_dir)
    assert_no_forbidden_features(feature_cols, "v73.16 final C1 selected feature list")
    assert_no_forbidden_features(feature_cols + C2_INTERNAL_FEATURES, "v73.16 final C2 feature list including this-round C1 outputs")

    # Strong method audit: these settings must match v73.8E except for the selected
    # feature list coming from v73.8F/v73.9 contract.
    summary = {
        "version": "v73_16_selected_feature_exact_v73_8e_method",
        "method_equivalence_to_v73_8e": {
            "c1_model_configs": "same make_model(): hgb and extra_trees defaults unless overridden",
            "hgb_params": {"max_iter": 220, "learning_rate": 0.055, "max_leaf_nodes": 31, "l2_regularization": 0.05, "early_stopping": True, "validation_fraction": 0.15},
            "extra_trees_params": {"n_estimators": 260, "max_depth": 7, "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": "balanced_subsample"},
            "preprocessing": "SafeFiniteClipper(+/-1e6) + SafeMedianImputer(fill=0), same as v73.8E",
            "oof_split": "seed-level leave-one-seed-out over sorted unique seed values, same as v73.8E",
            "c2_model": "hgb_shallow, same as v73.8E",
            "c2_route_modes": args.c2_route_modes,
            "c2_neg_weight": float(args.c2_neg_weight),
            "threshold_sweep": "same threshold_grid / recall>=99 min FAR policy search as v73.8E",
            "no_kbest": True,
            "difference": "feature_cols are the selected v73.8F/v73.9 standalone contract features, not the full v63+old_exact pool",
        },
        "model_ready_csv": str(args.model_ready_csv),
        "feature_contract_csv": str(args.feature_contract_csv),
        "output_dir": str(out_dir),
        "labels": labels,
        "c1_models": c1_models,
        "c2_route_modes": args.c2_route_modes,
        "c2_neg_weight": float(args.c2_neg_weight),
        "quick_rows": int(args.quick_rows),
        "include_aux": bool(args.include_aux),
        "merged_rows": int(len(df)),
        "unique_seeds": sorted(df["seed"].astype(str).unique().tolist()),
        "unique_seed_count": int(df["seed"].astype(str).nunique()),
        "label_positive_counts": {lab: int(df[lab].sum()) for lab in labels},
        "feature_selection_mode": "selected_exact_contract_no_kbest; same v73.8E training method",
        "total_c1_allowed_feature_count": int(len(feature_cols)),
    }
    pd.DataFrame({"feature": feature_cols}).to_csv(out_dir / "v73_16_final_c1_selected_feature_list.csv", index=False)

    independent_audit = {
        "version": "v73_16_independent_online_model_audit",
        "c1_uses_previous_detector_or_classifier_outputs": False,
        "c1_uses_any_model_output": False,
        "c2_uses_previous_detector_or_classifier_outputs": False,
        "c2_uses_this_round_c1_outputs": True,
        "two_c1_main_models": False,
        "allowed_c2_internal_features": C2_INTERNAL_FEATURES,
        "c1_feature_count": len(feature_cols),
        "feature_selection_mode": "selected_exact_contract_no_kbest",
        "training_method_matches_v73_8e_except_feature_count": True,
        "pass": True,
    }
    (out_dir / "v73_16_independent_online_model_audit.json").write_text(json.dumps(independent_audit, indent=2, ensure_ascii=False), encoding="utf-8")

    label_outputs = []
    for lab in labels:
        if df[lab].nunique() < 2:
            raise RuntimeError(f"Label {lab} has a single class; cannot train")
        label_outputs.append(run_label_pipeline(
            df=df,
            feature_cols=feature_cols,
            label=lab,
            out_dir=out_dir,
            c1_models=c1_models,
            c2_route_modes=[x.strip() for x in args.c2_route_modes.split(",") if x.strip()],
            c2_neg_weight=args.c2_neg_weight,
            random_seed=args.random_seed,
        ))

    best_all = flatten_best(out_dir, labels)
    if not best_all.empty and (out_dir / "v73_8e_best_summary_compact.csv").exists():
        comp = pd.read_csv(out_dir / "v73_8e_best_summary_compact.csv")
        comp.to_csv(out_dir / "v73_16_best_summary_compact.csv", index=False)
        log("Best operating point compact summary:")
        print(comp.to_string(index=False), flush=True)

    # Source map for split-used extraction. Also write the v73_8e-compatible filename
    # because the exact extractor from v73.8E expects it.
    source_rows = []
    source_map = {}
    if "source" in contract.columns:
        source_map = dict(zip(contract["feature"].astype(str), contract["source"].astype(str)))
    for f in feature_cols:
        if f.startswith("v73aux__"):
            source = "online_safe_aux_missingness"
            kind = "optional_aux_online_missingness"
        else:
            source = source_map.get(f, "selected_model_ready_contract")
            kind = "selected_online_model_ready_feature"
        source_rows.append({"feature": f, "source": source, "kind": kind})
    for f in C2_INTERNAL_FEATURES:
        source_rows.append({"feature": str(f), "source": "this_round_c1_internal", "kind": "c2_internal_from_this_round_c1"})
    src_df = pd.DataFrame(source_rows).drop_duplicates(["feature", "source"])
    src_df.to_csv(out_dir / "v73_16_feature_source_map.csv", index=False)
    src_df.to_csv(out_dir / "v73_8e_feature_source_map.csv", index=False)

    summary["label_outputs"] = label_outputs
    summary["end_time"] = now()
    (out_dir / "v73_16_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log("v73.16 selected-feature exact-v73.8E-method training completed successfully")


if __name__ == "__main__":
    main()
