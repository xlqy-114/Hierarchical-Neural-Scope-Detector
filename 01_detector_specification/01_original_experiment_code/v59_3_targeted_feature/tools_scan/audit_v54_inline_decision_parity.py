#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math, tarfile, tempfile, shutil
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import joblib


def load_detector_artifacts(artifact: Path, detectors_csv: str = "") -> Dict[str, Dict[str, Any]]:
    tmp = Path(tempfile.mkdtemp(prefix="v54_parity_artifact_"))
    try:
        with tarfile.open(artifact, "r:gz") as tf:
            tf.extractall(tmp)
        allow = {x.strip() for x in detectors_csv.split(",") if x.strip()}
        out: Dict[str, Dict[str, Any]] = {}
        for model_p in tmp.rglob("model.joblib"):
            d = model_p.parent
            name = d.name
            if allow and name not in allow:
                continue
            feat_p = d / "feature_columns.json"
            thr_p = d / "threshold.json"
            card_p = d / "model_card.json"
            if not feat_p.is_file() or not thr_p.is_file():
                continue
            threshold_info = json.loads(thr_p.read_text(encoding="utf-8"))
            card = json.loads(card_p.read_text(encoding="utf-8")) if card_p.is_file() else {}
            out[name] = {
                "model": joblib.load(model_p),
                "feature_columns": json.loads(feat_p.read_text(encoding="utf-8")),
                "threshold": float(threshold_info.get("threshold", threshold_info.get("selected_threshold", 0.5))),
                "clip_abs": float(card.get("clip_abs", threshold_info.get("clip_abs", 1_000_000.0))),
            }
        if not out:
            raise RuntimeError(f"no detector artifacts loaded from {artifact}")
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def score_model(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba, dtype=float).reshape(-1)
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X), dtype=float).reshape(-1)
        return 1.0 / (1.0 + np.exp(-z))
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-matrix", required=True)
    ap.add_argument("--score-cache", required=True)
    ap.add_argument("--detector-artifact", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--detectors", default="")
    ap.add_argument("--score-atol", type=float, default=1e-9)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    dets = load_detector_artifacts(Path(args.detector_artifact), args.detectors)
    fm = pd.read_csv(args.feature_matrix, low_memory=False)
    sc = pd.read_csv(args.score_cache, low_memory=False)
    key_cols = ["seed", "sample_id", "row_source"]
    for k in key_cols:
        fm[k] = fm[k].astype(str)
        sc[k] = sc[k].astype(str)
    summaries = []
    mismatches = []
    for det, info in sorted(dets.items()):
        cols = [str(c) for c in info["feature_columns"]]
        missing = [c for c in cols if c not in fm.columns]
        det_sc = sc[sc["detector"].astype(str) == det].copy()
        merged = fm.merge(det_sc, on=key_cols, how="inner", validate="one_to_one")
        row_count_match = int(len(merged) == len(det_sc) == len(fm))
        if missing:
            summaries.append({"detector": det, "rows": int(len(merged)), "row_count_match": row_count_match, "missing_matrix_columns": len(missing), "score_mismatch_count": None, "alarm_mismatch_count": None, "max_abs_score_diff": None})
            if args.strict:
                continue
        vals = []
        for c in cols:
            s = pd.to_numeric(merged[c], errors="coerce").astype(float)
            s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            clip_abs = float(info.get("clip_abs", 1_000_000.0))
            if clip_abs and clip_abs > 0:
                s = s.clip(-clip_abs, clip_abs)
            vals.append(s.to_numpy())
        X = pd.DataFrame(np.vstack(vals).T, columns=cols)
        score = score_model(info["model"], X)
        threshold = float(info["threshold"])
        alarm = (score >= threshold).astype(int)
        inline_score = pd.to_numeric(merged["score"], errors="coerce").to_numpy(dtype=float)
        inline_alarm = pd.to_numeric(merged["alarm"], errors="coerce").fillna(-1).astype(int).to_numpy()
        diff = np.abs(score - inline_score)
        score_bad = np.where(~np.isfinite(diff) | (diff > args.score_atol))[0]
        alarm_bad = np.where(alarm != inline_alarm)[0]
        summaries.append({
            "detector": det,
            "rows": int(len(merged)),
            "row_count_match": row_count_match,
            "missing_matrix_columns": len(missing),
            "score_mismatch_count": int(len(score_bad)),
            "alarm_mismatch_count": int(len(alarm_bad)),
            "max_abs_score_diff": float(np.nanmax(diff)) if len(diff) else None,
            "mean_abs_score_diff": float(np.nanmean(diff)) if len(diff) else None,
            "threshold": threshold,
        })
        bad_idx = list(score_bad[:20]) + [i for i in alarm_bad[:20] if i not in set(score_bad[:20])]
        for i in bad_idx:
            row = merged.iloc[int(i)]
            mismatches.append({"detector": det, "seed": row.get("seed", ""), "sample_id": row.get("sample_id", ""), "matrix_score": float(score[int(i)]), "inline_score": float(inline_score[int(i)]), "abs_diff": float(diff[int(i)]), "matrix_alarm": int(alarm[int(i)]), "inline_alarm": int(inline_alarm[int(i)])})
    summary_df = pd.DataFrame(summaries)
    mismatch_df = pd.DataFrame(mismatches)
    summary_csv = outdir / "v54_inline_decision_parity_summary.csv"
    mismatch_csv = outdir / "v54_inline_decision_parity_mismatches.csv"
    summary_df.to_csv(summary_csv, index=False)
    mismatch_df.to_csv(mismatch_csv, index=False)
    score_mismatch_total = 0
    alarm_mismatch_total = 0
    missing_matrix_columns_total = 0
    row_count_all_match = True
    max_abs_score_diff = 0.0
    mean_abs_score_diff_max = 0.0
    if not summary_df.empty:
        score_mismatch_total = int(pd.to_numeric(summary_df.get("score_mismatch_count", 0), errors="coerce").fillna(0).astype(int).sum())
        alarm_mismatch_total = int(pd.to_numeric(summary_df.get("alarm_mismatch_count", 0), errors="coerce").fillna(0).astype(int).sum())
        missing_matrix_columns_total = int(pd.to_numeric(summary_df.get("missing_matrix_columns", 0), errors="coerce").fillna(0).astype(int).sum())
        row_count_all_match = bool((pd.to_numeric(summary_df.get("row_count_match", 0), errors="coerce").fillna(0).astype(int) == 1).all())
        max_abs_score_diff = float(pd.to_numeric(summary_df.get("max_abs_score_diff", 0.0), errors="coerce").fillna(0.0).max())
        mean_abs_score_diff_max = float(pd.to_numeric(summary_df.get("mean_abs_score_diff", 0.0), errors="coerce").fillna(0.0).max())

    # v54 policy: strict engineering pass is decision parity, not exact floating-score parity.
    # Scores are retained as diagnostics because CSV/JSON round-trip can move a tiny number of
    # tree-model probabilities without changing any detector alarm.  A run fails only if the
    # detector decision differs, feature columns are missing, or row alignment is broken.
    decision_parity_pass = bool(row_count_all_match and missing_matrix_columns_total == 0 and alarm_mismatch_total == 0)
    score_exact_pass = bool(decision_parity_pass and score_mismatch_total == 0)

    out = {
        "version": "v54_inline_decision_parity_audit",
        "feature_matrix": str(args.feature_matrix),
        "score_cache": str(args.score_cache),
        "detectors": list(dets.keys()),
        "score_atol": args.score_atol,
        "policy": "decision_parity_pass: alarm parity must be exact; score mismatches are diagnostics unless they change alarm.",
        "pass": decision_parity_pass,
        "decision_parity_pass": decision_parity_pass,
        "score_exact_pass": score_exact_pass,
        "row_count_all_match": row_count_all_match,
        "missing_matrix_columns_total": missing_matrix_columns_total,
        "score_mismatch_total": score_mismatch_total,
        "alarm_mismatch_total": alarm_mismatch_total,
        "max_abs_score_diff": max_abs_score_diff,
        "max_mean_abs_score_diff_by_detector": mean_abs_score_diff_max,
        "outputs": {"summary_csv": str(summary_csv), "mismatches_csv": str(mismatch_csv)},
    }
    (outdir / "v54_inline_decision_parity_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.strict and not decision_parity_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
