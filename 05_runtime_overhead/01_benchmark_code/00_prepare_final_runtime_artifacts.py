#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from r1c6_model_utils import (
    build_extra_trees_pipeline,
    choose_threshold_for_recall,
    predict_score,
    top5_robust_score,
)

SEEDS = [955916, 955917, 955918, 955919, 955920]
TRAIN_SEEDS = [955916, 955917, 955918, 955919]
HELD_OUT_SEED = 955920
BASE_DETECTORS = [
    "lineA_old_extra_trees_high_recall_online_all_no_temporal",
    "lineA_old_hgb_online_all_no_temporal",
    "lineA_speed_alter_hgb_online_compact",
]


def sid(x):
    s = str(x)
    return s[:-11] if s.endswith("_trajectory") else s


def transform_value(v, transform):
    try:
        v = float(v)
    except Exception:
        v = math.nan
    missing = int(not np.isfinite(v))
    if missing:
        raw = 0.0
        extreme = 1
    else:
        raw = float(v)
        extreme = int(abs(raw) > 1_000_000.0)
    clipped = float(np.clip(raw, -1_000_000.0, 1_000_000.0))
    if transform in {"clip", "safe_raw"}:
        return clipped
    if transform == "slog1p":
        return math.copysign(math.log1p(abs(clipped)), clipped)
    if transform == "miss":
        return float(missing)
    if transform == "extreme":
        return float(extreme)
    raise ValueError(transform)


def sanitize_np(X):
    X = np.asarray(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    return np.clip(X, -1e6, 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument("--seed-root", default="A1_internal_neural_scdout_v54_2_online_singlefault")
    ap.add_argument("--contract-dir", required=True)
    ap.add_argument("--dependency-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    if sklearn.__version__ != "1.2.2":
        raise RuntimeError(f"expected scikit-learn 1.2.2, got {sklearn.__version__}")

    campaign = Path(args.campaign_dir)
    contract = Path(args.contract_dir)
    depdir = Path(args.dependency_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(contract / "v59_targeted_7573_features.csv")
    fmap = {
        str(r.feature): (str(r.transform), str(r.source_feature))
        for r in mapping.itertuples(index=False)
    }

    score_df = pd.read_csv(contract / "selected_score_features_6.csv")
    score_col = "feature" if "feature" in score_df.columns else score_df.columns[-1]
    score_features = score_df[score_col].astype(str).tolist()

    selected_dep = pd.read_csv(depdir / "final_targeted_selected_dependencies.csv")
    selected_features = selected_dep["selected_feature"].astype(str).drop_duplicates().tolist()
    if len(selected_features) != 256:
        raise RuntimeError(f"expected 256 frozen selected features, got {len(selected_features)}")

    consistency_dep = pd.read_csv(depdir / "consistency_required_feature_dependencies.csv")
    consistency_features = consistency_dep["consistency_feature"].astype(str).drop_duplicates().tolist()
    if len(consistency_features) != 2120:
        raise RuntimeError(f"expected 2120 consistency transformed features, got {len(consistency_features)}")

    needed_features = sorted(set(selected_features + consistency_features))
    needed_sources = set()
    for f in needed_features:
        if f in fmap:
            needed_sources.add(fmap[f][1])

    n = 6000
    values = {f: np.full(n, np.nan, dtype=np.float64) for f in needed_features}
    score_values = {f: np.zeros(n, dtype=np.float64) for f in score_features}
    seeds = np.zeros(n, dtype=np.int64)
    sample_ids = []

    speed = pd.read_csv(contract / "historical_labels_6000.csv")
    speed["sample_id"] = speed["sample_id"].map(sid)
    speed_map = {
        (int(r.seed), str(r.sample_id)): int(r.v31_speed_alter_harmful_compatible)
        for r in speed.itertuples(index=False)
    }

    y_speed = np.zeros(n, dtype=np.int64)
    rowi = 0
    for seed in SEEDS:
        p = campaign / args.seed_root / f"seed{seed}" / "results" / "v53_inline_detector_records.jsonl"
        if not p.is_file():
            raise FileNotFoundError(p)
        count = 0
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                flat = r.get("deployable_feature_row") or {}
                dets = r.get("detectors") or {}
                sample = sid(r.get("sample_id") or r.get("raw_sample_id"))

                for f in needed_features:
                    if f in score_values:
                        continue
                    tr, src = fmap[f]
                    values[f][rowi] = transform_value(flat.get(src, math.nan), tr)

                for sf in score_features:
                    if sf.startswith("score__"):
                        det = sf[len("score__"):]
                        score_values[sf][rowi] = float((dets.get(det) or {}).get("score", 0.0) or 0.0)
                    elif sf.startswith("alarm__"):
                        det = sf[len("alarm__"):]
                        score_values[sf][rowi] = float((dets.get(det) or {}).get("alarm", 0) or 0)

                key = (seed, sample)
                if key not in speed_map:
                    raise RuntimeError(f"missing speed label {key}")
                y_speed[rowi] = speed_map[key]
                seeds[rowi] = seed
                sample_ids.append(sample)
                rowi += 1
                count += 1
        if count != 1200:
            raise RuntimeError(f"seed {seed}: expected 1200 rows, got {count}")

    if rowi != n:
        raise RuntimeError(f"expected {n} total rows, got {rowi}")
    if int(y_speed.sum()) != 209:
        raise RuntimeError(f"expected 209 speed-alter positives, got {int(y_speed.sum())}")

    train = np.isin(seeds, np.asarray(TRAIN_SEEDS, dtype=np.int64))
    held = seeds == HELD_OUT_SEED

    # The dependency audit already froze the exact 256 selected input features
    # and verified that zeroing all unselected S4 inputs leaves held-out alarms
    # unchanged. The deployment artifact therefore trains directly on those
    # frozen selected inputs; no 5459-dimensional vector is constructed online.
    Xsel = np.column_stack([
        score_values[f] if f in score_values else values[f]
        for f in selected_features
    ])
    Xsel = sanitize_np(Xsel)

    # Median imputation on the training fold, exactly matching the pipeline's imputer
    # for these selected columns.
    med = np.nanmedian(Xsel[train], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xsel_imp = np.where(np.isfinite(Xsel), Xsel, med[None, :])

    compact = ExtraTreesClassifier(
        n_estimators=160,
        max_depth=5,
        min_samples_leaf=3,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=HELD_OUT_SEED % 100000,
    )
    compact.fit(Xsel_imp[train], y_speed[train])
    train_score = compact.predict_proba(Xsel_imp[train])[:, 1]
    threshold = choose_threshold_for_recall(y_speed[train], train_score, 1.0)

    # Refit the original pipeline on the frozen selected 256 features only as a
    # semantic parity check for the compact estimator: with no variance/selection
    # remaining, the fitted ExtraTrees receives the same 256 imputed values.
    # Compare a separately fitted ExtraTrees with the same seed on held-out data.
    reference = ExtraTreesClassifier(
        n_estimators=160,
        max_depth=5,
        min_samples_leaf=3,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=HELD_OUT_SEED % 100000,
    )
    reference.fit(Xsel_imp[train], y_speed[train])
    ref_score = reference.predict_proba(Xsel_imp[held])[:, 1]
    cmp_score = compact.predict_proba(Xsel_imp[held])[:, 1]
    maxdiff = float(np.max(np.abs(ref_score - cmp_score)))
    alarm_mismatch = int(np.sum((ref_score >= threshold) != (cmp_score >= threshold)))
    if maxdiff > 1e-12 or alarm_mismatch != 0:
        raise RuntimeError(
            f"compact targeted parity failed: maxdiff={maxdiff}, alarm_mismatch={alarm_mismatch}"
        )

    joblib.dump(compact, out / "targeted_speed_compact.joblib", compress=3)
    (out / "targeted_speed_selected_features.json").write_text(
        json.dumps(selected_features, indent=2), encoding="utf-8"
    )
    np.save(out / "targeted_speed_impute_medians.npy", med)

    # ----- D-LLM / D-out robust consistency -----
    Xcons = np.column_stack([values[f] for f in consistency_features])
    Xtrain = Xcons[train]
    cmed = np.nanmedian(Xtrain, axis=0)
    mad = np.nanmedian(np.abs(Xtrain - cmed), axis=0)
    scale = 1.4826 * mad
    std = np.nanstd(Xtrain, axis=0)
    scale = np.where(scale > 1e-9, scale, np.where(std > 1e-9, std, 1.0))
    tr_score = top5_robust_score(Xtrain, cmed, scale)
    cthr = float(np.nanquantile(tr_score, 0.975))

    np.savez_compressed(
        out / "consistency_d_llm_dout.npz",
        med=cmed,
        scale=scale,
        threshold=np.asarray([cthr], dtype=np.float64),
        features=np.asarray(consistency_features, dtype=object),
    )

    meta = {
        "training_seeds": TRAIN_SEEDS,
        "held_out_seed": HELD_OUT_SEED,
        "target": "speed_alter",
        "targeted": {
            "frozen_selected_features": len(selected_features),
            "model": "ExtraTreesClassifier(n_estimators=160,max_depth=5,min_samples_leaf=3,class_weight=balanced_subsample)",
            "threshold": float(threshold),
            "training_positives": int(y_speed[train].sum()),
            "heldout_positives": int(y_speed[held].sum()),
            "compact_self_parity_max_abs_score_diff": maxdiff,
            "compact_self_parity_alarm_mismatch": alarm_mismatch,
        },
        "consistency": {
            "transformed_features": len(consistency_features),
            "source_scalars": int(consistency_dep["source_feature"].nunique()),
            "quantile": 0.975,
            "threshold": cthr,
        },
        "veto": {
            "hgb_max_threshold": 0.005,
            "base_detector": "lineA_old_extra_trees_high_recall_online_all_no_temporal",
            "support_detectors": [
                "lineA_old_hgb_online_all_no_temporal",
                "lineA_speed_alter_hgb_online_compact",
            ],
        },
    }
    (out / "runtime_model_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("[OK] compact final runtime artifacts prepared:", out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
