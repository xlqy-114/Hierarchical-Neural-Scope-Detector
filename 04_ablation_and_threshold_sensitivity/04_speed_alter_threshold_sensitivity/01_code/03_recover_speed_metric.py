#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recover the historical continuous speed-alter metric without redefining it.

The historical inference pipeline did NOT parse ``plan_conv.json`` answers with
``ast.literal_eval``.  It used ``robust_retrieve_traj_with_meta`` and, when no
coordinate pairs could be recovered, supplied a six-waypoint all-zero fallback
trajectory to the evaluator/prediction-file path.  This script reproduces that
trajectory-retrieval behavior exactly for the current 5-seed campaign, including
optional stored E1 forced-trajectory objects.

A finite set of candidate step/mask interpretations is then tested against the
preserved 6000 historical binary speed-alter labels.  A speed threshold sweep is
allowed only if exactly one candidate matches all 6000 historical labels at the
historical 0.40 / 0.20 thresholds.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SEEDS = [955916, 955917, 955918, 955919, 955920]


def sid(x: Any) -> str:
    s = str(x)
    return s[:-11] if s.endswith("_trajectory") else s


def canonicalize_traj(obj: Any, *, name: str) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Convert stored planning trajectories to the canonical [6, 2] representation.

    The exported OpenDriveVLA GT reference may preserve a singleton batch
    dimension, e.g. [1, 6, 2].  The online fault and clean prediction paths use
    [6, 2].  This function removes *only* leading singleton dimensions and then
    requires a genuine waypoint-by-xy array.  It never flattens or guesses a
    non-singleton layout.
    """
    arr = np.asarray(obj, dtype=np.float64)
    raw_shape = tuple(int(x) for x in arr.shape)

    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 2:
        raise RuntimeError(
            f"{name}: expected trajectory reducible to [N,2] by removing "
            f"leading singleton dimensions; raw shape={raw_shape}, reduced shape={arr.shape}"
        )
    if arr.shape[1] != 2:
        raise RuntimeError(
            f"{name}: expected final xy dimension=2; raw shape={raw_shape}, "
            f"reduced shape={arr.shape}"
        )
    if arr.shape[0] < 6:
        raise RuntimeError(
            f"{name}: expected at least 6 planning waypoints; raw shape={raw_shape}, "
            f"reduced shape={arr.shape}"
        )

    arr = np.asarray(arr[:6, :2], dtype=np.float64)
    if arr.shape != (6, 2):
        raise RuntimeError(f"{name}: canonicalized shape is {arr.shape}, expected (6,2)")
    return arr, raw_shape


def canonicalize_gt_mask(obj: Any, *, name: str) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Normalize GT planning masks to a six-waypoint boolean vector."""
    arr = np.asarray(obj)
    raw_shape = tuple(int(x) for x in arr.shape)

    # Remove only leading singleton batch dimensions.
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        # OpenDriveVLA masks may carry an xy/component dimension.
        arr = np.any(arr != 0, axis=-1)
    elif arr.ndim == 1:
        arr = arr != 0
    else:
        raise RuntimeError(
            f"{name}: unsupported GT mask shape; raw shape={raw_shape}, "
            f"reduced shape={arr.shape}"
        )

    arr = np.asarray(arr, dtype=bool).reshape(-1)
    if arr.size < 6:
        raise RuntimeError(
            f"{name}: expected at least 6 mask entries; raw shape={raw_shape}, "
            f"reduced length={arr.size}"
        )
    return arr[:6], raw_shape


def normalize_forced_trajectory(obj: Any) -> Optional[List[Tuple[float, float]]]:
    """Exact narrow normalization used by the inference E1 forced-traj path."""
    try:
        arr = np.asarray(obj, dtype=np.float64)
        if arr.size < 2:
            return None
        arr = arr.reshape(-1, 2)
        if arr.shape[0] < 1:
            return None
        arr = arr[:6]
        if arr.shape[0] < 6:
            last = arr[-1:].copy()
            arr = np.concatenate([arr] + [last] * (6 - arr.shape[0]), axis=0)
        out: List[Tuple[float, float]] = []
        for x, y in arr[:6]:
            out.append((float(x), float(y)))
        return out
    except Exception:
        return None


def robust_retrieve_traj(answer: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Trajectory part of the historical robust_retrieve_traj_with_meta path."""
    if isinstance(answer, list):
        raw_text = " ".join(str(x) for x in answer)
    elif isinstance(answer, tuple):
        raw_text = " ".join(str(x) for x in answer)
    else:
        raw_text = str(answer)

    numeric_tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_text)

    text = re.sub(r"[a-zA-Z]", "", raw_text)
    text = re.sub(r"[\u4e00-\u9fff]", "", text)
    text = re.sub(r"(\d+)\.\.+", r"\1.", text)
    text = re.sub(r"[<>]", "", text)
    text = re.sub(
        r"(\d+\.\d+)\s+(\d+\.\d+)(?=\s*[,\)])", r"\1, \2", text
    )
    text = re.sub(r"(\d+\.\d+)-(\d+\.\d+)", r"\1, \2", text)
    text = re.sub(
        r"(\(\s*-?\d+\.\d+,\s*-?\d+\.\d+),\s*-?\d+\.\d+(\s*\))",
        r"\1\2",
        text,
    )
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(\d+\.\d+)\.(\d+)", r"\1\2", text)
    text = re.sub(r"(\d+)\.\.(\d+)", r"\1.\2", text)
    text = re.sub(r"(\d+)\.\-(\d+)", r"\1.\2", text)

    coord_pairs = re.findall(r"[\[\(]([-\deE.+]+),\s*([-\deE.+]+)[\]\)]", text)
    coords: List[Tuple[float, float]] = []
    parse_exception = 0
    try:
        coords = [(float(x), float(y)) for x, y in coord_pairs]
    except Exception:
        coords = []
        parse_exception = 1

    fallback = len(coords) == 0
    if fallback:
        traj = [(0.0, 0.0)] * 6
    else:
        if len(coords) < 6:
            coords.extend([coords[-1]] * (6 - len(coords)))
        elif len(coords) > 6:
            coords = coords[:6]
        traj = coords

    arr = np.asarray(traj, dtype=np.float64).reshape(-1, 2)[:6, :2]
    meta = {
        "raw_coord_pair_count": int(len(coord_pairs)),
        "parse_success": int(not fallback),
        "parse_fallback_zero": int(fallback),
        "parse_exception": int(parse_exception),
        "answer_numeric_token_count": int(len(numeric_tokens)),
        "parsed_all_zero": int(
            float(np.max(np.abs(np.nan_to_num(arr, nan=0.0)))) < 1e-9
        ),
    }
    return arr, meta


def load_inline_parse_audit(campaign: Path, seed_root: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """Load only the saved E-parse metadata needed to verify parser replay."""
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    base = campaign / seed_root
    for seed in SEEDS:
        p = base / f"seed{seed}" / "results" / "v53_inline_detector_records.jsonl"
        if not p.is_file():
            p = base / f"seed{seed}" / "results" / "v54_inline_detector_records.jsonl"
        if not p.is_file():
            raise FileNotFoundError(p)
        n = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                a = o.get("analysis_only") or {}
                ep = a.get("e_parse_features") or {}
                key = (seed, sid(o.get("sample_id") or o.get("raw_sample_id")))
                out[key] = {
                    "parse_fallback_zero": int(ep.get("parse_fallback_zero", 0)),
                    "raw_coord_pair_count": int(ep.get("raw_coord_pair_count", 0)),
                }
                n += 1
        if n != 1200:
            raise RuntimeError(f"seed {seed}: inline rows={n}, expected 1200")
    return out


def load_fault(campaign: Path, seed_root: str):
    out: Dict[Tuple[int, str], np.ndarray] = {}
    audit_rows: List[Dict[str, Any]] = []
    inline = load_inline_parse_audit(campaign, seed_root)
    base = campaign / seed_root

    total_fallback = 0
    total_forced = 0
    replay_mismatch = 0

    for seed in SEEDS:
        p = base / f"seed{seed}" / "plan_conv.json"
        if not p.is_file():
            raise FileNotFoundError(p)
        n = 0
        seed_fallback = 0
        seed_forced = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                sample = sid(o["id"])
                key = (seed, sample)

                forced = None
                if "v48_e1_forced_traj" in o:
                    forced = normalize_forced_trajectory(o.get("v48_e1_forced_traj"))

                answer_traj, meta = robust_retrieve_traj(o.get("answer", ""))
                if forced is not None:
                    traj = np.asarray(forced, dtype=np.float64).reshape(-1, 2)[:6, :2]
                    forced_used = 1
                    seed_forced += 1
                else:
                    traj = answer_traj
                    forced_used = 0

                if meta["parse_fallback_zero"]:
                    seed_fallback += 1

                saved = inline.get(key)
                if saved is None:
                    raise RuntimeError(f"missing inline parse audit for {key}")
                match = (
                    int(meta["parse_fallback_zero"]) == int(saved["parse_fallback_zero"])
                    and int(meta["raw_coord_pair_count"]) == int(saved["raw_coord_pair_count"])
                )
                if not match:
                    replay_mismatch += 1

                out[key] = traj
                audit_rows.append(
                    {
                        "seed": seed,
                        "sample_id": sample,
                        "parse_fallback_zero_replayed": int(meta["parse_fallback_zero"]),
                        "parse_fallback_zero_saved": int(saved["parse_fallback_zero"]),
                        "raw_coord_pair_count_replayed": int(meta["raw_coord_pair_count"]),
                        "raw_coord_pair_count_saved": int(saved["raw_coord_pair_count"]),
                        "forced_traj_used": forced_used,
                        "parser_replay_match": int(match),
                    }
                )
                n += 1

        if n != 1200:
            raise RuntimeError(f"seed {seed}: plan_conv rows={n}, expected 1200")
        print(
            f"[OK] seed {seed}: plan_conv=1200, "
            f"answer_parse_fallback={seed_fallback}, forced_traj={seed_forced}"
        )
        total_fallback += seed_fallback
        total_forced += seed_forced

    if len(out) != 6000:
        raise RuntimeError(f"fault trajectory rows={len(out)}, expected 6000")
    if replay_mismatch:
        raise RuntimeError(
            f"historical robust parser replay disagrees with saved inline parse metadata "
            f"for {replay_mismatch} / 6000 rows"
        )

    print(
        f"[OK] robust parser replay exactly matches saved inline parse metadata: 6000/6000"
    )
    print(
        f"[OK] fault trajectory construction: parse-fallback rows={total_fallback}, "
        f"forced-trajectory rows={total_forced}"
    )
    return out, pd.DataFrame(audit_rows)


def step_norms(t: np.ndarray, mode: str) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != 2:
        raise RuntimeError(f"step_norms received non-canonical trajectory shape={t.shape}")
    t = t[:6, :2]
    if mode == "waypoint_norm":
        return np.linalg.norm(t, axis=1)
    if mode == "segment_no_origin":
        return np.linalg.norm(np.diff(t, axis=0), axis=1)
    if mode == "segment_with_origin":
        z = np.vstack([np.zeros((1, 2), dtype=np.float64), t])
        return np.linalg.norm(np.diff(z, axis=0), axis=1)
    raise ValueError(mode)


def mask_for(gt_mask: Any, n: int, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(n, dtype=bool)
    m = np.asarray(gt_mask, dtype=bool).reshape(-1)[:6]
    if mode == "gt_waypoint_valid":
        return m[:n]
    if mode == "gt_segment_valid_no_origin":
        mm = m[:6]
        return (mm[:-1] & mm[1:])[:n]
    if mode == "gt_segment_valid_with_origin":
        return m[:n]
    raise ValueError(mode)


def mean_valid(x: np.ndarray, m: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=bool)
    return float(np.mean(x[m])) if m.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", required=True)
    ap.add_argument(
        "--seed-root", default="A1_internal_neural_scdout_v54_2_online_singlefault"
    )
    ap.add_argument("--clean-trajs", required=True)
    ap.add_argument("--gt-reference", required=True)
    ap.add_argument("--historical-labels", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    fault, parser_audit = load_fault(Path(args.campaign_dir), args.seed_root)
    parser_audit.to_csv(outdir / "fault_trajectory_parser_replay_audit.csv", index=False)

    clean0 = json.load(open(args.clean_trajs, encoding="utf-8"))
    clean: Dict[str, np.ndarray] = {}
    clean_shape_counts: Dict[str, int] = {}
    for k, v in clean0.items():
        sk = sid(k)
        arr, raw_shape = canonicalize_traj(v, name=f"clean[{sk}]")
        clean[sk] = arr
        key = str(raw_shape)
        clean_shape_counts[key] = clean_shape_counts.get(key, 0) + 1

    gtobj_raw = json.load(open(args.gt_reference, encoding="utf-8"))["samples"]
    gtobj: Dict[str, Dict[str, Any]] = {}
    gt_traj_shape_counts: Dict[str, int] = {}
    gt_mask_shape_counts: Dict[str, int] = {}
    for k, rec in gtobj_raw.items():
        sk = sid(k)
        gtraj, traj_shape = canonicalize_traj(
            rec["gt_traj"], name=f"gt_traj[{sk}]"
        )
        gmask, mask_shape = canonicalize_gt_mask(
            rec["gt_mask"], name=f"gt_mask[{sk}]"
        )
        gtobj[sk] = {"gt_traj": gtraj, "gt_mask": gmask}

        tk = str(traj_shape)
        mk = str(mask_shape)
        gt_traj_shape_counts[tk] = gt_traj_shape_counts.get(tk, 0) + 1
        gt_mask_shape_counts[mk] = gt_mask_shape_counts.get(mk, 0) + 1

    shape_audit = {
        "clean_trajectory_raw_shapes": clean_shape_counts,
        "gt_trajectory_raw_shapes": gt_traj_shape_counts,
        "gt_mask_raw_shapes": gt_mask_shape_counts,
        "canonical_trajectory_shape": [6, 2],
        "canonical_mask_length": 6,
        "normalization_rule": (
            "remove only leading singleton batch dimensions; require [N,2]; "
            "take first six planning waypoints"
        ),
    }
    (outdir / "trajectory_shape_normalization_audit.json").write_text(
        json.dumps(shape_audit, indent=2), encoding="utf-8"
    )

    print("[OK] trajectory representation normalized")
    print("     clean raw shapes:", clean_shape_counts)
    print("     GT trajectory raw shapes:", gt_traj_shape_counts)
    print("     GT mask raw shapes:", gt_mask_shape_counts)
    print("[OK] all clean/GT trajectories canonicalized to [6,2]")

    hist = pd.read_csv(args.historical_labels)
    hist["sample_id"] = hist["sample_id"].map(sid)
    hist = hist.sort_values(["seed", "sample_id"]).reset_index(drop=True)
    if len(hist) != 6000:
        raise RuntimeError(f"historical label rows={len(hist)}, expected 6000")

    expected = hist["v31_speed_alter_harmful_compatible"].astype(int).to_numpy()
    old = hist["harmful"].astype(int).to_numpy()

    rows = []
    metrics_rows = []
    modes = ["waypoint_norm", "segment_no_origin", "segment_with_origin"]
    gt_forms = ["signed_error_degradation", "absolute_error_degradation_delta"]

    for mode in modes:
        mask_modes = ["none"]
        if mode == "waypoint_norm":
            mask_modes += ["gt_waypoint_valid"]
        elif mode == "segment_no_origin":
            mask_modes += ["gt_segment_valid_no_origin"]
        else:
            mask_modes += ["gt_segment_valid_with_origin"]

        for mmode in mask_modes:
            cvals = []
            gvals = []
            valid = True

            for r in hist.itertuples(index=False):
                key = (int(r.seed), sid(r.sample_id))
                s = sid(r.sample_id)
                if key not in fault or s not in clean or s not in gtobj:
                    valid = False
                    break

                ftraj = np.asarray(fault[key], dtype=np.float64)
                ctraj = np.asarray(clean[s], dtype=np.float64)
                gtraj = np.asarray(gtobj[s]["gt_traj"], dtype=np.float64)
                for label_name, arr in (
                    ("fault", ftraj), ("clean", ctraj), ("gt", gtraj)
                ):
                    if arr.shape != (6, 2):
                        raise RuntimeError(
                            f"{key} {label_name} trajectory shape={arr.shape}; expected (6,2)"
                        )

                f = step_norms(ftraj, mode)
                c = step_norms(ctraj, mode)
                g = step_norms(gtraj, mode)
                n = min(len(f), len(c), len(g))
                if n <= 0:
                    raise RuntimeError(
                        f"{key} produced zero-length step sequence for mode={mode}: "
                        f"fault={f.shape}, clean={c.shape}, gt={g.shape}"
                    )
                f, c, g = f[:n], c[:n], g[:n]

                try:
                    m = mask_for(gtobj[s]["gt_mask"], n, mmode)
                except Exception:
                    valid = False
                    break
                if len(m) != n:
                    m = np.ones(n, dtype=bool)

                clean_delta = mean_valid(np.abs(f - c), m)
                fault_gt = mean_valid(np.abs(f - g), m)
                clean_gt = mean_valid(np.abs(c - g), m)
                cvals.append(clean_delta)
                gvals.append((fault_gt, clean_gt))

            if not valid:
                continue

            cvals = np.asarray(cvals)
            fg = np.asarray([x[0] for x in gvals])
            cg = np.asarray([x[1] for x in gvals])

            for form in gt_forms:
                if form == "signed_error_degradation":
                    gtdeg = fg - cg
                else:
                    gtdeg = np.abs(fg - cg)

                speed = (cvals >= 0.40) & (gtdeg >= 0.20)
                pred = (old == 1) | speed
                mismatch_mask = pred.astype(int) != expected
                mismatch = int(mismatch_mask.sum())

                rows.append(
                    {
                        "step_mode": mode,
                        "mask_mode": mmode,
                        "gt_degradation_form": form,
                        "mismatch_rows": mismatch,
                        "predicted_speed_alter_total": int(pred.sum()),
                        "speed_condition_total": int(speed.sum()),
                        "historical_total": int(expected.sum()),
                        "exact_match": int(mismatch == 0),
                    }
                )

                tag = f"{mode}__{mmode}__{form}"
                for i, r in enumerate(hist.itertuples(index=False)):
                    metrics_rows.append(
                        {
                            "candidate": tag,
                            "seed": int(r.seed),
                            "sample_id": sid(r.sample_id),
                            "fault_vs_clean_mean_step_norm_abs_delta": float(cvals[i]),
                            "degradation_vs_clean_gt_mean_step_norm_abs_delta": float(gtdeg[i]),
                            "speed_condition": int(speed[i]),
                            "speed_alter_pred": int(pred[i]),
                            "speed_alter_historical": int(expected[i]),
                        }
                    )

    res = pd.DataFrame(rows).sort_values(
        ["mismatch_rows", "step_mode", "mask_mode", "gt_degradation_form"]
    )
    res.to_csv(outdir / "speed_metric_candidate_recovery_summary.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(
        outdir / "speed_metric_candidate_values.csv", index=False
    )

    exact = res[res.exact_match.eq(1)]
    print(res.to_string(index=False))
    print(f"\nExact candidates: {len(exact)}")
    if len(exact) == 1:
        print(
            "[OK] unique exact candidate recovered; the speed threshold sweep may proceed."
        )
    elif len(exact) == 0:
        print(
            "[STOP] no exact candidate. Do not define a new speed metric; recover the "
            "historical behavior-audit source/continuous metric."
        )
    else:
        print(
            "[STOP] multiple exact candidates. Binary labels do not uniquely identify "
            "the historical metric; recover source or additional audit data."
        )


if __name__ == "__main__":
    main()
