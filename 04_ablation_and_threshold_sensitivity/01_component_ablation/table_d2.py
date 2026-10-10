from __future__ import annotations

import csv
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "component_ablation_sample_level.csv"
TARGETS = ("harmful", "severe_harmful", "speed_alter")
SEEDS = {str(s) for s in range(955916, 955921)}
STAGES = (
    "Base detector",
    "+ Targeted evidence",
    "+ Consistency signal (veto OFF)",
    "Conservative veto (same policy, ON)",
    "Recall-matched selected policy (veto ON)",
)


def binary(row, col):
    value = row[col]
    if value not in ("0", "1"):
        raise ValueError(f"Expected a binary 0/1 in {col}, got {value!r}")
    return int(value)


def prediction(row, target, stage):
    if stage == STAGES[0]:
        return binary(row, "base_alarm")
    if stage == STAGES[1]:
        return binary(row, f"{target}__after_targeted")
    if stage == STAGES[2]:
        return binary(row, f"{target}__after_consistency")
    if stage == STAGES[3]:
        # Same models, features, alarms, thresholds, and samples as stage 3.
        # Only the already-computed conservative veto is enabled.
        off = binary(row, f"{target}__after_consistency")
        veto = binary(row, "conservative_veto_reject")
        return int(off == 1 and veto == 0)
    if stage == STAGES[4]:
        # Separately selected recall-matched policy (NOT a paired-veto control).
        return binary(row, f"{target}__recall_matched_veto_prediction")
    raise ValueError(stage)


def confusion(rows, target, stage):
    tp = fn = fp = tn = 0
    for row in rows:
        y = binary(row, target)
        p = prediction(row, target, stage)
        if y == 1 and p == 1:
            tp += 1
        elif y == 1:
            fn += 1
        elif p == 1:
            fp += 1
        else:
            tn += 1
    recall = 100.0 * tp / (tp + fn)
    far = 100.0 * fp / (fp + tn)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": recall, "far": far}


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Original source not found: {SOURCE}\n"
            "Place this script alongside the released component_ablation_sample_level.csv."
        )
    with SOURCE.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"seed", "sample_id", "base_alarm", "consistency_alarm", "conservative_veto_reject"}
        for target in TARGETS:
            required.update({target, f"{target}__targeted_alarm", f"{target}__after_targeted",
                             f"{target}__after_consistency",
                             f"{target}__recall_matched_veto_prediction"})
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise KeyError(f"Missing columns in original CSV: {sorted(missing)}")
        rows = list(reader)

    if len(rows) != 6000:
        raise ValueError(f"Expected 6000 OOF records; found {len(rows)}")
    if {r["seed"] for r in rows} != SEEDS:
        raise ValueError("Unexpected OOF seeds")
    if len({(r["seed"], r["sample_id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate (seed, sample_id) pairs")

    # Validate the cumulative alarm construction against recorded source flags.
    for r in rows:
        for t in TARGETS:
            exp_targeted = int(binary(r, "base_alarm") or binary(r, f"{t}__targeted_alarm"))
            exp_consistency = int(exp_targeted or binary(r, "consistency_alarm"))
            if binary(r, f"{t}__after_targeted") != exp_targeted:
                raise ValueError(f"Inconsistent targeted alarm: {r['seed']} {r['sample_id']} {t}")
            if binary(r, f"{t}__after_consistency") != exp_consistency:
                raise ValueError(f"Inconsistent consistency alarm: {r['seed']} {r['sample_id']} {t}")

    result = {(s, t): confusion(rows, t, s) for s in STAGES for t in TARGETS}
    print(f"Original source: {SOURCE.name}; records={len(rows)}; seeds={len(SEEDS)}")
    print("\nTable D2: Recall / FAR (%; 3 decimal places)")
    print("| Configuration | Harmful | Severe harmful | Speed alter |")
    print("|---|---|---|---|")
    for s in STAGES:
        cells = [f"{result[(s, t)]['recall']:.2f}% / {result[(s, t)]['far']:.3f}%"
                 for t in TARGETS]
        print(f"| {s} | " + " | ".join(cells) + " |")

    print("\nStrictly paired veto comparison (same policy):")
    print("target, veto, TP, FN, FP, TN, Recall(%), FAR(%)")
    for t in TARGETS:
        off = result[(STAGES[2], t)]
        on = result[(STAGES[3], t)]
        for name, m in (("OFF", off), ("ON", on)):
            print(f"{t}, {name}, {m['tp']}, {m['fn']}, {m['fp']}, {m['tn']}, "
                  f"{m['recall']:.2f}, {m['far']:.3f}")
        print(f"  {t}: FP removed={off['fp'] - on['fp']}; "
              f"TP removed={off['tp'] - on['tp']}")


if __name__ == "__main__":
    main()
