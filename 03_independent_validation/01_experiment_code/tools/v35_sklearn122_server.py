#!/usr/bin/env python3
import argparse, json, math, sys, tarfile, tempfile, traceback
from pathlib import Path

import numpy as np
import joblib
import sklearn

REQUIRED_VERSION = "1.2.2"

def score_model(model, x):
    X = np.asarray(x, dtype=np.float64).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        a = np.asarray(p, dtype=np.float64)
        if a.ndim == 2 and a.shape[1] >= 2:
            return float(a[0, 1])
        return float(a.reshape(-1)[0])
    if hasattr(model, "decision_function"):
        z = float(np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1)[0])
        return float(1.0 / (1.0 + math.exp(-z)))
    return float(np.asarray(model.predict(X), dtype=np.float64).reshape(-1)[0])

def load_artifacts(artifact: Path, wanted):
    tmp = tempfile.TemporaryDirectory(prefix="r1c1_v35_sklearn122_")
    root = Path(tmp.name)
    with tarfile.open(artifact, "r:gz") as tf:
        tf.extractall(root)
    out = {}
    for feat_path in sorted(root.rglob("feature_columns.json")):
        det_dir = feat_path.parent
        det_name = det_dir.name
        if wanted and det_name not in wanted:
            continue
        model_path = det_dir / "model.joblib"
        if not model_path.is_file():
            continue
        cols = json.loads(feat_path.read_text(encoding="utf-8"))
        out[det_name] = {
            "model": joblib.load(model_path),
            "feature_count": len(cols),
        }
    return tmp, out

def emit(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--detectors", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    try:
        if sklearn.__version__ != REQUIRED_VERSION:
            raise RuntimeError(f"requires scikit-learn {REQUIRED_VERSION}, got {sklearn.__version__}")
        artifact = Path(args.artifact)
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        wanted = {x.strip() for x in args.detectors.split(',') if x.strip()}
        tmp, dets = load_artifacts(artifact, wanted)
        if not dets:
            raise RuntimeError("no detector artifacts loaded")
        if wanted and set(dets) != wanted:
            raise RuntimeError(f"detector mismatch loaded={sorted(dets)} wanted={sorted(wanted)}")
        ready = {
            "ready": True,
            "sklearn_version": sklearn.__version__,
            "detectors": {k: {"feature_count": int(v["feature_count"]), "model_type": type(v["model"]).__name__} for k,v in sorted(dets.items())}
        }
        emit(ready)
        if args.self_test:
            return
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                cmd = req.get("cmd")
                if cmd == "close":
                    emit({"ok": True, "closed": True})
                    return
                if cmd == "ping":
                    emit({"ok": True, "sklearn_version": sklearn.__version__})
                    continue
                if cmd != "score":
                    raise ValueError(f"unknown cmd={cmd}")
                name = str(req.get("detector", ""))
                if name not in dets:
                    raise KeyError(name)
                vals = req.get("values")
                if not isinstance(vals, list):
                    raise TypeError("values must be list")
                expected = int(dets[name]["feature_count"])
                if len(vals) != expected:
                    raise ValueError(f"feature_count mismatch detector={name} got={len(vals)} expected={expected}")
                s = score_model(dets[name]["model"], vals)
                if not math.isfinite(s):
                    raise ValueError(f"nonfinite score detector={name}: {s}")
                emit({"ok": True, "detector": name, "score": s})
            except Exception as e:
                emit({"ok": False, "error": f"{type(e).__name__}:{e}"})
    except Exception as e:
        emit({"ready": False, "error": f"{type(e).__name__}:{e}", "traceback": traceback.format_exc()})
        raise

if __name__ == "__main__":
    main()
