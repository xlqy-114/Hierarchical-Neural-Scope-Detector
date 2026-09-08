#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize a self-contained clean reference from a v73.16 clean pass.

This tool imports the package-contained inference module and uses its existing
trajectory parser/evaluation helpers. It does not read any historical experiment
output. Inputs are the current clean pass result directory and the current repo's
eval_share/GT files.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
from typing import Any, Dict, List


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out=[]
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out


def import_inference_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"missing inference module: {path}")
    spec = importlib.util.spec_from_file_location('v7314_inference_module', str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import inference module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules['v7314_inference_module'] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--result-dir', type=Path, required=True)
    ap.add_argument('--inference-module-path', type=Path, required=True)
    ap.add_argument('--expected-rows', type=int, required=True)
    ap.add_argument('--summary-json', type=Path, required=True)
    ap.add_argument('--strict', action='store_true')
    args=ap.parse_args()

    result_dir=args.result_dir
    plan=result_dir/'plan_conv.json'
    if not plan.is_file():
        raise FileNotFoundError(plan)
    inf=import_inference_module(args.inference_module_path)

    parse_info=inf.build_pred_files_from_plan_conv(result_dir)
    log_path=result_dir/'eval_results.log'
    per_sample_path, planning=inf.compute_subset_metrics(result_dir, log_path)
    planning['parser']=parse_info
    planning_path=result_dir/'planning_metrics.json'
    write_json(planning_path, planning)

    rows=read_jsonl(per_sample_path)
    output_thr=inf.build_output_sanity_thresholds(rows)
    out_sanity_path=result_dir/'output_sanity_thresholds.json'
    write_json(out_sanity_path, output_thr)

    summary={
        'version':'v73_16_clean_reference_materialization',
        'result_dir':str(result_dir),
        'plan_conv':str(plan),
        'pred_trajs_dict':str(result_dir/'pred_trajs_dict.json'),
        'pred_trajs_multi_modal_dict':str(result_dir/'pred_trajs_multi_modal_dict.json'),
        'trajectory_parse_metadata':str(result_dir/'trajectory_parse_metadata.jsonl'),
        'per_sample_planning_metrics':str(per_sample_path),
        'planning_metrics_json':str(planning_path),
        'output_sanity_thresholds_json':str(out_sanity_path),
        'eval_log':str(log_path),
        'expected_rows':int(args.expected_rows),
        'per_sample_rows':int(len(rows)),
        'parser_count':int(parse_info.get('count', 0)),
        'invalid_parse_count':int(parse_info.get('invalid_parse_count', 0)),
        'planning_available':bool(planning.get('available', True)),
        'pass': bool(len(rows)==int(args.expected_rows) and int(parse_info.get('count', 0))==int(args.expected_rows)),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.strict and not summary['pass']:
        raise SystemExit(6)

if __name__ == '__main__':
    main()
