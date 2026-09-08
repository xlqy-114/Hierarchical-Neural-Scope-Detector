#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--official-full-sample-ids', type=Path, required=True)
    ap.add_argument('--count', type=int, required=True)
    ap.add_argument('--output-json', type=Path, required=True)
    args=ap.parse_args()
    ids=json.loads(args.official_full_sample_ids.read_text(encoding='utf-8'))
    if not isinstance(ids, list):
        raise SystemExit('official ids must be a list')
    n=int(args.count)
    if n <= 0 or n > len(ids):
        raise SystemExit(f'count must be in 1..{len(ids)}, got {n}')
    out=[str(x) for x in ids[:n]]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'version':'v73_16_run_sample_ids','source':str(args.official_full_sample_ids),'count':len(out),'output':str(args.output_json),'first_id':out[0] if out else '', 'last_id':out[-1] if out else ''}, indent=2, ensure_ascii=False))

if __name__=='__main__':
    main()
