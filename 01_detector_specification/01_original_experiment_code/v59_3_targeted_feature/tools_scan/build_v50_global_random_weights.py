#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build target-count-proportional scope weights for v50 global-random S/C/D inference.

The inference script samples scope first and target within scope second. To emulate
one global random draw over all eligible targets, this script sets scope weights to
eligible target counts and all within-scope target weights to 1.

S0 is represented by can_bus groups in the existing inference code. Its target count
is therefore configurable; the default is 7, matching the v35/v38 can_bus groups:
translation_xyz, rotation_quat, small_signed_state, lp_9, lp_13, near_zero_slots,
patch_angle_pair.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_specs(text: str) -> List[Tuple[str, str]]:
    out=[]
    for part in str(text or '').strip().split(','):
        part=part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f"Invalid target spec without '=': {part[:120]}")
        name, suffix = part.split('=', 1)
        name=name.strip(); suffix=suffix.strip()
        if name:
            out.append((name, suffix))
    return out


def weights_from_names(names: List[str]) -> str:
    return ','.join(f'{n}:1' for n in names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--s-target-specs-file', required=True)
    ap.add_argument('--d-target-specs-file', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--s0-target-count', type=int, default=7)
    ap.add_argument('--c-target-specs', default='c_mm_projector_scene=mm_projector_scene,c_mm_projector_scene_layer0=mm_projector_scene.0,c_mm_projector_scene_layer1=mm_projector_scene.1,c_mm_projector_scene_layer2=mm_projector_scene.2')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    s_specs_text = Path(args.s_target_specs_file).read_text(encoding='utf-8').strip()
    d_specs_text = Path(args.d_target_specs_file).read_text(encoding='utf-8').strip()
    s_items = parse_specs(s_specs_text)
    d_items = parse_specs(d_specs_text)
    c_items = parse_specs(args.c_target_specs)

    s_names = [x[0] for x in s_items]
    d_names = [x[0] for x in d_items]
    c_names = [x[0] for x in c_items]

    if args.s0_target_count <= 0:
        raise ValueError('--s0-target-count must be positive')
    if not s_names:
        raise ValueError('No S targets found')
    if not c_names:
        raise ValueError('No C targets found')
    if not d_names:
        raise ValueError('No D targets found')

    counts = {
        'S0_only': int(args.s0_target_count),
        'S1_only': int(len(s_names)),  # historical argument name; logically S1-S5 full S leaf activations
        'C_only': int(len(c_names)),
        'D_only': int(len(d_names)),
    }
    scope_weights = ','.join(f'{k}:{v}' for k, v in counts.items())

    (out / 'v50_global_random_scope_weights.txt').write_text(scope_weights, encoding='utf-8')
    (out / 'v50_s_full_target_weights_global_random.txt').write_text(weights_from_names(s_names), encoding='utf-8')
    (out / 'v50_c_target_weights_global_random.txt').write_text(weights_from_names(c_names), encoding='utf-8')
    (out / 'v50_d_target_weights_global_random.txt').write_text(weights_from_names(d_names), encoding='utf-8')

    summary = {
        'version': 'v50_global_random_scope_weights_v1',
        'policy': 'scope weights are proportional to eligible target counts; within-scope target weights are all 1',
        'important_note': 'This emulates a single global random target draw even though the legacy inference CLI samples scope first.',
        's0_target_count_assumption': counts['S0_only'],
        's0_target_groups': ['translation_xyz', 'rotation_quat', 'small_signed_state', 'lp_9', 'lp_13', 'near_zero_slots', 'patch_angle_pair'],
        'target_counts': counts,
        'scope_weights': scope_weights,
        's_target_count': len(s_names),
        'c_target_count': len(c_names),
        'd_target_count': len(d_names),
        'outputs': {
            'scope_weights': str(out / 'v50_global_random_scope_weights.txt'),
            's_weights': str(out / 'v50_s_full_target_weights_global_random.txt'),
            'c_weights': str(out / 'v50_c_target_weights_global_random.txt'),
            'd_weights': str(out / 'v50_d_target_weights_global_random.txt'),
        }
    }
    (out / 'v50_global_random_scope_weights_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
