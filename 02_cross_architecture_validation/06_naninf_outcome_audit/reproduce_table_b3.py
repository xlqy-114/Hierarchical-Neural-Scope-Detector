#!/usr/bin/env python3
"""Recompute Table B3 from the released, minimal sample-level audit flags."""
import csv
from pathlib import Path

path = Path(__file__).with_name('v351_sample_flags.csv')
with path.open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 6019, len(rows)
selected = [r for r in rows if r['effective_fault'] == '1' and r['observed_hook_nonfinite'] == '1']
assert len(selected) == 900, len(selected)
nonfinite = sum(r['final_traj_nonfinite'] == '1' for r in selected)
finite_deviation = sum(r['final_traj_nonfinite'] == '0' and r['finite_max_deviation_gt_1m'] == '1' for r in selected)
normal = len(selected) - nonfinite - finite_deviation
print('Table B3: outcome | samples | percentage')
for name, count in [
    ('Non-finite planning output', nonfinite),
    ('Finite output, max deviation > 1.0 m', finite_deviation),
    ('Total output abnormalities', nonfinite + finite_deviation),
    ('No output abnormality under the criterion', normal),
    ('Total', len(selected)),
]:
    print(f'{name} | {count} | {100 * count / len(selected):.2f}%')
assert (nonfinite, finite_deviation, normal) == (483, 10, 407)
