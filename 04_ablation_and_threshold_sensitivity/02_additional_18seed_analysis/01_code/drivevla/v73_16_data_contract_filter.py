#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime helper for the v73.16 standalone data contract.

This module is intentionally detector-free. It only filters feature dictionaries to the exact
v73.16 runtime/source feature contract and can compute the selected v73gen features for a batch DataFrame.
It is meant to be imported at the point where the existing OpenDriveVLA inference already has a per-sample
feature row. It must not receive or emit detector score/alarm/threshold/hgbmax columns.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, List
import json, re
import numpy as np
import pandas as pd

KEY_COLS = ["seed", "sample_id"]

class V7314StandaloneDataContractFilter:
    def __init__(self, resources_dir: str | Path):
        self.resources_dir = Path(resources_dir)
        self.collect = pd.read_csv(self.resources_dir / "v73_9_hgb_runtime_collection_columns.csv")
        self.contract = pd.read_csv(self.resources_dir / "v73_9_hgb_online_feature_contract.csv")
        self.gen_source = pd.read_csv(self.resources_dir / "v73_9_hgb_generated_consistency_source_columns.csv")
        self.gen_contract = pd.read_csv(self.resources_dir / "v73_9_hgb_generated_feature_contract.csv")
        self.runtime_columns = self.collect["feature"].drop_duplicates().tolist()
        self.model_features = self.contract["feature"].drop_duplicates().tolist()
        self.generated_features = self.gen_contract["feature"].drop_duplicates().tolist()
    def filter_feature_row(self, row: Dict[str, Any], keep_audit: bool = True) -> Dict[str, Any]:
        out = {k: row.get(k) for k in KEY_COLS if k in row}
        if keep_audit:
            for k in ["exact_layer", "row_source", "sample_token", "fault_scope", "fault_type", "target_group", "bit_idx", "applied", "effective"]:
                if k in row: out[k] = row.get(k)
        for c in self.runtime_columns:
            out[c] = row.get(c, np.nan)
        return out
    @staticmethod
    def _families(feature: str) -> List[str]:
        out=[]
        m=re.match(r"^v73gen__(.+?)__row_", str(feature))
        if m: out.append(m.group(1))
        m=re.match(r"^v73gen__consistency__(.+?)_(?:minus|over)_(.+?)__", str(feature))
        if m: out.extend([m.group(1), m.group(2)])
        return sorted(set(out))
    def compute_generated_for_frame(self, source_df: pd.DataFrame) -> pd.DataFrame:
        needed_families=sorted({fam for f in self.generated_features for fam in self._families(f)})
        stats={}; eps=1e-6
        for fam in needed_families:
            cols=self.gen_source.loc[self.gen_source.generated_family.eq(fam), 'source_feature'].drop_duplicates().tolist()
            missing=[c for c in cols if c not in source_df.columns]
            if missing:
                raise RuntimeError(f"v73.16 generated family {fam} missing source columns: {missing[:10]}")
            arr=source_df[cols].to_numpy(dtype=np.float64, copy=False)
            arr=np.where(np.isfinite(arr), arr, np.nan)
            ab=np.abs(arr)
            with np.errstate(all='ignore'):
                stats[fam]={
                    'row_mean_abs':np.nanmean(ab, axis=1),
                    'row_p95_abs':np.nanpercentile(ab, 95, axis=1),
                    'row_std':np.nanstd(arr, axis=1),
                    'row_nan_rate':np.mean(~np.isfinite(arr), axis=1),
                }
        gen={}
        for f in self.generated_features:
            m=re.match(r"^v73gen__(.+?)__(row_[A-Za-z0-9_]+)$", f)
            if m:
                gen[f]=stats[m.group(1)][m.group(2)]; continue
            m=re.match(r"^v73gen__consistency__(.+?)_minus_(.+?)__(mean_abs|p95_abs)$", f)
            if m:
                gen[f]=stats[m.group(1)]['row_'+m.group(3)]-stats[m.group(2)]['row_'+m.group(3)]; continue
            m=re.match(r"^v73gen__consistency__(.+?)_over_(.+?)__(mean_abs|p95_abs)_logratio$", f)
            if m:
                gen[f]=np.log((stats[m.group(1)]['row_'+m.group(3)]+eps)/(stats[m.group(2)]['row_'+m.group(3)]+eps)); continue
            raise RuntimeError(f"Unsupported v73gen formula: {f}")
        return pd.DataFrame(gen, index=source_df.index).replace([np.inf, -np.inf], np.nan)
    def build_model_ready_frame(self, source_df: pd.DataFrame) -> pd.DataFrame:
        G=self.compute_generated_for_frame(source_df)
        non_gen=[c for c in self.model_features if not c.startswith('v73gen__')]
        missing=[c for c in non_gen if c not in source_df.columns]
        if missing:
            raise RuntimeError(f"Missing v73.16 non-generated model-ready features: {missing[:20]}")
        keys=[c for c in KEY_COLS if c in source_df.columns]
        return pd.concat([source_df[keys], source_df[non_gen], G[[c for c in self.model_features if c.startswith('v73gen__')]]], axis=1)


# Backward-compatible alias for older export helpers; new code should use V7314StandaloneDataContractFilter.
V739HGBFeatureFilter = V7314StandaloneDataContractFilter
