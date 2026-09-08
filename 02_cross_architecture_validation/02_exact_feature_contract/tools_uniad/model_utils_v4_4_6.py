#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pickle-stable preprocessing classes for UniAD2 v4.4.6 model bundles."""
from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SafeFiniteClipper(BaseEstimator, TransformerMixin):
    """Replace nonfinite values with NaN and clamp finite values to +/- abs_cap."""

    def __init__(self, abs_cap: float = 1_000_000.0):
        self.abs_cap = float(abs_cap)
        self.n_features_in_: Optional[int] = None

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise RuntimeError(f"SafeFiniteClipper expects 2D input, got {arr.shape}")
        self.n_features_in_ = int(arr.shape[1])
        return self

    def transform(self, X):
        if self.n_features_in_ is None:
            raise RuntimeError("SafeFiniteClipper is not fitted")
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2 or int(arr.shape[1]) != self.n_features_in_:
            raise RuntimeError(
                f"SafeFiniteClipper width mismatch: expected {self.n_features_in_}, got {arr.shape}"
            )
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return np.clip(arr, -self.abs_cap, self.abs_cap)


class SafeMedianImputer(BaseEstimator, TransformerMixin):
    """Median imputer that preserves all columns and maps all-NaN columns to zero."""

    def __init__(self, fill_value: float = 0.0, abs_cap: float = 1_000_000.0):
        self.fill_value = float(fill_value)
        self.abs_cap = float(abs_cap)
        self.statistics_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        arr = np.clip(arr, -self.abs_cap, self.abs_cap)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(arr, axis=0)
        med[~np.isfinite(med)] = self.fill_value
        self.statistics_ = np.clip(med, -self.abs_cap, self.abs_cap).astype(np.float64)
        return self

    def transform(self, X):
        if self.statistics_ is None:
            raise RuntimeError("SafeMedianImputer is not fitted")
        arr = np.asarray(X, dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        arr = np.clip(arr, -self.abs_cap, self.abs_cap)
        out = np.where(np.isfinite(arr), arr, self.statistics_)
        return np.clip(out, -self.abs_cap, self.abs_cap)
