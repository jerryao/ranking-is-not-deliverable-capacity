"""Label collapse fix (Instruction 9).

Replaces the single-class pv_label / work_rest_label with quantile bins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _quantile_label(s: pd.Series, n_bins: int, names: list[str]) -> pd.Series:
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(s.to_numpy(dtype=float), q)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    out = pd.cut(s, bins=edges, labels=names, include_lowest=True)
    return out.astype(str)


def fix_labels(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    cfg = config
    df = df.copy()
    n = cfg.label_n_bins
    names = cfg.label_names[:n]
    if len(names) < n:
        names = names + [f"q{i}" for i in range(len(names), n)]

    if "pv_ratio" in df.columns:
        df["pv_ratio_v2"] = df["pv_ratio"].astype(float)
        df["pv_label_v2"] = _quantile_label(df["pv_ratio_v2"], n, names)
    if "work_rest_ratio" in df.columns:
        df["work_rest_ratio_v2"] = df["work_rest_ratio"].astype(float)
        df["work_rest_label_v2"] = _quantile_label(df["work_rest_ratio_v2"], n, names)
    return df