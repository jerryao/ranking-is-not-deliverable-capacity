"""Customer-level clustering utilities (Instruction 8).

The primary causal unit in v2 is the *customer*, not the episode. We assign
each user a stable customer_cluster_id derived from (city, base_cluster,
industry_type, capacity_quartile) so that downstream bootstrap / cluster-robust
SEs have a clean grouping variable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_customer_clusters(users: pd.DataFrame) -> pd.DataFrame:
    df = users.copy()
    df["capacity_quartile"] = pd.qcut(df["nominal_capacity_kw"], 4, labels=False,
                                       duplicates="drop")
    df["customer_cluster_id"] = (
        df["city"].astype(str) + "|" +
        df["base_cluster"].astype(str) + "|" +
        df["industry_type"].astype(str) + "|" +
        df["capacity_quartile"].astype(str)
    ).astype("category").cat.codes
    return df


def cluster_bootstrap_groups(df: pd.DataFrame, cluster_col: str = "customer_cluster_id") -> np.ndarray:
    return df[cluster_col].to_numpy()


def check_cluster_no_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                              cluster_col: str = "customer_cluster_id") -> bool:
    train_set = set(train[cluster_col].unique().tolist())
    val_overlap = train_set & set(val[cluster_col].unique().tolist())
    test_overlap = train_set & set(test[cluster_col].unique().tolist())
    return len(val_overlap) == 0 and len(test_overlap) == 0