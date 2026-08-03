"""Intervention pair generation (Instruction 1).

For each base row from the source task_assessments, we create:

    pair_id        = unique per (city, user, task, event, seed, intervention_kind)
    source         = "do(V=0)" | "do(V=1)"
    Y_delivery_0   = outcome under V=0
    Y_delivery_1   = outcome under V=1
    Y_comfort_0/1
    Y_rebound_0/1
    true_tau_*     = Y_*_1 - Y_*_0  (oracle ground-truth effect)

The same epsilon noise is used for both arms (shared_noise_stable).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import Config
from .oracle_simulator import OracleSimulator


def _stable_pair_id(row: pd.Series, salt: int) -> str:
    """Deterministic 16-char id derived from (city, user, task, event, salt)."""
    h = hashlib.sha1(
        f"{row['city']}|{int(row['user_id'])}|{row['task_id']}|{row['event_id']}|{salt}"
        .encode("utf-8")
    ).hexdigest()
    return h[:16]


def generate_intervention_pairs(
    base: pd.DataFrame,
    *,
    config: Config,
    seed_salt: int | None = None,
) -> pd.DataFrame:
    """Build the long-form intervention-pairs table.

    Each base row contributes TWO long-form rows (V=0, V=1) sharing a pair_id.
    Schema includes the violation tensor (all 20 columns) and the outcomes.
    """
    if seed_salt is None:
        seed_salt = config.raw["seed_policy"]["intervention_pair_offset"]

    base = base.reset_index(drop=True).copy()
    base["pair_id"] = [
        _stable_pair_id(row, seed_salt) for _, row in base.iterrows()
    ]

    oracle = OracleSimulator(config)

    rows_long = []
    rows_tau = []
    # Group by (city, user_id) for seed reuse (matches meta.json seed formula)
    for (city, user_id), grp in base.groupby(["city", "user_id"], sort=False):
        seed = config.seed_for(city, int(user_id), salt=seed_salt)
        df0, df1, df_tau = oracle.simulate_pair(grp, seed=seed)
        df0["source"] = "do(V=0)"
        df1["source"] = "do(V=1)"
        long_part = pd.concat([df0, df1], ignore_index=True)
        rows_long.append(long_part)
        rows_tau.append(df_tau)

    long_df = pd.concat(rows_long, ignore_index=True)
    tau_df = pd.concat(rows_tau, ignore_index=True)
    return long_df, tau_df