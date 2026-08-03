"""Safety-cost and reward mapping (Instruction 6).

    safety_cost = α1 * comfort_loss + α2 * rebound_risk
                + α3 * contract_penalty + α4 * instability

    R = delivery_kw_normalized − λ * safety_cost
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _norm_delivery(s: pd.Series) -> np.ndarray:
    # Normalize to [0,1] by dividing by the 99th percentile observed
    p99 = float(np.nanpercentile(s.to_numpy(), 99))
    if p99 <= 0:
        p99 = 1.0
    return (s.to_numpy(dtype=np.float64) / p99).clip(0.0, 1.0)


def compute_safety_cost(df: pd.DataFrame, config: Config) -> np.ndarray:
    a = config.alpha_weights
    return (float(a["comfort_loss"])     * df["comfort_loss"].to_numpy()
          + float(a["rebound_risk"])     * df["rebound_risk"].to_numpy()
          + float(a["contract_penalty"]) * df["contract_penalty"].to_numpy()
          + float(a["instability"])      * df["instability"].to_numpy())


def compute_reward(df: pd.DataFrame, config: Config) -> np.ndarray:
    delivery_norm = _norm_delivery(df["delivery"])
    safety = compute_safety_cost(df, config)
    return delivery_norm - config.lambda_safety * safety


def add_safety_columns(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = df.copy()
    df["safety_cost"] = compute_safety_cost(df, config)
    df["reward"]      = compute_reward(df, config)
    return df