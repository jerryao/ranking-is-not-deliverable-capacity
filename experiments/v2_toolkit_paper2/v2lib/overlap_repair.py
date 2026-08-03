"""Overlap / propensity repair (Instruction 4).

Augments the observational layer with synthetic rows such that, within each
stratum (capacity-quantile × demand-ratio × event-intensity × city-cluster),
the propensity P(V|X) ∈ [0.05, 0.95].

We do this by:
  1. Fitting a quick logistic-propensity estimator from existing data.
  2. Stratifying by the configured bin counts.
  3. For strata with extreme propensity (<min or >max), generating additional
     rows by perturbing capacity/demand/event intensity within the stratum
     and forcing the under-represented arm until the local P(V|X) is in range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .violation_tensor import generate_violation_tensor


def _quantile_bins(s: pd.Series, q: int) -> pd.Series:
    return pd.qcut(s, q=q, labels=False, duplicates="drop")


def _strata(df: pd.DataFrame, cfg: Config) -> pd.Series:
    strata_cfg = cfg.raw["overlap_repair"]["strata"]
    n_cap = strata_cfg["user_capacity_quantiles"]
    n_dem = strata_cfg["task_demand_ratio_bins"]
    n_ei  = strata_cfg["event_intensity_bins"]

    cap_bin = _quantile_bins(df["nominal_capacity_kw"].fillna(df["nominal_capacity_kw"].median()), n_cap)
    dem_ratio = (df["required_capacity_kw"] / df["nominal_capacity_kw"]).clip(0.05, 5.0)
    dem_bin = _quantile_bins(dem_ratio, n_dem)
    ei_bin  = _quantile_bins(df["event_intensity"].fillna(df["event_intensity"].median()), n_ei)
    city = df["city"]
    return (city.astype(str) + "|" + cap_bin.astype(str) + "|" +
            dem_bin.astype(str) + "|" + ei_bin.astype(str))


def repair_overlap(
    df: pd.DataFrame,
    *,
    config: Config,
    treatment_col: str = "V_any_flag",
) -> pd.DataFrame:
    """Return original + augmented rows.

    Adds:
        V_any_flag       — treatment indicator (any non-contract violation)
        treatment_stratum
    """
    cfg = config
    pmin, pmax = cfg.propensity_bounds

    if treatment_col not in df.columns:
        df = df.copy()
        df[treatment_col] = 0  # placeholder; observ rows can be filled later

    df = df.copy()
    df["treatment_stratum"] = _strata(df, cfg)

    augmented = []
    n_orig = len(df)
    max_factor = int(cfg.raw["overlap_repair"]["max_augmentation_factor"])
    rng = np.random.default_rng(cfg.seed + 11)

    for stratum, sub in df.groupby("treatment_stratum"):
        if len(sub) < 5:
            continue
        p = float(sub[treatment_col].mean()) if treatment_col in sub else 0.5
        if pmin <= p <= pmax:
            continue
        # Decide which arm is under-represented
        if p < pmin:
            target_arm = 1
            n_add = int(round(len(sub) * (pmin - p) * 3))
        else:
            target_arm = 0
            n_add = int(round(len(sub) * (p - pmax) * 3))
        n_add = min(n_add, len(sub) * max_factor)
        if n_add <= 0:
            continue
        # Sample base rows and perturb
        idx = rng.choice(sub.index, size=n_add, replace=True)
        new = df.loc[idx].copy()
        # light numeric perturbation to make the rows feel distinct
        for col in ["nominal_capacity_kw", "response_ramp_score",
                    "availability_rate", "dr_history_success"]:
            if col in new.columns:
                jitter = (rng.random(len(new)) - 0.5) * 0.02 * new[col]
                new[col] = (new[col] + jitter).clip(lower=0)
        # Force arm by injecting a violation tensor
        vt = generate_violation_tensor(
            n_rows=len(new),
            base_rates={k: (0.99 if target_arm == 1 else 0.0)
                        for k in cfg.violation_classes},
            severity_dist=cfg.raw["violation_tensor"]["severity_dist"],
            duration_dist=cfg.raw["violation_tensor"]["duration_dist"],
            scope_dist=cfg.raw["violation_tensor"]["scope_dist"],
            seed=cfg.seed + 17,
        )
        v_frame = vt.to_frame()
        for col in v_frame.columns:
            new[col] = v_frame[col].to_numpy()
        new[treatment_col] = target_arm
        new["is_overlap_augmented"] = 1
        augmented.append(new)

    df["is_overlap_augmented"] = 0
    if augmented:
        df = pd.concat([df] + augmented, ignore_index=True)
    print(f"[overlap] rows: {n_orig:,} → {len(df):,} "
          f"(+{len(df)-n_orig:,} augmented)")
    return df