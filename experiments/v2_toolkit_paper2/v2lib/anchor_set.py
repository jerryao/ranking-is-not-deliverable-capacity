"""Cross-city anchor set (Instruction 10).

Build a small synthetic grid of (X_cluster, task, demand, event) tuples, then
*for each tuple* instantiate the same setup under each city's dynamics.

This is the experiment that disentangles *distribution shift* (P(X) differs
across cities) from *mechanism shift* (P(Y|X) differs across cities).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .violation_tensor import generate_violation_tensor
from .dose_response import compute_outcomes, sample_shared_noise
from .safety_cost import add_safety_columns


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


def generate_anchor_set(
    base: pd.DataFrame,
    *,
    config: Config,
) -> pd.DataFrame:
    cfg = config
    n = cfg.n_anchors
    rng = _rng(cfg.seed + cfg.raw["seed_policy"]["anchor_offset"])

    cities = sorted(base["city"].unique().tolist())
    user_clusters = sorted(base["user_id"].unique().tolist())[:max(1, len(set(base['user_id'])))]
    # Stratify capacity from observed marginal
    capacities = np.quantile(base["nominal_capacity_kw"].to_numpy(),
                             np.linspace(0.1, 0.9, 8))
    task_ids = sorted(base["task_id"].unique().tolist())
    event_intensities = np.quantile(base["event_intensity"].to_numpy(),
                                    np.linspace(0.1, 0.9, 5))

    out = []
    idx = 0
    for cluster in user_clusters[:min(60, len(user_clusters))]:
        for ti, task in enumerate(task_ids):
            for cap in capacities:
                for ei in event_intensities:
                    if len(out) >= n * len(cities):
                        break
                    base_demand = cap * rng.uniform(0.4, 1.2)
                    base_event_intensity = float(ei)
                    for city in cities:
                        # Match city-specific response rate / response delay
                        city_sub = base[base["city"] == city]
                        rr = float(np.clip(
                            city_sub["response_rate"].mean() +
                            0.1 * (rng.random() - 0.5), 0.3, 0.95))
                        rd = int(rng.integers(3, 13))
                        out.append({
                            "anchor_id": f"A{idx:05d}",
                            "shared_X_cluster": int(cluster),
                            "task_id": task,
                            "city": city,
                            "nominal_capacity_kw": float(cap),
                            "required_capacity_kw": float(base_demand),
                            "event_intensity_band": float(round(base_event_intensity, 4)),
                            "response_rate": rr,
                            "response_delay_min": rd,
                        })
                        idx += 1
                        if len([o for o in out if o["anchor_id"] == f"A{(idx-1):05d}"]) == len(cities):
                            pass
                if len(out) >= n * len(cities):
                    break
            if len(out) >= n * len(cities):
                break
        if len(out) >= n * len(cities):
            break

    df = pd.DataFrame(out)
    # Cap to n_anchors unique anchor_ids (each replicated across cities)
    keep = df["anchor_id"].drop_duplicates().head(n).tolist()
    df = df[df["anchor_id"].isin(keep)].reset_index(drop=True)

    # Run a small dose-response simulation
    n_rows = len(df)
    vt = generate_violation_tensor(
        n_rows=n_rows,
        base_rates={k: float(cfg.raw["violation_tensor"]["base_rates"][k])
                    for k in cfg.violation_classes},
        severity_dist=cfg.raw["violation_tensor"]["severity_dist"],
        duration_dist=cfg.raw["violation_tensor"]["duration_dist"],
        scope_dist=cfg.raw["violation_tensor"]["scope_dist"],
        seed=cfg.seed + cfg.raw["seed_policy"]["anchor_offset"] + 1,
    )
    v_frame = vt.to_frame()
    for col in v_frame.columns:
        df[col] = v_frame[col].to_numpy()
    df["V_any_flag"] = (df[[c for c in v_frame.columns if c.endswith("_flag")]].sum(axis=1) > 0).astype(int)

    out_obj = compute_outcomes(
        base_delivery_kw=(df["nominal_capacity_kw"] * df["response_rate"]).to_numpy(),
        response_rate=df["response_rate"].to_numpy(),
        response_delay_min=df["response_delay_min"].to_numpy(),
        event_intensity=df["event_intensity_band"].to_numpy(),
        V=vt, weights=cfg.dose_weights,
        noise=sample_shared_noise(rng, n_rows),
    )
    for ch in ["delivery","comfort_loss","rebound_risk","contract_penalty","instability"]:
        df[ch] = getattr(out_obj, ch)
    df["pred_reliable_deliverable_capacity_kw"] = df["nominal_capacity_kw"] * df["response_rate"]
    df = add_safety_columns(df, cfg)
    return df