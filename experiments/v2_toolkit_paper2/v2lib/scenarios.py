"""Decision-boundary scenario generator (Instruction 7).

Produces three scenario classes:
    easy      — demand/capacity ≪ 1 (no constraint hits)
    boundary  — demand/capacity ∈ [0.85, 1.15]   (50% of generated)
    stress    — demand/capacity ≫ 1 + capacity clipped to 60% (20%)

Each scenario is a fully-realized row of the same schema as task_assessments,
so downstream RL pipelines can consume it directly.
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


def _sample_template_rows(template: pd.DataFrame, n: int, *,
                          rng: np.random.Generator,
                          keep_keys: list[str]) -> pd.DataFrame:
    if len(template) == 0:
        raise ValueError("template empty")
    idx = rng.integers(0, len(template), size=n)
    return template.iloc[idx].reset_index(drop=True)[keep_keys].copy()


def generate_decision_scenarios(
    base: pd.DataFrame,
    *,
    config: Config,
    n_total: int = 1500,
) -> pd.DataFrame:
    """Generate Easy / Boundary / Stress scenarios with ratios 30/50/20."""
    cfg = config
    ratios = cfg.scenario_ratios
    n_easy = int(round(n_total * ratios["easy"]))
    n_bnd  = int(round(n_total * ratios["boundary"]))
    n_str  = n_total - n_easy - n_bnd  # remaining goes to stress

    rng = _rng(cfg.seed + cfg.raw["seed_policy"]["scenario_offset"])

    keep = ["city","user_id","industry_type","user_type","nominal_capacity_kw",
            "event_type","response_rate","response_ramp_score","availability_rate",
            "dr_history_success","process_constraint_score","comfort_constraint_score",
            "rebound_tendency","delivery_uncertainty_score","task_id","event_id",
            "event_intensity","duration_h","start_hour","required_capacity_kw",
            "response_rate_req","reliability_req"]

    rows = []
    # --- Easy ---
    e = _sample_template_rows(base, n_easy, rng=rng, keep_keys=keep)
    # demand << capacity
    e["required_capacity_kw"] = (e["nominal_capacity_kw"] *
                                  rng.uniform(0.30, 0.60, size=len(e)))
    e["scenario_class"] = "easy"
    rows.append(e)

    # --- Boundary ---
    b = _sample_template_rows(base, n_bnd, rng=rng, keep_keys=keep)
    lo, hi = cfg.boundary_band
    b["required_capacity_kw"] = (b["nominal_capacity_kw"] *
                                  rng.uniform(lo, hi, size=len(b)))
    b["scenario_class"] = "boundary"
    rows.append(b)

    # --- Stress ---
    s = _sample_template_rows(base, n_str, rng=rng, keep_keys=keep)
    s["nominal_capacity_kw"] = s["nominal_capacity_kw"] * cfg.stress_clip
    s["required_capacity_kw"] = (s["nominal_capacity_kw"] *
                                  rng.uniform(1.10, 1.50, size=len(s)))
    s["scenario_class"] = "stress"
    rows.append(s)

    df = pd.concat(rows, ignore_index=True)
    # simulate responses using the dose-response model on V=0 path (best-case)
    n = len(df)
    vt = generate_violation_tensor(
        n_rows=n,
        base_rates={k: float(cfg.raw["violation_tensor"]["base_rates"][k])
                    for k in cfg.violation_classes},
        severity_dist=cfg.raw["violation_tensor"]["severity_dist"],
        duration_dist=cfg.raw["violation_tensor"]["duration_dist"],
        scope_dist=cfg.raw["violation_tensor"]["scope_dist"],
        seed=cfg.seed + cfg.raw["seed_policy"]["scenario_offset"] + 1,
    )
    v_frame = vt.to_frame()
    for col in v_frame.columns:
        df[col] = v_frame[col].to_numpy()
    df["response_delay_min"] = rng.integers(3, 14, size=n)
    df["V_any_flag"] = (df[[c for c in v_frame.columns if c.endswith("_flag")]].sum(axis=1) > 0).astype(int)

    out = compute_outcomes(
        base_delivery_kw=(df["nominal_capacity_kw"] * df["response_rate"]).to_numpy(),
        response_rate=df["response_rate"].to_numpy(),
        response_delay_min=df["response_delay_min"].to_numpy(),
        event_intensity=df["event_intensity"].to_numpy(),
        V=vt, weights=cfg.dose_weights,
        noise=sample_shared_noise(rng, n),
    )
    for ch in ["delivery","comfort_loss","rebound_risk","contract_penalty","instability"]:
        df[ch] = getattr(out, ch)
    df["pred_reliable_deliverable_capacity_kw"] = (
        df["nominal_capacity_kw"] * df["response_rate"]
    )
    df["pair_id"] = [f"SCN{i:07d}" for i in range(n)]
    df = add_safety_columns(df, cfg)
    return df