"""Oracle simulator (Instruction 5).

Given (X, T, event, seed, V_override), returns Y0, Y1, and tau_true
where Y1 is computed under V=1 (every violation class forced ON at deterministic
level) and Y0 under V=0 (no violations). Both paths share the same epsilon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .violation_tensor import (
    ViolationTensor,
    zero_violation_tensor,
    full_violation_tensor,
)
from .dose_response import Outcomes, compute_outcomes, sample_shared_noise
from .xi_sampler import sample_xi


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


class OracleSimulator:
    """Pure functional simulator with shared noise across counterfactual paths."""

    def __init__(self, config: Config):
        self.cfg = config
        self.v_classes = config.violation_classes
        self.weights = config.dose_weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def simulate_pair(
        self,
        base: pd.DataFrame,
        *,
        seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run do(V=0) and do(V=1) on every row of `base`.

        Returns three DataFrames:
          - df0: same as base plus V tensor (all zero) and Y_*_0 columns
          - df1: same as base plus V tensor (all ones) and Y_*_1 columns
          - df_tau: per-row tau (Y1 - Y0) for each outcome
        """
        n = len(base)
        V0 = zero_violation_tensor(n, self.v_classes)
        V1 = full_violation_tensor(n, self.v_classes, seed=seed)

        # Item 3 fix (Paper 2 review): use connected-load nominal_capacity_kw as
        # the pre-response-rate base, so that response_rate is applied exactly
        # once (in dose_response.compute_outcomes). The previous field
        # `pred_reliable_deliverable_capacity_kw` already bakes in a partial
        # response-rate adjustment from the v1 platform-assessment model,
        # which resulted in an effective ~1.5x compounding of response_rate.
        base_delivery = base["nominal_capacity_kw"].to_numpy(dtype=np.float64)
        response_rate = base["response_rate"].to_numpy(dtype=np.float64)
        response_delay = base["response_delay_min"].to_numpy(dtype=np.float64)
        event_intensity = base["event_intensity"].to_numpy(dtype=np.float64)

        rng = _rng(seed)
        noise = sample_shared_noise(rng, n)

        rng_xi = np.random.default_rng(np.random.SeedSequence(seed + 999))
        xi_cfg = self.cfg.raw.get("execution_factor", {})
        xi = sample_xi(
            rng_xi, n,
            user_ids=base["user_id"].to_numpy(dtype=np.int64) if "user_id" in base.columns else None,
            mean=xi_cfg.get("mean", 0.67),
            std=xi_cfg.get("std", 0.15),
            dist=xi_cfg.get("dist", "beta"),
            persistence=xi_cfg.get("persistence", "none"),
        )

        out0 = compute_outcomes(
            base_delivery_kw=base_delivery,
            response_rate=response_rate,
            response_delay_min=response_delay,
            event_intensity=event_intensity,
            V=V0, weights=self.weights, noise=noise,
            execution_factor=xi,
        )
        out1 = compute_outcomes(
            base_delivery_kw=base_delivery,
            response_rate=response_rate,
            response_delay_min=response_delay,
            event_intensity=event_intensity,
            V=V1, weights=self.weights, noise=noise,
            execution_factor=xi,
        )

        df0 = base.copy().reset_index(drop=True)
        df1 = base.copy().reset_index(drop=True)
        V0f = V0.to_frame()
        V1f = V1.to_frame()
        for col in V0f.columns:
            df0[col] = V0f[col].to_numpy()
            df1[col] = V1f[col].to_numpy()

        for ch in ["delivery","comfort_loss","rebound_risk","contract_penalty","instability"]:
            df0[f"Y_{ch}_0"] = getattr(out0, ch)
            df1[f"Y_{ch}_1"] = getattr(out1, ch)

        df_tau = pd.DataFrame({
            "pair_id": base["pair_id"].to_numpy() if "pair_id" in base.columns
                       else np.arange(n),
            "city": base["city"].to_numpy(),
            "user_id": base["user_id"].to_numpy(),
            "task_id": base["task_id"].to_numpy(),
            "event_id": base["event_id"].to_numpy(),
            "seed": np.full(n, seed, dtype=np.int64),
            "true_tau_delivery":   out1.delivery - out0.delivery,
            "true_tau_comfort_loss": out1.comfort_loss - out0.comfort_loss,
            "true_tau_rebound_risk": out1.rebound_risk - out0.rebound_risk,
            "true_tau_contract_penalty": out1.contract_penalty - out0.contract_penalty,
            "true_tau_instability": out1.instability - out0.instability,
            "Y_delivery_0": out0.delivery,
            "Y_delivery_1": out1.delivery,
            "Y_comfort_0": out0.comfort_loss,
            "Y_comfort_1": out1.comfort_loss,
            "Y_rebound_0": out0.rebound_risk,
            "Y_rebound_1": out1.rebound_risk,
        })
        return df0, df1, df_tau