"""Dose-response mechanism (Instruction 3).

Y = f(X, A, V_k, severity_k) — a transparent linear-formula generator so that
ground-truth tau is auditable. Five outcome channels:

    Y_delivery        (kW, lower-bounded at 0)
    Y_comfort_loss    ([0,1])
    Y_rebound_risk    ([0,1])
    Y_contract_penalty([0,1])
    Y_instability     ([0,1])

Per-class impact weights are loaded from config.json (dose_response_weights).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .violation_tensor import ViolationTensor


@dataclass
class Outcomes:
    delivery: np.ndarray          # (N,)
    comfort_loss: np.ndarray      # (N,)
    rebound_risk: np.ndarray      # (N,)
    contract_penalty: np.ndarray  # (N,)
    instability: np.ndarray       # (N,)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _safe_lower(x: np.ndarray, floor: float = 0.0) -> np.ndarray:
    return np.maximum(x, floor)


def _class_score(V: ViolationTensor, weights: dict[str, float]) -> np.ndarray:
    """Sum_k weight_k * indicator_k * severity_k * scope_k."""
    n = V.n_rows
    score = np.zeros(n, dtype=np.float64)
    for j, k in enumerate(V.classes):
        w = float(weights[f"{k}_on_delivery"]) if f"{k}_on_delivery" in weights else 0.0
        # generic key shape: {class}_on_{outcome}
        score += w * V.indicator[:, j] * V.severity[:, j] * V.scope[:, j]
    return score


def _outcome_score(V: ViolationTensor, outcome: str, weights: dict[str, float]) -> np.ndarray:
    """Generic per-outcome score using weights keys of shape '{class}_on_{outcome}'."""
    n = V.n_rows
    score = np.zeros(n, dtype=np.float64)
    for j, k in enumerate(V.classes):
        key = f"{k}_on_{outcome}"
        if key in weights:
            score += float(weights[key]) * V.indicator[:, j] * V.severity[:, j] * V.scope[:, j]
    return score


def compute_outcomes(
    *,
    base_delivery_kw: np.ndarray,
    response_rate: np.ndarray,
    response_delay_min: np.ndarray,
    event_intensity: np.ndarray,
    V: ViolationTensor,
    weights: dict[str, float],
    noise: dict[str, np.ndarray] | None = None,
    execution_factor: np.ndarray | None = None,
) -> Outcomes:
    """Apply dose-response model.

    `base_delivery_kw` is the user-task-event capacity under no violation.
    `response_rate` is the no-violation response rate.
    `noise` is a dict of pre-sampled epsilons shared between Y0 and Y1 paths
    (Instruction 1 / 5). When None, fresh noise is drawn (used only by tests).
    `execution_factor` is an optional per-row multiplicative execution degradation
    factor xi in (0, 1] representing real-world delivery shortfall sources
    (equipment, behaviour, comms, baseline error) beyond the violation model.
    When provided, both V=0 and V=1 paths share the same xi, preserving the
    exact closed-form tau. Calibrated to PJM/MISO/NYISO empirical delivery
    ratios (mean ~0.67, see Paper 2 literature survey).
    """
    n = V.n_rows
    if base_delivery_kw.shape[0] != n:
        raise ValueError("base_delivery_kw length mismatch with V.n_rows")

    # Per-outcome scores
    s_delivery = _outcome_score(V, "delivery", weights)
    s_comfort  = _outcome_score(V, "comfort",  weights)
    s_rebound  = _outcome_score(V, "rebound",  weights)
    s_contract = _outcome_score(V, "contract_penalty", weights)
    s_instab   = _outcome_score(V, "instability", weights)

    # delivery: capacity * response_rate * exp(score)  +  clipping
    raw_delivery = base_delivery_kw * response_rate * np.exp(s_delivery)
    # also scale mildly by event_intensity (supports demand shock)
    raw_delivery = raw_delivery * (1.0 + 0.15 * (event_intensity - 0.3))
    if execution_factor is not None:
        raw_delivery = raw_delivery * execution_factor
    delivery = _safe_lower(raw_delivery, 0.0)

    # comfort loss: bounded in [0,1]
    comfort_loss = _clip01(0.10 + s_comfort + 0.10 * response_delay_min / 30.0)

    # rebound risk: bounded
    rebound = _clip01(0.05 + s_rebound + 0.05 * (1.0 - response_rate))

    # contract penalty
    contract = _clip01(0.02 + s_contract)

    # instability
    instab = _clip01(0.05 + s_instab + 0.10 * response_delay_min / 30.0)

    # Inject shared noise if provided (additive, small)
    if noise is not None:
        if "delivery" in noise:
            delivery = np.maximum(delivery + noise["delivery"], 0.0)
        if "comfort_loss" in noise:
            comfort_loss = _clip01(comfort_loss + noise["comfort_loss"])
        if "rebound_risk" in noise:
            rebound = _clip01(rebound + noise["rebound_risk"])
        if "contract_penalty" in noise:
            contract = _clip01(contract + noise["contract_penalty"])
        if "instability" in noise:
            instab = _clip01(instab + noise["instability"])

    return Outcomes(delivery=delivery, comfort_loss=comfort_loss,
                    rebound_risk=rebound, contract_penalty=contract,
                    instability=instab)


def sample_shared_noise(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Sample epsilons once per row, shared between Y0 and Y1 paths."""
    return {
        "delivery":        rng.normal(0.0, 0.5,  size=n),  # kW
        "comfort_loss":    rng.normal(0.0, 0.02, size=n),
        "rebound_risk":    rng.normal(0.0, 0.02, size=n),
        "contract_penalty":rng.normal(0.0, 0.01, size=n),
        "instability":     rng.normal(0.0, 0.02, size=n),
    }