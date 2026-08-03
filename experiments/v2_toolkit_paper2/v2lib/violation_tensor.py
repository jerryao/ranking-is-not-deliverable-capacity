"""5-dimensional violation tensor (Instruction 2).

For each row we materialize:

    V = (indicator, severity, duration_h, scope)
        for k in {physical, mutex, comfort, hierarchy, contract}

Severity is in R+, duration in hours, scope in [0,1].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ViolationTensor:
    """Stores 5 classes × 4 components = 20 floats per row."""
    classes: list[str]
    indicator: np.ndarray   # (N, 5) in {0,1}
    severity:  np.ndarray   # (N, 5) in R+
    duration:  np.ndarray   # (N, 5) in R+ (hours)
    scope:     np.ndarray   # (N, 5) in [0,1]

    @property
    def n_rows(self) -> int:
        return int(self.indicator.shape[0])

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        cols = {}
        for j, k in enumerate(self.classes):
            cols[f"V_{k}_flag"]      = self.indicator[:, j].astype(int)
            cols[f"V_{k}_severity"]  = self.severity[:, j]
            cols[f"V_{k}_duration_h"]= self.duration[:, j]
            cols[f"V_{k}_scope"]     = self.scope[:, j]
        return pd.DataFrame(cols)

    @classmethod
    def from_frame(cls, df: "pd.DataFrame", classes: list[str]) -> "ViolationTensor":
        n = len(df)
        ind  = np.zeros((n, len(classes)), dtype=np.float32)
        sev  = np.zeros((n, len(classes)), dtype=np.float32)
        dur  = np.zeros((n, len(classes)), dtype=np.float32)
        scp  = np.zeros((n, len(classes)), dtype=np.float32)
        for j, k in enumerate(classes):
            ind[:, j] = df[f"V_{k}_flag"].to_numpy(dtype=np.float32)
            sev[:, j] = df[f"V_{k}_severity"].to_numpy(dtype=np.float32)
            dur[:, j] = df[f"V_{k}_duration_h"].to_numpy(dtype=np.float32)
            scp[:, j] = df[f"V_{k}_scope"].to_numpy(dtype=np.float32)
        return cls(classes=classes, indicator=ind, severity=sev, duration=dur, scope=scp)

    def any_violation(self) -> np.ndarray:
        return (self.indicator.sum(axis=1) > 0).astype(int)

    def treatment(self) -> np.ndarray:
        """Aggregate 'treatment indicator' (used for propensity analysis).
        A row is 'treated' if it experiences at least one non-contract violation,
        since contract is purely economic per the spec."""
        non_contract = [j for j, k in enumerate(self.classes) if k != "contract"]
        return (self.indicator[:, non_contract].sum(axis=1) > 0).astype(int)


# ----- Generators -----

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


def _lognormal(rng: np.random.Generator, n: int, mu: float, sigma: float) -> np.ndarray:
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


def _exponential(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return rng.exponential(scale=scale, size=n)


def _beta(rng: np.random.Generator, n: int, a: float, b: float) -> np.ndarray:
    return rng.beta(a, b, size=n)


def generate_violation_tensor(
    n_rows: int,
    *,
    base_rates: dict[str, float],
    severity_dist: dict,
    duration_dist: dict,
    scope_dist: dict,
    seed: int,
    do_override: dict[str, np.ndarray] | None = None,
) -> ViolationTensor:
    """Generate a ViolationTensor with the given marginal rates.

    `do_override` lets callers force indicator values for specific classes
    (used by intervention-pair generation to set V=0 or V=1 deterministically).
    """
    classes = list(base_rates.keys())
    rng = _rng(seed)
    n = n_rows
    ind = np.zeros((n, len(classes)), dtype=np.float32)
    sev = np.zeros((n, len(classes)), dtype=np.float32)
    dur = np.zeros((n, len(classes)), dtype=np.float32)
    scp = np.zeros((n, len(classes)), dtype=np.float32)

    for j, k in enumerate(classes):
        if do_override is not None and k in do_override:
            ind[:, j] = do_override[k].astype(np.float32)
        else:
            ind[:, j] = (rng.random(n) < float(base_rates[k])).astype(np.float32)
        # only generate sub-fields where indicator=1
        mask = ind[:, j] == 1
        nm = int(mask.sum())
        if nm > 0:
            sev[mask, j] = _lognormal(rng, nm,
                                       float(severity_dist["mu"]),
                                       float(severity_dist["sigma"]))
            dur[mask, j] = _exponential(rng, nm,
                                        float(duration_dist["scale_hours"]))
            scp[mask, j] = _beta(rng, nm,
                                  float(scope_dist["a"]),
                                  float(scope_dist["b"]))
    return ViolationTensor(classes=classes, indicator=ind, severity=sev, duration=dur, scope=scp)


def zero_violation_tensor(n_rows: int, classes: list[str]) -> ViolationTensor:
    """V=0 counterfactual: zero out all violations."""
    z = np.zeros((n_rows, len(classes)), dtype=np.float32)
    return ViolationTensor(classes=classes, indicator=z, severity=z.copy(),
                           duration=z.copy(), scope=z.copy())


def full_violation_tensor(n_rows: int, classes: list[str], *,
                          max_severity: float = 1.0,
                          max_duration_h: float = 4.0,
                          max_scope: float = 0.5,
                          seed: int | None = None) -> ViolationTensor:
    """V=1 counterfactual: turn every violation class ON at deterministic level.

    Using deterministic (not random) values for V=1 so that tau is a clean
    signal of the dose-response surface, not noise.
    """
    rng = _rng(seed if seed is not None else 0)
    n = n_rows
    ind = np.ones((n, len(classes)), dtype=np.float32)
    sev = np.full((n, len(classes)), max_severity, dtype=np.float32)
    dur = np.full((n, len(classes)), max_duration_h, dtype=np.float32)
    scp = np.full((n, len(classes)), max_scope, dtype=np.float32)
    # tiny deterministic jitter so values are not exactly equal across rows
    if n > 0:
        jitter = (rng.random((n, len(classes))) - 0.5) * 0.02
        sev = np.clip(sev + jitter, 0.0, None)
    return ViolationTensor(classes=classes, indicator=ind, severity=sev,
                           duration=dur, scope=scp)