"""Centralized configuration loader.

Reads the toolkit-level config.json and exposes a typed `Config` dataclass for
the rest of the pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    raw: dict[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls(raw=raw)

    # ---- Convenience accessors used across modules ----
    @property
    def seed(self) -> int:
        return int(self.raw["seed_policy"]["master_seed"])

    def seed_for(self, city: str, user_id: int, salt: int = 0) -> int:
        codes = self.raw["seed_policy"]["city_codes"]
        cc = codes[city]
        return int(self.raw["seed_policy"]["master_seed"]) + cc * 1_000_000 + int(user_id) + salt

    @property
    def violation_classes(self) -> list[str]:
        return list(self.raw["violation_tensor"]["classes"])

    @property
    def dose_weights(self) -> dict[str, float]:
        return dict(self.raw["violation_tensor"]["dose_response_weights"])

    @property
    def alpha_weights(self) -> dict[str, float]:
        return dict(self.raw["safety_cost"]["alpha_weights"])

    @property
    def lambda_safety(self) -> float:
        return float(self.raw["safety_cost"]["lambda"])

    @property
    def scenario_ratios(self) -> dict[str, float]:
        return dict(self.raw["scenarios"]["ratios"])

    @property
    def boundary_band(self) -> tuple[float, float]:
        b = self.raw["scenarios"]["boundary_demand_over_capacity"]
        return float(b["low"]), float(b["high"])

    @property
    def stress_clip(self) -> float:
        return float(self.raw["scenarios"]["stress_capacity_clip"])

    @property
    def propensity_bounds(self) -> tuple[float, float]:
        o = self.raw["overlap_repair"]
        return float(o["target_propensity_min"]), float(o["target_propensity_max"])

    @property
    def label_n_bins(self) -> int:
        return int(self.raw["label_fix"]["n_bins"])

    @property
    def label_names(self) -> list[str]:
        return list(self.raw["label_fix"]["labels"])

    @property
    def n_anchors(self) -> int:
        return int(self.raw["anchor_set"]["n_anchors"])