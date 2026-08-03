"""Manifest-driven loader for the unified V1+V2 dataset.

Usage:
    from load_complete import CompleteDataset
    ds = CompleteDataset(r"D:/.../Sichuan2024KGSimDataset_complete")
    users     = ds.users()
    profiles  = ds.profiles_daily(city="City-A")        # chunked reader
    events    = ds.events()
    labels_dy = ds.structured_labels_dynamic()
    ta_v1     = ds.task_assessments_v1()
    ta_v2     = ds.task_assessments_v2_enhanced()
    pairs     = ds.intervention_pairs_long()
    pairs_tau = ds.intervention_pairs_tau()
    scn       = ds.decision_scenarios()
    anc       = ds.anchor_scenarios()
    safety    = ds.safety_cost_labels()

The loader is read-only and caches nothing: callers should cache the
DataFrames they reuse.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd


class CompleteDataset:
    """Manifest-driven reader for the unified Sichuan2024KGSimDataset_complete."""

    CITIES = ("City-A", "City-B", "City-C")

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._manifest_path = self.root / "MANIFEST.json"
        if not self._manifest_path.exists():
            raise FileNotFoundError(f"MANIFEST.json not found at {self._manifest_path}")
        with open(self._manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)
        # Quick index: file path → manifest entry
        self._by_rel = {e["rel"]: e for e in self.manifest.get("files", [])}

    # ---------- low-level path resolution ----------
    def _path(self, rel: str) -> Path:
        """Resolve a manifest-relative path. Works for both source_v1/* and v2/*."""
        if rel not in self._by_rel:
            raise KeyError(f"manifest does not list: {rel}")
        return self.root / rel

    def _csv(self, rel: str, **kw) -> pd.DataFrame:
        return pd.read_csv(self._path(rel), **kw)

    # ---------- V1 surface ----------
    def meta_v1(self) -> dict:
        return json.loads((self.root / "source_v1" / "meta.json").read_text(encoding="utf-8"))

    def complete_meta(self) -> dict:
        return json.loads((self.root / "complete_meta.json").read_text(encoding="utf-8"))

    def validation_results(self) -> dict:
        return json.loads((self.root / "validation" / "validation_results.json").read_text(encoding="utf-8"))

    def users(self) -> pd.DataFrame:
        parts = [self._csv(f"source_v1/{c}/users.csv") for c in self.CITIES]
        return pd.concat(parts, ignore_index=True)

    def events(self) -> pd.DataFrame:
        parts = [self._csv(f"source_v1/{c}/events.csv") for c in self.CITIES]
        return pd.concat(parts, ignore_index=True)

    def structured_labels(self) -> pd.DataFrame:
        parts = [self._csv(f"source_v1/{c}/structured_labels.csv") for c in self.CITIES]
        return pd.concat(parts, ignore_index=True)

    def structured_labels_dynamic(self) -> pd.DataFrame:
        parts = [self._csv(f"source_v1/{c}/structured_labels_dynamic.csv") for c in self.CITIES]
        return pd.concat(parts, ignore_index=True)

    def task_assessments_v1(self) -> pd.DataFrame:
        parts = [self._csv(f"source_v1/{c}/task_assessments.csv") for c in self.CITIES]
        return pd.concat(parts, ignore_index=True)

    def tasks(self) -> pd.DataFrame:
        return self._csv("source_v1/tasks.csv")

    def task_event_bindings(self) -> pd.DataFrame:
        return self._csv("source_v1/task_event_bindings.csv")

    # ---------- profiles_daily: chunked reader to save memory ----------
    def profiles_daily(self, city: str | None = None,
                       chunksize: int = 200_000) -> pd.DataFrame | Iterator[pd.DataFrame]:
        """If chunksize is None, load all (≈ 165 MB). Otherwise stream chunks."""
        cities = [city] if city else list(self.CITIES)
        if chunksize is None:
            parts = [self._csv(f"source_v1/{c}/profiles_daily.csv") for c in cities]
            return pd.concat(parts, ignore_index=True)
        def gen():
            for c in cities:
                for chunk in pd.read_csv(self._path(f"source_v1/{c}/profiles_daily.csv"),
                                         chunksize=chunksize):
                    yield chunk
        return gen()

    # ---------- V2 surface ----------
    def task_assessments_v2_enhanced(self) -> pd.DataFrame:
        return self._csv("observational/task_assessments_v2_enhanced.csv")

    def intervention_pairs_long(self) -> pd.DataFrame:
        return self._csv("intervention_pairs/intervention_pairs.csv")

    def intervention_pairs_tau(self) -> pd.DataFrame:
        return self._csv("intervention_pairs/intervention_pairs_tau.csv")

    def decision_scenarios(self) -> pd.DataFrame:
        return self._csv("scenario_sets/decision_scenarios.csv")

    def boundary_scenarios(self) -> pd.DataFrame:
        return self._csv("scenario_sets/boundary_scenarios.csv")

    def stress_scenarios(self) -> pd.DataFrame:
        return self._csv("scenario_sets/stress_scenarios.csv")

    def anchor_scenarios(self) -> pd.DataFrame:
        return self._csv("anchor_cross_city/anchor_scenarios.csv")

    def safety_cost_labels(self) -> pd.DataFrame:
        return self._csv("safety_cost_labels/safety_cost_labels.csv")

    def structured_labels_v2(self) -> pd.DataFrame:
        return self._csv("metadata/structured_labels_v2.csv")

    # ---------- utility ----------
    def summary(self) -> dict:
        return {
            "unified_root": str(self.root),
            "totals": self.manifest.get("totals"),
            "v2_validation": self.validation_results(),
        }