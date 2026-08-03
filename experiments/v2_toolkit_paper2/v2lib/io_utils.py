"""I/O helpers — reading the v1 source and writing the v2 outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_task_assessments(source_root: str | Path) -> pd.DataFrame:
    """Read task_assessments.csv from each city and concat."""
    source_root = Path(source_root)
    parts = []
    for city in ["City-A", "City-B", "City-C"]:
        p = source_root / city / "task_assessments.csv"
        df = pd.read_csv(p)
        df["__source_city_dir__"] = str(p.parent)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    return df


def load_users(source_root: str | Path) -> pd.DataFrame:
    parts = []
    for city in ["City-A", "City-B", "City-C"]:
        parts.append(pd.read_csv(Path(source_root) / city / "users.csv"))
    return pd.concat(parts, ignore_index=True)


def load_structured_labels(source_root: str | Path) -> pd.DataFrame:
    parts = []
    for city in ["City-A", "City-B", "City-C"]:
        parts.append(pd.read_csv(Path(source_root) / city / "structured_labels.csv"))
    return pd.concat(parts, ignore_index=True)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dirs(out_root: str | Path) -> dict[str, Path]:
    out_root = Path(out_root)
    sub = {
        "observational":     out_root / "observational",
        "intervention_pairs":out_root / "intervention_pairs",
        "scenario_sets":     out_root / "scenario_sets",
        "anchor_cross_city": out_root / "anchor_cross_city",
        "metadata":          out_root / "metadata",
        "validation":        out_root / "validation",
        "safety_cost_labels":out_root / "safety_cost_labels",
    }
    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    return sub