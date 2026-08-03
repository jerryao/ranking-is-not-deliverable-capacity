"""Multi-seed driver for Items 3 + 7.

For each seed in SEEDS:
  1. Create config with master_seed = 2048 + seed * 10**9
  2. Run upgrade_dataset_v2.py to generate fresh v2 dataset
  3. Run e3_replay_engine_v3.py to compute TSR/OCR/shortfall
  4. Collect summary

Outputs:
  - results/multiseed_summary.csv  (one row per seed * strategy * regime * level)
  - results/multiseed_tsr_table.csv (TSR mean/std/min/max across seeds)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOOLKIT = HERE / "v2_toolkit_paper2"
CONFIG_TEMPLATE = TOOLKIT / "config.json"
UPGRADE_SCRIPT = TOOLKIT / "upgrade_dataset_v2.py"
E3_V3 = HERE / "e3_replay_engine_v3.py"

V1_ROOT = r"D:\项目\在研\四川\Dataset\Sichuan2024KGSimDataset"
WORK_ROOT = HERE / "_multiseed_work"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# Number of seeds (smoke test = 3; full run = 30)
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
SEED_BASE = 2048
SEED_STRIDE = 10 ** 9


def make_config_for_seed(seed_id: int, out_path: Path) -> int:
    """Copy config.json and rewrite master_seed; return new master_seed value."""
    with open(CONFIG_TEMPLATE, encoding="utf-8") as f:
        cfg = json.load(f)
    new_seed = SEED_BASE + seed_id * SEED_STRIDE
    cfg["seed_policy"]["master_seed"] = new_seed
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return new_seed


def run_seed(seed_id: int) -> dict | None:
    """Generate dataset for one seed and run replay. Returns summary dict."""
    print(f"\n{'=' * 70}\n[seed {seed_id}] starting\n{'=' * 70}")
    work_dir = WORK_ROOT / f"seed_{seed_id:03d}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    cfg_path = work_dir / "config.json"
    new_seed = make_config_for_seed(seed_id, cfg_path)
    data_root = work_dir / "dataset_v2"

    # 1) generate
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, str(UPGRADE_SCRIPT),
         "--input-root", V1_ROOT,
         "--output-root", str(data_root),
         "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"[seed {seed_id}] upgrade FAILED:")
        print(r.stderr[-2000:])
        return None
    gen_t = time.time() - t0
    print(f"[seed {seed_id}] generation: {gen_t:.1f}s  master_seed={new_seed}")

    # 2) replay
    t1 = time.time()
    out_csv = work_dir / "replay.csv"
    r = subprocess.run(
        [sys.executable, str(E3_V3),
         "--data-v2", str(data_root),
         "--output", str(out_csv)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"[seed {seed_id}] replay FAILED:")
        print(r.stderr[-2000:])
        return None
    rep_t = time.time() - t1
    print(f"[seed {seed_id}] replay: {rep_t:.1f}s")

    # 3) collect
    df = pd.read_csv(out_csv)
    df["seed_id"] = seed_id
    df["master_seed"] = new_seed
    # Save per-seed replay into results
    df.to_csv(RESULTS / f"replay_seed_{seed_id:03d}.csv", index=False)

    summary = {
        "seed_id": seed_id,
        "master_seed": new_seed,
        "gen_time_s": gen_t,
        "replay_time_s": rep_t,
    }
    # Pool-level means
    for regime in ["routine", "cross_regime"]:
        for level in ["1x", "2x", "5x", "10x"]:
            sub = df[(df["regime"] == regime) & (df["p_req_level"] == level)]
            for strat in sub["strategy"].unique():
                ss = sub[sub["strategy"] == strat]
                summary[f"tsr_{regime}_{level}_{strat}"] = ss["success"].mean() * 100
                summary[f"ocr_{regime}_{level}_{strat}"] = ss["ocr"].mean()
                summary[f"nsel_{regime}_{level}_{strat}"] = ss["n_selected"].mean()
    return summary


def main():
    print(f"Multi-seed driver: N_SEEDS = {N_SEEDS}")
    WORK_ROOT.mkdir(exist_ok=True)
    summaries = []
    for sid in range(N_SEEDS):
        s = run_seed(sid)
        if s is not None:
            summaries.append(s)

    if not summaries:
        print("ALL SEEDS FAILED")
        return 1

    sdf = pd.DataFrame(summaries)
    sdf.to_csv(RESULTS / "multiseed_summary.csv", index=False)
    print(f"\nSaved summary: {RESULTS / 'multiseed_summary.csv'}")

    # Aggregate TSR table
    strat_cols = [c for c in sdf.columns if c.startswith("tsr_")]
    rows = []
    for c in strat_cols:
        # parse: tsr_<regime>_<level>_<strat>  where regime may be "routine" or "cross_regime"
        rest = c[len("tsr_"):]
        if rest.startswith("cross_regime_"):
            regime = "cross_regime"
            rest = rest[len("cross_regime_"):]
        elif rest.startswith("routine_"):
            regime = "routine"
            rest = rest[len("routine_"):]
        else:
            continue
        level, strat = rest.split("_", 1)
        vals = sdf[c].dropna().values
        rows.append({
            "regime": regime,
            "p_req_level": level,
            "strategy": strat,
            "tsr_mean": float(np.mean(vals)),
            "tsr_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "tsr_min": float(np.min(vals)) if len(vals) else float("nan"),
            "tsr_max": float(np.max(vals)) if len(vals) else float("nan"),
            "n_seeds": len(vals),
        })
    agg = pd.DataFrame(rows)
    agg.to_csv(RESULTS / "multiseed_tsr_table.csv", index=False)
    print(f"Saved TSR aggregate: {RESULTS / 'multiseed_tsr_table.csv'}")

    # Print headline comparisons
    print("\n" + "=" * 70)
    print("HEADLINE TSR COMPARISON (mean across seeds)")
    print("=" * 70)
    for regime in ["routine", "cross_regime"]:
        for level in ["1x", "2x", "5x", "10x"]:
            print(f"\n  {regime} {level}:")
            sub = agg[(agg["regime"] == regime) & (agg["p_req_level"] == level)]
            for _, row in sub.iterrows():
                print(f"    {row['strategy']:25s}: {row['tsr_mean']:6.1f}%  (std {row['tsr_std']:.1f}, range {row['tsr_min']:.0f}-{row['tsr_max']:.0f}, n={int(row['n_seeds'])})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
