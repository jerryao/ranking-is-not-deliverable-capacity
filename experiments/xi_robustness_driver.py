"""ξ robustness sweep driver.

Sweeps the execution realization factor ξ across 4 dimensions:
  1. Mean: {0.58, 0.67, 0.75, 0.80}
  2. Std:  {0.05, 0.10, 0.15, 0.20}
  3. Distribution: {beta, truncnorm, logitnorm}
  4. Persistence: {none, half, full}

For each configuration, runs N_SEEDS independent seeds. Each seed:
  - Creates config with modified execution_factor section
  - Runs upgrade_dataset_v2.py to generate v2 dataset
  - Runs e3_replay_engine_v3.py to compute TSR/OCR
  - Collects headline metrics

Outputs:
  - results/xi_robustness_raw.csv (one row per config × seed × strategy × regime × level)
  - results/xi_robustness_summary.csv (aggregated by config)
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
WORK_ROOT = HERE / "_xi_robustness_work"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

N_SEEDS = int(os.environ.get("N_SEEDS", "10"))

# Define the sweep configurations.
# Each entry is (config_name, xi_params_dict).
# The baseline (mean=0.67, std=0.15, beta, none) is already covered by the
# main N=30 run; we run it here too at N=10 for direct comparability.
SWEEPS = []

# 1. Mean sensitivity (other params fixed at baseline)
for m in [0.58, 0.67, 0.75, 0.80]:
    SWEEPS.append((
        f"mean_{m}",
        {"mean": m, "std": 0.15, "dist": "beta", "persistence": "none"},
    ))

# 2. Std sensitivity (mean fixed at 0.67)
for s in [0.05, 0.10, 0.15, 0.20]:
    SWEEPS.append((
        f"std_{s}",
        {"mean": 0.67, "std": s, "dist": "beta", "persistence": "none"},
    ))

# 3. Distribution sensitivity (mean=0.67, std=0.15)
for d in ["beta", "truncnorm", "logitnorm"]:
    SWEEPS.append((
        f"dist_{d}",
        {"mean": 0.67, "std": 0.15, "dist": d, "persistence": "none"},
    ))

# 4. Persistence sensitivity (mean=0.67, std=0.15, beta)
for p in ["none", "half", "full"]:
    SWEEPS.append((
        f"persistence_{p}",
        {"mean": 0.67, "std": 0.15, "dist": "beta", "persistence": p},
    ))

# Deduplicate (baseline appears in all 4 groups)
seen = set()
UNIQUE_SWEEPS = []
for name, params in SWEEPS:
    key = (params["mean"], params["std"], params["dist"], params["persistence"])
    if key not in seen:
        seen.add(key)
        UNIQUE_SWEEPS.append((name, params))


def make_config(xi_params: dict, out_path: Path, master_seed: int) -> None:
    with open(CONFIG_TEMPLATE, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["seed_policy"]["master_seed"] = master_seed
    cfg["execution_factor"] = {
        "mean": xi_params["mean"],
        "std": xi_params["std"],
        "dist": xi_params["dist"],
        "persistence": xi_params["persistence"],
        "calibration_source": "PJM Summer 2025 DR performance ratio" if xi_params["mean"] == 0.67 else "sensitivity test",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def run_one(config_name: str, xi_params: dict, seed_id: int) -> dict | None:
    work = WORK_ROOT / config_name / f"seed_{seed_id:03d}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    master_seed = 2048 + seed_id * 10**9
    cfg_path = work / "config.json"
    make_config(xi_params, cfg_path, master_seed)
    data_root = work / "dataset_v2"

    # Generate
    r = subprocess.run(
        [sys.executable, str(UPGRADE_SCRIPT),
         "--input-root", V1_ROOT,
         "--output-root", str(data_root),
         "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  [{config_name}/seed_{seed_id}] GEN FAIL: {r.stderr[-500:]}")
        return None

    # Replay
    out_csv = work / "replay.csv"
    r = subprocess.run(
        [sys.executable, str(E3_V3),
         "--data-v2", str(data_root),
         "--output", str(out_csv)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  [{config_name}/seed_{seed_id}] REPLAY FAIL: {r.stderr[-500:]}")
        return None

    df = pd.read_csv(out_csv)
    df["config_name"] = config_name
    df["seed_id"] = seed_id
    df["xi_mean"] = xi_params["mean"]
    df["xi_std"] = xi_params["std"]
    df["xi_dist"] = xi_params["dist"]
    df["xi_persistence"] = xi_params["persistence"]
    df.to_csv(RESULTS / f"xi_rob_{config_name}_seed_{seed_id:03d}.csv", index=False)

    # Collect headline metrics
    summary = {"config_name": config_name, "seed_id": seed_id, **xi_params}
    for regime in ["routine", "cross_regime"]:
        for level in ["5x", "10x"]:
            for strat in ["S0a_RoutineOracle", "S1_CapRR", "S3_Lin4",
                          "S4_GlobalCalib", "S6_Q10"]:
                sub = df[(df["regime"] == regime)
                         & (df["p_req_level"] == level)
                         & (df["strategy"] == strat)]
                if len(sub):
                    summary[f"tsr_{regime}_{level}_{strat}"] = float(sub["success"].mean() * 100)
                    summary[f"ocr_{regime}_{level}_{strat}"] = float(sub["ocr"].mean())
                    summary[f"nsel_{regime}_{level}_{strat}"] = float(sub["n_selected"].mean())

    # Also extract rho from replay stdout
    for line in r.stdout.split("\n"):
        if "rho(Delivery-Linear4)" in line:
            parts = line.split("=")
            if len(parts) >= 2:
                summary["rho_lin4"] = float(parts[-1].split()[0])
        if "rho(Cap*RR)" in line:
            parts = line.split("=")
            if len(parts) >= 2:
                summary["rho_caprr"] = float(parts[-1].split()[0])
        if "M_routine: mean=" in line:
            summary["M_routine_mean"] = float(line.split("mean=")[1].split()[0])

    return summary


def main():
    print(f"ξ Robustness Sweep: {len(UNIQUE_SWEEPS)} configs × {N_SEEDS} seeds = {len(UNIQUE_SWEEPS) * N_SEEDS} runs")
    WORK_ROOT.mkdir(exist_ok=True)

    all_summaries = []
    t0 = time.time()
    for ci, (config_name, xi_params) in enumerate(UNIQUE_SWEEPS):
        print(f"\n[{ci+1}/{len(UNIQUE_SWEEPS)}] {config_name}: "
              f"mean={xi_params['mean']}, std={xi_params['std']}, "
              f"dist={xi_params['dist']}, pers={xi_params['persistence']}")
        for sid in range(N_SEEDS):
            s = run_one(config_name, xi_params, sid)
            if s:
                all_summaries.append(s)
                print(f"  seed {sid}: OK  "
                      f"S1_5x={s.get('tsr_routine_5x_S1_CapRR', '?'):.1f}%  "
                      f"S4_5x={s.get('tsr_routine_5x_S4_GlobalCalib', '?'):.1f}%")
            else:
                print(f"  seed {sid}: FAIL")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    sdf = pd.DataFrame(all_summaries)
    sdf.to_csv(RESULTS / "xi_robustness_summary.csv", index=False)
    print(f"\nSaved: {RESULTS / 'xi_robustness_summary.csv'}")

    # Aggregate by config
    print("\n" + "=" * 100)
    print("ξ ROBUSTNESS SUMMARY: TSR @5x routine (mean ± std across seeds)")
    print("=" * 100)
    print(f"{'config':25s} {'mean':>5s} {'std':>5s} {'dist':>10s} {'pers':>5s} | "
          f"{'S1_5x':>12s} {'S4_5x':>12s} {'S6_5x':>12s} {'rho_CapRR':>10s}")
    print("-" * 100)
    for config_name, xi_params in UNIQUE_SWEEPS:
        sub = sdf[sdf["config_name"] == config_name]
        if sub.empty:
            continue
        def fmt(col):
            if col not in sub.columns:
                return "N/A"
            vals = sub[col].dropna()
            if len(vals) == 0:
                return "N/A"
            return f"{vals.mean():.1f}±{vals.std():.1f}"
        print(f"{config_name:25s} {xi_params['mean']:5.2f} {xi_params['std']:5.2f} "
              f"{xi_params['dist']:>10s} {xi_params['persistence']:>5s} | "
              f"{fmt('tsr_routine_5x_S1_CapRR'):>12s} "
              f"{fmt('tsr_routine_5x_S4_GlobalCalib'):>12s} "
              f"{fmt('tsr_routine_5x_S6_Q10'):>12s} "
              f"{fmt('rho_caprr'):>10s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
