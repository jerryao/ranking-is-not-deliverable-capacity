"""Compute Wilson 95% CI for binary TSR outcomes and aggregate across 30 seeds.

For TSR at a given (regime, level, strategy):
  - Per-seed TSR = (n_success_cells / 15) * 100
  - Across-seed mean and std are reported by multiseed_driver
  - Here we ALSO compute cell-level Wilson CI: pool all 30 seeds * 15 cells = 450
    binary outcomes and compute the Wilson interval for the resulting proportion.

Outputs:
  - results/multiseed_tsr_with_ci.csv (one row per regime*level*strategy)
  - Prints headline comparison table
"""
import os
import math
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Pool all per-seed replay files
dfs = []
for f in sorted(RESULTS.glob("replay_seed_*.csv")):
    dfs.append(pd.read_csv(f))
pooled = pd.concat(dfs, ignore_index=True)
print(f"Pooled {len(dfs)} seed files, total {len(pooled)} rows")
print(f"Per (regime, level, strategy): {len(pooled) // (2*4*8)} rows = 30 seeds * 15 cells")


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = math.isqrt(2) if False else 1.959963984540054  # z_{0.975}
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


STRATEGIES = sorted(pooled["strategy"].unique())
rows = []
for regime in ["routine", "cross_regime"]:
    for level in ["1x", "2x", "5x", "10x"]:
        for strat in STRATEGIES:
            sub = pooled[(pooled["regime"] == regime)
                         & (pooled["p_req_level"] == level)
                         & (pooled["strategy"] == strat)]
            n_total = len(sub)
            k_success = int(sub["success"].sum())
            # Across-seed stats
            per_seed = sub.groupby("seed_id")["success"].mean().values * 100
            # Pooled Wilson (cell-level across all 30 seeds * 15 cells)
            wil_low, wil_high = wilson_ci(k_success, n_total)
            row = {
                "regime": regime,
                "p_req_level": level,
                "strategy": strat,
                "tsr_pct_pooled": k_success / n_total * 100 if n_total else float("nan"),
                "tsr_mean_across_seeds_pct": float(np.mean(per_seed)) if len(per_seed) else float("nan"),
                "tsr_std_across_seeds_pct": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                "tsr_min_across_seeds_pct": float(np.min(per_seed)) if len(per_seed) else float("nan"),
                "tsr_max_across_seeds_pct": float(np.max(per_seed)) if len(per_seed) else float("nan"),
                "n_success_cells": k_success,
                "n_total_cells": n_total,
                "n_seeds": int(sub["seed_id"].nunique()),
                "wilson95_low_pct": wil_low * 100,
                "wilson95_high_pct": wil_high * 100,
                "mean_ocr": float(sub["ocr"].mean()),
                "mean_shortfall_kw": float(sub["shortfall"].mean()),
                "mean_n_selected": float(sub["n_selected"].mean()),
            }
            rows.append(row)

agg = pd.DataFrame(rows)
out_path = RESULTS / "multiseed_tsr_with_ci.csv"
agg.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# Print headline
print("\n" + "=" * 110)
print(f"{'Regime':12s} {'Lvl':4s} {'Strategy':24s} {'TSR%':>6s} {'Wilson95% CI':>20s} "
      f"{'SeedMean±Std':>15s} {'OCR':>6s} {'|S|':>5s}")
print("=" * 110)
for regime in ["routine", "cross_regime"]:
    for level in ["1x", "2x", "5x", "10x"]:
        for strat in STRATEGIES:
            r = agg[(agg["regime"] == regime)
                    & (agg["p_req_level"] == level)
                    & (agg["strategy"] == strat)].iloc[0]
            ci_str = f"[{r['wilson95_low_pct']:5.1f}, {r['wilson95_high_pct']:5.1f}]"
            seed_str = f"{r['tsr_mean_across_seeds_pct']:.1f}±{r['tsr_std_across_seeds_pct']:.1f}"
            count_str = f"{r['n_success_cells']}/{r['n_total_cells']}"
            print(f"{regime:12s} {level:4s} {strat:24s} {r['tsr_pct_pooled']:6.1f} {ci_str:>20s} "
                  f"{seed_str:>15s} {r['mean_ocr']:6.3f} {r['mean_n_selected']:5.1f}  ({count_str})")
    print("-" * 110)
