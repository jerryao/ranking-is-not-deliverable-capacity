"""Extract seed-clustered bootstrap CI for all key strategies at 5x."""
import numpy as np, pandas as pd
from pathlib import Path

RESULTS = Path(r"D:\项目\在研\四川\Dataset\paper2_calibration\experiments\results")
dfs = [pd.read_csv(f) for f in sorted(RESULTS.glob("replay_seed_*.csv"))]
pooled = pd.concat(dfs, ignore_index=True)
rng = np.random.default_rng(42)

KEYS = [
    ("routine", "S1_CapRR"),
    ("routine", "S4_GlobalCalib"),
    ("routine", "S6_Q10"),
    ("cross_regime", "S0a_RoutineOracle"),
    ("cross_regime", "S4_GlobalCalib"),
    ("cross_regime", "S6_Q10"),
]

for regime, strat in KEYS:
    sub = pooled[(pooled["regime"] == regime) & (pooled["p_req_level"] == "5x") & (pooled["strategy"] == strat)]
    per_seed = sub.groupby("seed_id")["success"].mean().values
    n = len(per_seed)
    boot = []
    for _ in range(5000):
        idx = rng.integers(0, n, size=n)
        boot.append(per_seed[idx].mean())
    boot = np.array(boot) * 100
    low = np.percentile(boot, 2.5)
    high = np.percentile(boot, 97.5)
    mean_tsr = per_seed.mean() * 100
    print(f"{regime:14s} {strat:25s}: {mean_tsr:.1f}%  bootstrap CI [{low:.1f}, {high:.1f}]")
