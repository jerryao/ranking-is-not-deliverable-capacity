"""P1: Top-k metrics + seed-clustered bootstrap CI + feasibility table.

Three analyses in one script:
  1. Top-k overlap and capacity regret at portfolio-relevant k values
  2. Seed-clustered bootstrap CI (vs pooled Wilson CI)
  3. Pool feasibility and utilisation table (reviewer Items 15/17)

Uses existing N=30 xi-enhanced datasets + replay results.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MULTISEED_WORK = HERE / "_multiseed_work"
V1_ROOT = r"D:\项目\在研\四川\Dataset\Sichuan2024KGSimDataset"

N_EXISTING = 10  # Use 10 seeds for top-k and feasibility (fast enough)


# ============================================================
# 1. Top-k metrics
# ============================================================
def compute_topk_metrics(seed_id):
    data_root = MULTISEED_WORK / f"seed_{seed_id:03d}" / "dataset_v2"
    if not data_root.exists():
        return None
    obs = pd.read_csv(data_root / "observational" / "task_assessments_v2_enhanced.csv")
    tasks = pd.read_csv(os.path.join(V1_ROOT, "tasks.csv"))
    obs["M_routine"] = obs["delivery"].clip(lower=0)
    obs["S1_pred"] = (obs["nominal_capacity_kw"] * obs["response_rate"]).clip(lower=0)
    obs["S4_pred"] = np.nan

    # Fit GlobalCalib via LOTO for S4 predictions
    from numpy.linalg import lstsq
    for hold_task in sorted(obs["task_id"].unique()):
        train_mask = obs["task_id"] != hold_task
        test_mask = obs["task_id"] == hold_task
        train = obs[train_mask]
        pr_tr = train["S1_pred"].values
        X_g = np.column_stack([np.ones(len(train)), pr_tr])
        beta_g, _, _, _ = lstsq(X_g, train["M_routine"].values, rcond=None)
        pr_te = obs.loc[test_mask, "S1_pred"].values
        X_te = np.column_stack([np.ones(test_mask.sum()), pr_te])
        obs.loc[test_mask, "S4_pred"] = np.clip(X_te @ beta_g, 0, None)

    task_p = {t["task_id"]: t["required_capacity_kw"] for _, t in tasks.iterrows()}
    rows = []

    for city in sorted(obs["city"].unique()):
        for task_id in sorted(obs["task_id"].unique()):
            pool = obs[(obs["city"] == city) & (obs["task_id"] == task_id)].copy()
            if pool.empty:
                continue
            p_base = task_p[task_id]

            # Oracle ranking (by actual M_routine) — keep original index
            oracle_ranked = pool.sort_values("M_routine", ascending=False)

            for strat, pred_col in [("S1_CapRR", "S1_pred"), ("S4_GlobalCalib", "S4_pred")]:
                strat_ranked = pool.sort_values(pred_col, ascending=False)

                for mult in [1, 2, 5, 10]:
                    p_req = p_base * mult
                    # Determine k from the strategy's own selection
                    cum = strat_ranked[pred_col].cumsum()
                    k = min(int((cum < p_req).sum()) + 1, len(strat_ranked))
                    if k == 0:
                        continue

                    # Top-k sets (original indices)
                    strat_topk_ids = set(strat_ranked.index[:k])
                    oracle_topk_ids = set(oracle_ranked.index[:k])
                    overlap = len(strat_topk_ids & oracle_topk_ids) / k

                    # Capacity of each top-k set
                    cap_strat = pool.loc[list(strat_topk_ids), "M_routine"].sum()
                    cap_oracle = pool.loc[list(oracle_topk_ids), "M_routine"].sum()
                    regret = 1 - cap_strat / cap_oracle if cap_oracle > 0 else 0.0

                    rows.append(dict(
                        seed_id=seed_id, city=city, task_id=task_id,
                        strategy=strat, req_level=f"{mult}x", k=k,
                        topk_overlap=overlap, capacity_regret=regret,
                        cap_strat=cap_strat, cap_oracle=cap_oracle,
                    ))
    return pd.DataFrame(rows)


# ============================================================
# 2. Seed-clustered bootstrap CI
# ============================================================
def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def seed_clustered_bootstrap(replay_df, n_bootstrap=2000, alpha=0.05):
    """Bootstrap CI that resamples SEEDS (not individual cells).

    Captures between-seed variability, which pooled Wilson CI misses.
    """
    rng = np.random.default_rng(42)
    results = {}
    grouped = replay_df.groupby(["regime", "p_req_level", "strategy"])

    for (regime, level, strat), sub in grouped:
        # Per-seed success rates
        per_seed = sub.groupby("seed_id")["success"].mean().values
        n_seeds = len(per_seed)
        if n_seeds == 0:
            continue

        # Bootstrap: resample seeds with replacement
        boot_means = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_seeds, size=n_seeds)
            boot_means.append(per_seed[idx].mean())
        boot_means = np.array(boot_means) * 100

        results[(regime, level, strat)] = {
            "tsr_mean": float(np.mean(per_seed) * 100),
            "boot_ci_low": float(np.percentile(boot_means, alpha / 2 * 100)),
            "boot_ci_high": float(np.percentile(boot_means, (1 - alpha / 2) * 100)),
            "n_seeds": n_seeds,
        }
    return results


# ============================================================
# 3. Feasibility table
# ============================================================
def compute_feasibility(seed_id):
    data_root = MULTISEED_WORK / f"seed_{seed_id:03d}" / "dataset_v2"
    if not data_root.exists():
        return None
    obs = pd.read_csv(data_root / "observational" / "task_assessments_v2_enhanced.csv")
    ip = pd.read_csv(data_root / "intervention_pairs" / "intervention_pairs_tau.csv")
    tasks = pd.read_csv(os.path.join(V1_ROOT, "tasks.csv"))

    obs["M_routine"] = obs["delivery"].clip(lower=0)
    obs = obs.merge(ip[["city", "user_id", "task_id", "event_id", "Y_delivery_1"]],
                    on=["city", "user_id", "task_id", "event_id"])
    obs["M_stress"] = obs["Y_delivery_1"].clip(lower=0)

    task_p = {t["task_id"]: t["required_capacity_kw"] for _, t in tasks.iterrows()}
    rows = []
    for city in sorted(obs["city"].unique()):
        for task_id in sorted(obs["task_id"].unique()):
            pool = obs[(obs["city"] == city) & (obs["task_id"] == task_id)]
            if pool.empty:
                continue
            p_base = task_p[task_id]
            total_nominal = pool["nominal_capacity_kw"].sum()
            total_routine = pool["M_routine"].sum()
            total_stress = pool["M_stress"].sum()
            n_users = len(pool)

            for mult_label, mult in [("1x", 1), ("5x", 5), ("10x", 10)]:
                req = p_base * mult
                rows.append(dict(
                    seed_id=seed_id, city=city, task_id=task_id,
                    n_pool=n_users, req_level=mult_label, req_kw=req,
                    total_nominal=total_nominal,
                    total_routine=total_routine, total_stress=total_stress,
                    util_routine=req / total_routine if total_routine > 0 else float("inf"),
                    util_stress=req / total_stress if total_stress > 0 else float("inf"),
                    feas_routine=total_routine >= req,
                    feas_stress=total_stress >= req,
                ))
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 80)
    print("P1: TOP-K METRICS + BOOTSTRAP CI + FEASIBILITY TABLE")
    print("=" * 80)

    # --- 1. Top-k ---
    print("\n[1] Computing top-k overlap and capacity regret...")
    topk_rows = []
    for sid in range(N_EXISTING):
        df = compute_topk_metrics(sid)
        if df is not None:
            topk_rows.append(df)
    topk = pd.concat(topk_rows, ignore_index=True)
    topk.to_csv(RESULTS / "topk_metrics.csv", index=False)

    print("\n" + "=" * 90)
    print("TOP-K OVERLAP AND CAPACITY REGRET (mean across 10 seeds × 15 cells)")
    print("=" * 90)
    print(f"{'Strategy':15s} {'Req':4s} {'mean_k':>7s} | {'overlap':>10s} {'cap_regret':>12s}")
    print("-" * 60)
    for strat in ["S1_CapRR", "S4_GlobalCalib"]:
        for level in ["1x", "2x", "5x", "10x"]:
            sub = topk[(topk["strategy"] == strat) & (topk["req_level"] == level)]
            if sub.empty:
                continue
            print(f"{strat:15s} {level:4s} {sub['k'].mean():7.1f} | "
                  f"{sub['topk_overlap'].mean():10.4f} {sub['capacity_regret'].mean():12.6f}")

    # --- 2. Bootstrap CI ---
    print("\n[2] Computing seed-clustered bootstrap CI...")
    # Load all 30 replay files for the xi-enhanced run
    replay_dfs = []
    for f in sorted(RESULTS.glob("replay_seed_*.csv")):
        df = pd.read_csv(f)
        if "success" in df.columns and "seed_id" in df.columns:
            replay_dfs.append(df)
    if not replay_dfs:
        print("  No replay files found. Skipping bootstrap CI.")
    else:
        replay_pooled = pd.concat(replay_dfs, ignore_index=True)
        n_seeds = replay_pooled["seed_id"].nunique()
        print(f"  Loaded {len(replay_pooled)} rows from {n_seeds} seeds")

        boot_ci = seed_clustered_bootstrap(replay_pooled)

        # Also compute pooled Wilson for comparison
        print("\n" + "=" * 100)
        print("CI COMPARISON: Pooled Wilson vs Seed-Clustered Bootstrap @5x")
        print("=" * 100)
        print(f"{'Regime':14s} {'Strategy':24s} {'TSR%':>6s} | "
              f"{'Wilson95':>18s} {'Boot95':>18s}")
        print("-" * 100)
        for regime in ["routine", "cross_regime"]:
            for level in ["5x"]:
                for strat in ["S0a_RoutineOracle", "S1_CapRR", "S4_GlobalCalib",
                              "S6_Q10", "S0b_StressOracle"]:
                    sub = replay_pooled[(replay_pooled["regime"] == regime)
                                        & (replay_pooled["p_req_level"] == level)
                                        & (replay_pooled["strategy"] == strat)]
                    if sub.empty:
                        continue
                    k_success = int(sub["success"].sum())
                    n_total = len(sub)
                    tsr = k_success / n_total * 100
                    wil_low, wil_high = wilson_ci(k_success, n_total)
                    key = (regime, level, strat)
                    if key in boot_ci:
                        bc = boot_ci[key]
                        print(f"{regime:14s} {strat:24s} {tsr:6.1f} | "
                              f"[{wil_low*100:5.1f}, {wil_high*100:5.1f}]   "
                              f"[{bc['boot_ci_low']:5.1f}, {bc['boot_ci_high']:5.1f}]")
            print("-" * 100)

    # --- 3. Feasibility ---
    print("\n[3] Computing feasibility table...")
    feas_rows = []
    for sid in range(min(N_EXISTING, 5)):  # 5 seeds enough for feasibility
        df = compute_feasibility(sid)
        if df is not None:
            feas_rows.append(df)
    feas = pd.concat(feas_rows, ignore_index=True)
    feas.to_csv(RESULTS / "feasibility_table.csv", index=False)

    print("\n" + "=" * 100)
    print("POOL FEASIBILITY AND UTILISATION (mean across 5 seeds)")
    print("=" * 100)
    print(f"{'City':8s} {'Task':5s} {'N':>5s} {'Req':>6s} | "
          f"{'TotNom':>10s} {'TotRout':>10s} {'TotStress':>10s} | "
          f"{'Util_R':>8s} {'Util_S':>8s} {'Feas_S':>6s}")
    print("-" * 100)
    sub5 = feas[feas["req_level"] == "5x"]
    for _, row in sub5.groupby(["city", "task_id"]).mean(numeric_only=True).reset_index().iterrows():
        feas_s = "Y" if row["feas_stress"] > 0.5 else "N"
        print(f"{row['city']:8s} {row['task_id']:5s} {row['n_pool']:5.0f} {row['req_kw']:6.0f} | "
              f"{row['total_nominal']:10.0f} {row['total_routine']:10.0f} {row['total_stress']:10.0f} | "
              f"{row['util_routine']:8.4f} {row['util_stress']:8.4f} {feas_s:>6s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
