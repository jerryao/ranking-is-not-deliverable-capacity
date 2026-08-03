"""P0-4: User-level and city-level holdout experiments.

Tests whether calibration generalises beyond the leave-one-task-out protocol:
  1. Leave-one-task-out (existing): same users, new task template
  2. User holdout: NEW users (all records held out)
  3. City holdout: NEW city configuration

For each protocol, fits GlobalCalib and Q10 on training folds only,
evaluates on held-out records. Reports TSR at 5x routine.

Output: results/holdout_sensitivity.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MULTISEED_WORK = HERE / "_multiseed_work"
V1_ROOT = r"D:\项目\在研\四川\Dataset\Sichuan2024KGSimDataset"

N_EXISTING = 10


def replay_one(pool, mhat_col, p_req, oracle_col):
    ranked = pool.sort_values(mhat_col, ascending=False).reset_index(drop=True)
    if ranked.empty or ranked[mhat_col].iloc[0] <= 0:
        return dict(success=0.0, n_selected=0, shortfall=float(p_req), ocr=0.0)
    cum = ranked[mhat_col].cumsum()
    n_sel = min(int((cum < p_req).sum()) + 1, len(ranked))
    sel = ranked.iloc[:n_sel]
    p_commit = sel[mhat_col].sum()
    p_deliv = sel[oracle_col].sum()
    shortfall = max(0.0, p_req - p_deliv)
    ocr = max(0.0, p_commit - p_deliv) / p_commit if p_commit > 0 else 0.0
    return dict(success=1.0 if p_deliv >= p_req else 0.0, n_selected=n_sel,
                p_commit=p_commit, p_deliv=p_deliv, shortfall=shortfall, ocr=ocr)


def fit_globalcalib(train_pr, train_y):
    X = np.column_stack([np.ones(len(train_pr)), train_pr])
    beta, _, _, _ = lstsq(X, train_y, rcond=None)
    return beta


def fit_q_quantile(train_pr, train_y, alpha):
    ratio = train_y / np.clip(train_pr, 1e-6, None)
    return np.quantile(ratio, alpha)


def run_holdout_experiment(seed_id):
    data_root = MULTISEED_WORK / f"seed_{seed_id:03d}" / "dataset_v2"
    if not data_root.exists():
        return None
    obs = pd.read_csv(data_root / "observational" / "task_assessments_v2_enhanced.csv")
    tasks = pd.read_csv(os.path.join(V1_ROOT, "tasks.csv"))
    obs["M_routine"] = obs["delivery"].clip(lower=0)
    obs["pr"] = (obs["nominal_capacity_kw"] * obs["response_rate"]).clip(lower=0)
    obs["mhat_S1"] = obs["pr"]

    task_p = {t["task_id"]: t["required_capacity_kw"] for _, t in tasks.iterrows()}
    rows = []

    # --- Protocol 1: Leave-one-task-out (baseline, already computed) ---
    obs["mhat_S4_loto"] = np.nan
    obs["mhat_Q10_loto"] = np.nan
    for hold_task in sorted(obs["task_id"].unique()):
        train_mask = obs["task_id"] != hold_task
        test_mask = obs["task_id"] == hold_task
        train = obs[train_mask]
        beta = fit_globalcalib(train["pr"].values, train["M_routine"].values)
        obs.loc[test_mask, "mhat_S4_loto"] = np.clip(
            np.column_stack([np.ones(test_mask.sum()), obs.loc[test_mask, "pr"].values]) @ beta, 0, None)
        q10 = fit_q_quantile(train["pr"].values, train["M_routine"].values, 0.10)
        obs.loc[test_mask, "mhat_Q10_loto"] = np.clip(obs.loc[test_mask, "pr"].values * q10, 0, None)

    # --- Protocol 2: User-level holdout (5-fold by user) ---
    users = obs[["city", "user_id"]].drop_duplicates().sort_values(["city", "user_id"]).reset_index(drop=True)
    rng = np.random.default_rng(seed_id + 7000)
    users["user_fold"] = rng.integers(0, 5, size=len(users))
    obs = obs.merge(users, on=["city", "user_id"], how="left")

    obs["mhat_S4_user"] = np.nan
    obs["mhat_Q10_user"] = np.nan
    for fold in range(5):
        train_mask = obs["user_fold"] != fold
        test_mask = obs["user_fold"] == fold
        train = obs[train_mask]
        beta = fit_globalcalib(train["pr"].values, train["M_routine"].values)
        obs.loc[test_mask, "mhat_S4_user"] = np.clip(
            np.column_stack([np.ones(test_mask.sum()), obs.loc[test_mask, "pr"].values]) @ beta, 0, None)
        q10 = fit_q_quantile(train["pr"].values, train["M_routine"].values, 0.10)
        obs.loc[test_mask, "mhat_Q10_user"] = np.clip(obs.loc[test_mask, "pr"].values * q10, 0, None)

    # --- Protocol 3: City-level holdout ---
    obs["mhat_S4_city"] = np.nan
    obs["mhat_Q10_city"] = np.nan
    for hold_city in sorted(obs["city"].unique()):
        train_mask = obs["city"] != hold_city
        test_mask = obs["city"] == hold_city
        train = obs[train_mask]
        beta = fit_globalcalib(train["pr"].values, train["M_routine"].values)
        obs.loc[test_mask, "mhat_S4_city"] = np.clip(
            np.column_stack([np.ones(test_mask.sum()), obs.loc[test_mask, "pr"].values]) @ beta, 0, None)
        q10 = fit_q_quantile(train["pr"].values, train["M_routine"].values, 0.10)
        obs.loc[test_mask, "mhat_Q10_city"] = np.clip(obs.loc[test_mask, "pr"].values * q10, 0, None)

    # Run replay for each protocol × strategy at 5x
    STRAT_MAP = {
        ("LOTO", "S1_CapRR"): "mhat_S1",
        ("LOTO", "S4_GlobalCalib"): "mhat_S4_loto",
        ("LOTO", "Q10"): "mhat_Q10_loto",
        ("UserHoldout", "S1_CapRR"): "mhat_S1",
        ("UserHoldout", "S4_GlobalCalib"): "mhat_S4_user",
        ("UserHoldout", "Q10"): "mhat_Q10_user",
        ("CityHoldout", "S1_CapRR"): "mhat_S1",
        ("CityHoldout", "S4_GlobalCalib"): "mhat_S4_city",
        ("CityHoldout", "Q10"): "mhat_Q10_city",
    }

    for city in sorted(obs["city"].unique()):
        for task_id in sorted(obs["task_id"].unique()):
            pool = obs[(obs["city"] == city) & (obs["task_id"] == task_id)].copy()
            if pool.empty:
                continue
            p_req = task_p[task_id] * 5
            for (protocol, strat), mhat_col in STRAT_MAP.items():
                r = replay_one(pool, mhat_col, p_req, "M_routine")
                rows.append(dict(seed_id=seed_id, protocol=protocol, city=city,
                                 task_id=task_id, strategy=strat, **r))

    return pd.DataFrame(rows)


def main():
    print(f"Holdout sensitivity: {N_EXISTING} seeds × 3 protocols × 3 strategies @5x routine")
    all_rows = []
    for sid in range(N_EXISTING):
        df = run_holdout_experiment(sid)
        if df is not None:
            all_rows.append(df)
            print(f"  seed {sid}: {len(df)} rows")

    pooled = pd.concat(all_rows, ignore_index=True)
    pooled.to_csv(RESULTS / "holdout_sensitivity.csv", index=False)
    print(f"\nSaved: {RESULTS / 'holdout_sensitivity.csv'} ({len(pooled)} rows)")

    print("\n" + "=" * 90)
    print("HOLDOUT SENSITIVITY: TSR @5x routine by generalisation protocol")
    print("=" * 90)
    print(f"{'Protocol':15s} {'Strategy':20s} {'TSR%':>14s} {'OCR':>8s} {'Shortfall':>10s} {'|S|':>6s}")
    print("-" * 90)
    for protocol in ["LOTO", "UserHoldout", "CityHoldout"]:
        for strat in ["S1_CapRR", "S4_GlobalCalib", "Q10"]:
            sub = pooled[(pooled["protocol"] == protocol) & (pooled["strategy"] == strat)]
            if sub.empty:
                continue
            per_seed = sub.groupby("seed_id")["success"].mean().values * 100
            print(f"{protocol:15s} {strat:20s} {per_seed.mean():6.1f}±{per_seed.std():4.1f}   "
                  f"{sub['ocr'].mean():7.4f}  {sub['shortfall'].mean():8.1f}  {sub['n_selected'].mean():5.1f}")
        print("-" * 90)

    # City-specific breakdown for CityHoldout
    print("\n" + "=" * 80)
    print("CITY HOLDOUT DETAIL: TSR @5x by held-out city")
    print("=" * 80)
    ch = pooled[pooled["protocol"] == "CityHoldout"]
    for city in sorted(ch["city"].unique()):
        print(f"\n  Held-out city = {city}:")
        for strat in ["S1_CapRR", "S4_GlobalCalib", "Q10"]:
            sub = ch[(ch["city"] == city) & (ch["strategy"] == strat)]
            per_seed = sub.groupby("seed_id")["success"].mean().values * 100
            if len(per_seed):
                print(f"    {strat:20s}: {per_seed.mean():.1f}±{per_seed.std():.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
