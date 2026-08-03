"""P0-2: Stress scenario sweep.

Uses the analytical paired-arm relationship to recompute M_stress for
arbitrary stress configurations WITHOUT regenerating data:

    Y(new) = Y(0) + C * (exp(s_new) - 1)
    where C = [Y(full_stress) - Y(0)] / [exp(s_full) - 1]

This is exact when no arm hits the zero floor (verified: min delivery > 18 kW).

Sweeps:
  1. K_active: {1, 2, 3, 5} violation classes (severity=1.0, scope=0.5)
  2. Severity: {0.25, 0.50, 0.75, 1.00} (K=5, scope=0.5)
  3. Scope: {0.25, 0.50, 0.75} (K=5, severity=1.0)
  4. Individual classes: 5 single-class activations

Runs on N_EXISTING seeds from the N=30 ξ-enhanced datasets.
Output: results/stress_sweep_summary.csv
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import lstsq

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MULTISEED_WORK = HERE / "_multiseed_work"
V1_ROOT = r"D:\项目\在研\四川\Dataset\Sichuan2024KGSimDataset"

N_EXISTING = 10  # Use first 10 of the 30 ξ-enhanced seeds

WEIGHTS = {"physical": -0.55, "mutex": -0.18, "comfort": -0.08,
           "hierarchy": -0.04, "contract": 0.0}
CLASS_ORDER = ["physical", "mutex", "comfort", "hierarchy", "contract"]
S_FULL = sum(WEIGHTS.values()) * 1.0 * 0.5  # = -0.425
EXP_S_FULL_MINUS_1 = math.exp(S_FULL) - 1  # = -0.346


def build_stress_configs():
    configs = []
    # 1. K_active sweep
    for K in [1, 2, 3, 5]:
        active = CLASS_ORDER[:K]
        s = sum(WEIGHTS[c] for c in active) * 1.0 * 0.5
        configs.append({"name": f"K{K}_sev1.0_sco0.5", "s_new": s,
                        "K_active": K, "severity": 1.0, "scope": 0.5})
    # 2. Severity sweep (K=5)
    for sev in [0.25, 0.50, 0.75, 1.00]:
        s = sum(WEIGHTS.values()) * sev * 0.5
        configs.append({"name": f"K5_sev{sev}_sco0.5", "s_new": s,
                        "K_active": 5, "severity": sev, "scope": 0.5})
    # 3. Scope sweep (K=5, sev=1.0)
    for sco in [0.25, 0.50, 0.75]:
        s = sum(WEIGHTS.values()) * 1.0 * sco
        configs.append({"name": f"K5_sev1.0_sco{sco}", "s_new": s,
                        "K_active": 5, "severity": 1.0, "scope": sco})
    # 4. Individual class (sev=1.0, sco=0.5)
    for cls in CLASS_ORDER:
        s = WEIGHTS[cls] * 1.0 * 0.5
        configs.append({"name": f"cls_{cls}", "s_new": s,
                        "K_active": 1, "severity": 1.0, "scope": 0.5,
                        "active_class": cls})
    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        if c["name"] not in seen:
            seen.add(c["name"])
            unique.append(c)
    return unique


def replay_one(pool, mhat_col, p_req, oracle_col):
    ranked = pool.sort_values(mhat_col, ascending=False).reset_index(drop=True)
    if ranked.empty or ranked[mhat_col].iloc[0] <= 0:
        return dict(success=0.0, n_selected=0, p_commit=0.0, p_deliv=0.0,
                    shortfall=float(p_req), ocr=0.0)
    cum = ranked[mhat_col].cumsum()
    n_sel = min(int((cum < p_req).sum()) + 1, len(ranked))
    sel = ranked.iloc[:n_sel]
    p_commit = sel[mhat_col].sum()
    p_deliv = sel[oracle_col].sum()
    shortfall = max(0.0, p_req - p_deliv)
    ocr = max(0.0, p_commit - p_deliv) / p_commit if p_commit > 0 else 0.0
    return dict(success=1.0 if p_deliv >= p_req else 0.0, n_selected=n_sel,
                p_commit=p_commit, p_deliv=p_deliv, shortfall=shortfall, ocr=ocr)


def run_stress_config(obs_merged, tasks, stress_cfg):
    """Recompute M_stress for given config and run cross-regime replay."""
    s_new = stress_cfg["s_new"]
    # Y_delivery_new = Y0 + C * (exp(s_new) - 1)
    # C = (Y1 - Y0) / (exp(s_full) - 1)
    C = (obs_merged["Y_delivery_1"] - obs_merged["Y_delivery_0"]) / EXP_S_FULL_MINUS_1
    M_stress_new = (obs_merged["Y_delivery_0"] + C * (math.exp(s_new) - 1)).clip(lower=0)

    obs_work = obs_merged.copy()
    obs_work["M_stress_new"] = M_stress_new

    # Strategies already computed in obs_merged (mhat_S1, mhat_S4, etc.)
    STRATEGIES = {
        "S0a_RoutineOracle": "mhat_S0a",
        "S0b_StressOracle_new": "M_stress_new",  # oracle for THIS stress config
        "S1_CapRR": "mhat_S1",
        "S4_GlobalCalib": "mhat_S4",
        "S6_Q10": "mhat_S6",
    }

    task_p = {t["task_id"]: t["required_capacity_kw"] for _, t in tasks.iterrows()}
    rows = []
    for city in sorted(obs_work["city"].unique()):
        for task_id in sorted(obs_work["task_id"].unique()):
            pool = obs_work[(obs_work["city"] == city) & (obs_work["task_id"] == task_id)].copy()
            if pool.empty:
                continue
            p_base = task_p[task_id]
            for mult in [5]:  # Focus on 5x (the headline)
                p_req = p_base * mult
                for strat, mhat_col in STRATEGIES.items():
                    r = replay_one(pool, mhat_col, p_req, "M_stress_new")
                    rows.append(dict(
                        stress_config=stress_cfg["name"],
                        s_new=s_new, K_active=stress_cfg["K_active"],
                        severity=stress_cfg["severity"], scope=stress_cfg["scope"],
                        city=city, task_id=task_id, p_req_level=f"{mult}x",
                        strategy=strat, **r))
    return pd.DataFrame(rows)


def prepare_seed(seed_id):
    """Load and prepare a single seed's dataset."""
    data_root = MULTISEED_WORK / f"seed_{seed_id:03d}" / "dataset_v2"
    if not data_root.exists():
        return None

    obs = pd.read_csv(data_root / "observational" / "task_assessments_v2_enhanced.csv")
    ip = pd.read_csv(data_root / "intervention_pairs" / "intervention_pairs_tau.csv")
    tasks = pd.read_csv(os.path.join(V1_ROOT, "tasks.csv"))

    obs["M_routine"] = obs["delivery"].clip(lower=0)
    obs = obs.merge(ip[["city", "user_id", "task_id", "event_id", "Y_delivery_0", "Y_delivery_1"]],
                    on=["city", "user_id", "task_id", "event_id"], how="left")

    # Compute strategy predictions (same as e3_replay_engine_v3)
    feat_cols = ["nominal_capacity_kw", "response_rate", "availability_rate", "event_intensity"]
    obs["mhat_S0a"] = obs["M_routine"]
    obs["mhat_S1"] = (obs["nominal_capacity_kw"] * obs["response_rate"]).clip(lower=0)

    # Leave-one-task-out for S4 and S6
    obs["mhat_S4"] = np.nan
    obs["mhat_S6"] = np.nan
    for hold_task in sorted(obs["task_id"].unique()):
        train_mask = obs["task_id"] != hold_task
        test_mask = obs["task_id"] == hold_task
        train = obs[train_mask]
        pr_tr = train["nominal_capacity_kw"] * train["response_rate"]
        X_g = np.column_stack([np.ones(len(train)), pr_tr.values])
        beta_g, _, _, _ = lstsq(X_g, train["M_routine"].values, rcond=None)
        pr_te = obs.loc[test_mask, "nominal_capacity_kw"] * obs.loc[test_mask, "response_rate"]
        X_te = np.column_stack([np.ones(test_mask.sum()), pr_te.values])
        obs.loc[test_mask, "mhat_S4"] = np.clip(X_te @ beta_g, 0, None)
        # Q10
        ratio = train["M_routine"].values / (pr_tr.values)
        q10 = np.quantile(ratio, 0.10)
        obs.loc[test_mask, "mhat_S6"] = np.clip(pr_te.values * q10, 0, None)

    return obs, tasks


def main():
    configs = build_stress_configs()
    print(f"Stress sweep: {len(configs)} configs × {N_EXISTING} seeds")
    print(f"S_full = {S_FULL:.4f}, exp(S_full)-1 = {EXP_S_FULL_MINUS_1:.4f}")

    all_rows = []
    for sid in range(N_EXISTING):
        result = prepare_seed(sid)
        if result is None:
            print(f"  seed {sid}: data not found, skipping")
            continue
        obs, tasks = result
        print(f"  seed {sid}: loaded {len(obs)} rows")
        for cfg in configs:
            df = run_stress_config(obs, tasks, cfg)
            df["seed_id"] = sid
            all_rows.append(df)

    pooled = pd.concat(all_rows, ignore_index=True)
    out_path = RESULTS / "stress_sweep_summary.csv"
    pooled.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(pooled)} rows)")

    # Aggregate: TSR by stress config × strategy
    print("\n" + "=" * 100)
    print("STRESS SWEEP: Cross-regime TSR @5x by stress configuration")
    print("(Lower TSR = more stress-induced delivery loss)")
    print("=" * 100)
    print(f"{'stress_config':30s} {'s_new':>7s} {'K':>3s} {'sev':>5s} {'sco':>5s} | "
          f"{'S0a_Routine':>12s} {'S1_CapRR':>12s} {'S4_GlobalCalib':>12s} {'S6_Q10':>12s}")
    print("-" * 100)

    for cfg in configs:
        sub = pooled[pooled["stress_config"] == cfg["name"]]
        if sub.empty:
            continue
        vals = {}
        for strat in ["S0a_RoutineOracle", "S1_CapRR", "S4_GlobalCalib", "S6_Q10"]:
            ss = sub[sub["strategy"] == strat]
            if len(ss):
                tsr_vals = ss.groupby("seed_id")["success"].mean().values * 100
                vals[strat] = f"{tsr_vals.mean():.1f}±{tsr_vals.std():.1f}"
            else:
                vals[strat] = "N/A"
        active_cls = cfg.get("active_class", "")
        name_display = cfg["name"] if not active_cls else f"cls={active_cls}"
        print(f"{name_display:30s} {cfg['s_new']:7.4f} {cfg['K_active']:3d} "
              f"{cfg['severity']:5.2f} {cfg['scope']:5.2f} | "
              f"{vals['S0a_RoutineOracle']:>12s} {vals['S1_CapRR']:>12s} "
              f"{vals['S4_GlobalCalib']:>12s} {vals['S6_Q10']:>12s}")

    # Stress-response curve: TSR vs s_new
    print("\n" + "=" * 80)
    print("STRESS-RESPONSE CURVE: TSR vs violation dose s_new")
    print("=" * 80)
    curve = pooled.groupby(["stress_config", "s_new"]).agg(
        tsr_s0a=("success", lambda x: x[pooled.loc[x.index, "strategy"] == "S0a_RoutineOracle"].mean() * 100),
        tsr_s1=("success", lambda x: x[pooled.loc[x.index, "strategy"] == "S1_CapRR"].mean() * 100),
        tsr_s4=("success", lambda x: x[pooled.loc[x.index, "strategy"] == "S4_GlobalCalib"].mean() * 100),
    ).reset_index()
    curve = curve.sort_values("s_new")
    for _, row in curve.iterrows():
        print(f"  s_new={row['s_new']:7.4f}  S0a={row['tsr_s0a']:5.1f}%  S1={row['tsr_s1']:5.1f}%  S4={row['tsr_s4']:5.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
