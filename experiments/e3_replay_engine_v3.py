"""E3 Operational Replay Engine (v3) — Item 3 + Item 7 fix.

Key changes vs v2:
  * M_routine = observational.delivery  (v2 toolkit computes this from
    base=nominal_capacity_kw * rr * exp(s_obs) * event_factor + noise)
  * M_stress = intervention_pairs_tau.Y_delivery_1  (oracle_simulator
    computes do(V=full stress), which by definition matches Paper 2 §4.3)
  * No dependency on paper1 dual_oracle_results.csv.
  * Data path is parameterised via --data-v2 to support multi-seed runs.
  * Output path is parameterised via --output to support multi-seed runs.

Item 3 (Paper 2 review): response_rate is now applied exactly once at the
generator level. Previous v2 engine used v1's `actual_delivered_kw`
(generated with a different base) merged with paper1's `tau_dual`/
`tau_stress_formula` (computed with `pred_reliable` as base), creating an
inconsistent mix. v3 reads only v2-toolkit outputs that share the same
base = nominal_capacity_kw.

Item 7 (Paper 2 review): because all inputs now flow through v2 toolkit,
changing master_seed in config.json produces independent benchmark
realisations, and TSR/OCR/shortfall will have a real distribution.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import lstsq


def run_replay(data_v2: str, output_csv: str, v1_root: str | None = None) -> dict:
    """Run the full replay on a v2 dataset; return summary dict."""
    BASE = Path(__file__).resolve().parent
    if v1_root is None:
        # Default to the canonical v1 location
        v1_root = r"D:\项目\在研\四川\Dataset\Sichuan2024KGSimDataset"

    WEIGHTS = {"physical": -0.55, "mutex": -0.18, "comfort": -0.08,
               "hierarchy": -0.04, "contract": 0.0}
    P_REQ_MULT = [1, 2, 5, 10]

    print(f"\n[load] data_v2 = {data_v2}")
    obs = pd.read_csv(os.path.join(data_v2, "observational", "task_assessments_v2_enhanced.csv"))
    tasks = pd.read_csv(os.path.join(v1_root, "tasks.csv"))
    ip_tau = pd.read_csv(os.path.join(data_v2, "intervention_pairs", "intervention_pairs_tau.csv"))

    # ---- M_routine: v2-toolkit delivery (uses nominal_capacity_kw base) ----
    if "delivery" not in obs.columns:
        raise KeyError(
            "Column 'delivery' not found in observational data. "
            "Ensure v2 toolkit was run with the Item 3 fix."
        )
    obs["M_routine"] = obs["delivery"].clip(lower=0)

    # ---- M_stress: from intervention_pairs (do(V=full stress)) ----
    # Merge Y_delivery_1 (= stress delivery) onto obs by (city, user_id, task_id, event_id)
    obs = obs.merge(
        ip_tau[["city", "user_id", "task_id", "event_id", "Y_delivery_1"]],
        on=["city", "user_id", "task_id", "event_id"], how="left"
    )
    obs["M_stress"] = obs["Y_delivery_1"].clip(lower=0)

    n_nan_stress = obs["M_stress"].isna().sum()
    if n_nan_stress > 0:
        print(f"  WARNING: {n_nan_stress} obs rows missing stress delivery after merge")

    # ---- Item 25 zero-floor diagnostic ----
    n_zero_routine = (obs["M_routine"] == 0).sum()
    n_zero_stress = (obs["M_stress"] == 0).sum()
    print(f"  M_routine: mean={obs['M_routine'].mean():.2f}  zero_count={n_zero_routine}/{len(obs)}")
    print(f"  M_stress:  mean={obs['M_stress'].mean():.2f}   zero_count={n_zero_stress}/{len(obs)}")

    # ---- Local ρ computation (delivery-ranking) ----
    s_obs = np.zeros(len(obs))
    for k, w in WEIGHTS.items():
        s_obs += w * obs[f"V_{k}_flag"].values * obs[f"V_{k}_severity"].values * obs[f"V_{k}_scope"].values
    obs["s_obs"] = s_obs

    feat_cols = ["nominal_capacity_kw", "response_rate", "availability_rate", "event_intensity"]
    X_rho = obs[feat_cols].values
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(X_rho, obs["M_routine"].values)
    rho_lin4 = stats.spearmanr(lr.predict(X_rho), obs["M_routine"].values)[0]
    rho_caprr = stats.spearmanr(obs["nominal_capacity_kw"] * obs["response_rate"], obs["M_routine"])[0]
    print(f"  rho(Delivery-Linear4) = {rho_lin4:.3f}   rho(Cap*RR) = {rho_caprr:.3f}")

    # ---- Leave-one-task-out cross-fitting ----
    obs["mhat_S3_lin4"] = np.nan
    obs["mhat_S4_gcal"] = np.nan
    obs["mhat_S5_ccal"] = np.nan
    obs["mhat_S6_q10"] = np.nan

    TASK_IDS = sorted(obs["task_id"].unique())
    for hold_task in TASK_IDS:
        train_mask = obs["task_id"] != hold_task
        test_mask = obs["task_id"] == hold_task
        train = obs[train_mask]

        # S3: Linear4 — OLS predicting M_routine from operational features
        X_tr = np.column_stack([np.ones(len(train))] + [train[c].values for c in feat_cols])
        y_tr = train["M_routine"].values
        beta, _, _, _ = lstsq(X_tr, y_tr, rcond=None)
        X_te = np.column_stack([np.ones(test_mask.sum())] + [obs.loc[test_mask, c].values for c in feat_cols])
        obs.loc[test_mask, "mhat_S3_lin4"] = X_te @ beta

        # S4: Global linear calibration — predict M_routine from nominal*rr
        # (Item 3 fix: use nominal*rr as the deployable-capacity proxy, not pred_reliable)
        pr_train = train["nominal_capacity_kw"] * train["response_rate"]
        X_g = np.column_stack([np.ones(len(train)), pr_train.values])
        beta_g, _, _, _ = lstsq(X_g, y_tr, rcond=None)
        pr_test = obs.loc[test_mask, "nominal_capacity_kw"] * obs.loc[test_mask, "response_rate"]
        X_te_g = np.column_stack([np.ones(test_mask.sum()), pr_test.values])
        obs.loc[test_mask, "mhat_S4_gcal"] = X_te_g @ beta_g

        # S5: City-specific calibration
        for c in sorted(obs["city"].unique()):
            tr_c = train[train["city"] == c]
            te_c_mask = test_mask & (obs["city"] == c)
            if len(tr_c) > 5 and te_c_mask.sum() > 0:
                pr_tr_c = tr_c["nominal_capacity_kw"] * tr_c["response_rate"]
                X_c = np.column_stack([np.ones(len(tr_c)), pr_tr_c.values])
                beta_c, _, _, _ = lstsq(X_c, tr_c["M_routine"].values, rcond=None)
                pr_te_c = obs.loc[te_c_mask, "nominal_capacity_kw"] * obs.loc[te_c_mask, "response_rate"]
                X_te_c = np.column_stack([np.ones(te_c_mask.sum()), pr_te_c.values])
                obs.loc[te_c_mask, "mhat_S5_ccal"] = X_te_c @ beta_c

        # S6: Conservative Q10
        ratio = train["M_routine"].values / (train["nominal_capacity_kw"] * train["response_rate"]).values
        q10 = np.quantile(ratio, 0.10)
        pr_te = obs.loc[test_mask, "nominal_capacity_kw"] * obs.loc[test_mask, "response_rate"]
        obs.loc[test_mask, "mhat_S6_q10"] = pr_te.values * q10

    # Assign non-fitted strategies
    obs["mhat_S0a_routine_oracle"] = obs["M_routine"]
    obs["mhat_S0b_stress_oracle"] = obs["M_stress"]
    obs["mhat_S1_caprr"] = obs["nominal_capacity_kw"] * obs["response_rate"]
    # S2 Platform: keep as v1 platform-assessment rating (pred_reliable) per Paper 2 §5.1.
    # This is a simulator-side proxy, NOT a claim about the deployed Sichuan platform (Item 32).
    obs["mhat_S2_platform"] = obs["pred_reliable_deliverable_capacity_kw"]

    # Clip all predictions to non-negative (Item 10 already in v2; retained)
    for col in [c for c in obs.columns if c.startswith("mhat_")]:
        obs[col] = obs[col].clip(lower=0)

    STRATEGIES = {
        "S0a_RoutineOracle": "mhat_S0a_routine_oracle",
        "S0b_StressOracle": "mhat_S0b_stress_oracle",
        "S1_CapRR": "mhat_S1_caprr",
        "S2_Platform": "mhat_S2_platform",
        "S3_Lin4": "mhat_S3_lin4",
        "S4_GlobalCalib": "mhat_S4_gcal",
        "S5_CityCalib": "mhat_S5_ccal",
        "S6_Q10": "mhat_S6_q10",
    }

    def replay(pool, mhat_col, p_req, oracle_col):
        ranked = pool.sort_values(mhat_col, ascending=False).reset_index(drop=True)
        if ranked.empty or ranked[mhat_col].iloc[0] <= 0:
            return dict(n_selected=0, p_commit=0.0, p_deliv=0.0,
                        shortfall=float(p_req), ocr=0.0, success=0.0)
        cum = ranked[mhat_col].cumsum()
        n_sel = min(int((cum < p_req).sum()) + 1, len(ranked))
        sel = ranked.iloc[:n_sel]
        p_commit = sel[mhat_col].sum()
        p_deliv = sel[oracle_col].sum()
        shortfall = max(0.0, p_req - p_deliv)
        ocr = max(0.0, p_commit - p_deliv) / p_commit if p_commit > 0 else 0.0
        success = 1.0 if p_deliv >= p_req else 0.0
        return dict(n_selected=n_sel, p_commit=round(p_commit, 1),
                    p_deliv=round(p_deliv, 1), shortfall=round(shortfall, 1),
                    ocr=round(ocr, 4), success=success)

    CITIES = sorted(obs["city"].unique())
    task_p = {t["task_id"]: t["required_capacity_kw"] for _, t in tasks.iterrows()}

    rows = []
    for city in CITIES:
        for task_id in TASK_IDS:
            pool = obs[(obs["city"] == city) & (obs["task_id"] == task_id)].copy()
            if pool.empty:
                continue
            p_base = task_p[task_id]
            pool_stress_total = pool["M_stress"].sum()
            pool_routine_total = pool["M_routine"].sum()

            for mult in P_REQ_MULT:
                p_req = p_base * mult
                feas_stress = pool_stress_total >= p_req
                feas_routine = pool_routine_total >= p_req

                for strat, mhat_col in STRATEGIES.items():
                    r1 = replay(pool, mhat_col, p_req, "M_routine")
                    rows.append(dict(city=city, task_id=task_id, p_req_level=f"{mult}x",
                                     p_req=p_req, strategy=strat, regime="routine",
                                     eval_outcome="M_routine", pool_size=len(pool),
                                     pool_feasible=feas_routine, **r1))
                    r2 = replay(pool, mhat_col, p_req, "M_stress")
                    rows.append(dict(city=city, task_id=task_id, p_req_level=f"{mult}x",
                                     p_req=p_req, strategy=strat, regime="cross_regime",
                                     eval_outcome="M_stress", pool_size=len(pool),
                                     pool_feasible=feas_stress, **r2))

    results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv} ({len(results)} rows)")

    # ---- Summary: routine + cross-regime TSR ----
    summary_out = {
        "rho_lin4": float(rho_lin4),
        "rho_caprr": float(rho_caprr),
        "n_zero_routine": int(n_zero_routine),
        "n_zero_stress": int(n_zero_stress),
        "M_routine_mean": float(obs["M_routine"].mean()),
        "M_stress_mean": float(obs["M_stress"].mean()),
        "tsr": {},
    }
    for regime in ["routine", "cross_regime"]:
        for level in ["1x", "2x", "5x", "10x"]:
            sub = results[(results["regime"] == regime) & (results["p_req_level"] == level)]
            tsr = sub.groupby("strategy")["success"].mean() * 100
            for s in STRATEGIES:
                key = f"{regime}_{level}_{s}"
                summary_out["tsr"][key] = float(tsr.get(s, float("nan")))

    return summary_out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-v2", required=True, help="Path to v2 dataset root")
    p.add_argument("--output", required=True, help="Path to output CSV")
    p.add_argument("--v1-root", default=r"D:\项目\在研\四川\Dataset\Sichuan2024KGSimDataset",
                   help="Path to v1 dataset root (for tasks.csv)")
    args = p.parse_args(argv)
    summary = run_replay(args.data_v2, args.output, args.v1_root)
    print("\n=== TSR summary ===")
    for k, v in summary["tsr"].items():
        print(f"  {k}: {v:.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
