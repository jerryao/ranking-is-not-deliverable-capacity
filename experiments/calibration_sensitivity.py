"""P0-3: Calibration method sensitivity + Q-quantile sweep.

Adds three new calibration strategies and sweeps Q-quantile levels:
  S7_Isotonic: monotonic nonlinear calibration (sklearn IsotonicRegression)
  S8_ConformalLCB: split-conformal lower prediction bound (90% coverage)
  Q05/Q10/Q20/Q30: conservative quantile estimation at alpha = {0.05, 0.10, 0.20, 0.30}

For each strategy, reports at 5x routine:
  TSR, OCR, Shortfall, n_selected, pool_fraction, unused_delivered_capacity

Runs on N_EXISTING seeds from the N=30 xi-enhanced datasets.
Output: results/calibration_sensitivity.csv
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
        return dict(success=0.0, n_selected=0, p_commit=0.0, p_deliv=0.0,
                    shortfall=float(p_req), ocr=0.0, unused_deliv=0.0)
    cum = ranked[mhat_col].cumsum()
    n_sel = min(int((cum < p_req).sum()) + 1, len(ranked))
    sel = ranked.iloc[:n_sel]
    p_commit = sel[mhat_col].sum()
    p_deliv = sel[oracle_col].sum()
    shortfall = max(0.0, p_req - p_deliv)
    ocr = max(0.0, p_commit - p_deliv) / p_commit if p_commit > 0 else 0.0
    unused = max(0.0, p_deliv - p_req)
    pool_frac = n_sel / len(ranked)
    return dict(success=1.0 if p_deliv >= p_req else 0.0, n_selected=n_sel,
                p_commit=p_commit, p_deliv=p_deliv, shortfall=shortfall, ocr=ocr,
                unused_deliv=unused, pool_fraction=pool_frac)


def prepare_and_run(seed_id):
    """Load seed dataset, compute all strategy predictions, run replay at 5x."""
    data_root = MULTISEED_WORK / f"seed_{seed_id:03d}" / "dataset_v2"
    if not data_root.exists():
        return None

    obs = pd.read_csv(data_root / "observational" / "task_assessments_v2_enhanced.csv")
    tasks = pd.read_csv(os.path.join(V1_ROOT, "tasks.csv"))
    obs["M_routine"] = obs["delivery"].clip(lower=0)
    pr = obs["nominal_capacity_kw"] * obs["response_rate"]

    # Strategy columns to fill
    for col in ["mhat_S1", "mhat_S4_gcal", "mhat_S7_iso", "mhat_S8_conf",
                "mhat_Q05", "mhat_Q10", "mhat_Q20", "mhat_Q30"]:
        obs[col] = np.nan

    obs["mhat_S1"] = pr.clip(lower=0)

    # Per-fold quantile storage for reporting
    fold_quantiles = {}

    for hold_task in sorted(obs["task_id"].unique()):
        train_mask = obs["task_id"] != hold_task
        test_mask = obs["task_id"] == hold_task
        train = obs[train_mask]
        y_tr = train["M_routine"].values
        pr_tr = (train["nominal_capacity_kw"] * train["response_rate"]).values
        pr_te = (obs.loc[test_mask, "nominal_capacity_kw"]
                 * obs.loc[test_mask, "response_rate"]).values

        # S4 GlobalCalib: OLS
        X_g = np.column_stack([np.ones(len(train)), pr_tr])
        beta_g, _, _, _ = lstsq(X_g, y_tr, rcond=None)
        X_te_g = np.column_stack([np.ones(test_mask.sum()), pr_te])
        obs.loc[test_mask, "mhat_S4_gcal"] = np.clip(X_te_g @ beta_g, 0, None)

        # S7 Isotonic: monotonic calibration
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0)
        iso.fit(pr_tr, y_tr)
        obs.loc[test_mask, "mhat_S7_iso"] = iso.predict(pr_te)

        # S8 Split-conformal lower bound (90% coverage)
        n_tr = len(train)
        perm = np.random.default_rng(42).permutation(n_tr)
        n_fit = int(0.8 * n_tr)
        fit_idx = perm[:n_fit]
        cal_idx = perm[n_fit:]
        X_fit = np.column_stack([np.ones(n_fit), pr_tr[fit_idx]])
        beta_fit, _, _, _ = lstsq(X_fit, y_tr[fit_idx], rcond=None)
        cal_pred = np.column_stack([np.ones(len(cal_idx)), pr_tr[cal_idx]]) @ beta_fit
        residuals = y_tr[cal_idx] - cal_pred
        # Lower bound: pred_test - q_{0.90}(|residuals|) for 90% one-sided coverage
        conformal_margin = np.quantile(np.abs(residuals), 0.90)
        te_pred = np.column_stack([np.ones(test_mask.sum()), pr_te]) @ beta_fit
        obs.loc[test_mask, "mhat_S8_conf"] = np.clip(te_pred - conformal_margin, 0, None)

        # Q-quantile sensitivity
        ratio = y_tr / np.clip(pr_tr, 1e-6, None)
        for alpha, col in [(0.05, "mhat_Q05"), (0.10, "mhat_Q10"),
                           (0.20, "mhat_Q20"), (0.30, "mhat_Q30")]:
            q = np.quantile(ratio, alpha)
            obs.loc[test_mask, col] = np.clip(pr_te * q, 0, None)
            fold_quantiles.setdefault(col, {})[hold_task] = q

    task_p = {t["task_id"]: t["required_capacity_kw"] for _, t in tasks.iterrows()}
    rows = []
    STRATEGIES = {
        "S1_CapRR": "mhat_S1",
        "S4_GlobalCalib": "mhat_S4_gcal",
        "S7_Isotonic": "mhat_S7_iso",
        "S8_ConformalLCB": "mhat_S8_conf",
        "Q05": "mhat_Q05",
        "Q10": "mhat_Q10",
        "Q20": "mhat_Q20",
        "Q30": "mhat_Q30",
    }

    for city in sorted(obs["city"].unique()):
        for task_id in sorted(obs["task_id"].unique()):
            pool = obs[(obs["city"] == city) & (obs["task_id"] == task_id)].copy()
            if pool.empty:
                continue
            p_req = task_p[task_id] * 5
            for strat, mhat_col in STRATEGIES.items():
                r = replay_one(pool, mhat_col, p_req, "M_routine")
                rows.append(dict(seed_id=seed_id, city=city, task_id=task_id,
                                 strategy=strat, **r))

    return pd.DataFrame(rows), fold_quantiles


def main():
    print(f"Calibration sensitivity: {N_EXISTING} seeds × 8 strategies @5x routine")
    all_rows = []
    all_fold_q = []
    for sid in range(N_EXISTING):
        result = prepare_and_run(sid)
        if result is None:
            continue
        df, fold_q = result
        all_rows.append(df)
        for col, task_q in fold_q.items():
            for task, q in task_q.items():
                all_fold_q.append(dict(seed_id=sid, strategy=col, task_id=task, quantile=q))
        print(f"  seed {sid}: {len(df)} rows")

    pooled = pd.concat(all_rows, ignore_index=True)
    pooled.to_csv(RESULTS / "calibration_sensitivity.csv", index=False)
    print(f"\nSaved: {RESULTS / 'calibration_sensitivity.csv'} ({len(pooled)} rows)")

    if all_fold_q:
        qdf = pd.DataFrame(all_fold_q)
        qdf.to_csv(RESULTS / "q_fold_quantiles.csv", index=False)

    # Summary table
    print("\n" + "=" * 120)
    print("CALIBRATION SENSITIVITY: TSR @5x routine (mean ± std across seeds)")
    print("=" * 120)
    print(f"{'Strategy':20s} {'TSR%':>14s} {'OCR':>10s} {'Shortfall':>10s} "
          f"{'|S|':>6s} {'pool_frac':>10s} {'unused_kW':>10s}")
    print("-" * 120)
    for strat in ["S1_CapRR", "S4_GlobalCalib", "S7_Isotonic", "S8_ConformalLCB",
                  "Q05", "Q10", "Q20", "Q30"]:
        sub = pooled[pooled["strategy"] == strat]
        if sub.empty:
            continue
        per_seed = sub.groupby("seed_id")["success"].mean().values * 100
        print(f"{strat:20s} {per_seed.mean():6.1f}±{per_seed.std():4.1f}   "
              f"{sub['ocr'].mean():8.4f}  {sub['shortfall'].mean():8.1f}  "
              f"{sub['n_selected'].mean():5.1f}  {sub['pool_fraction'].mean():8.4f}   "
              f"{sub['unused_deliv'].mean():8.1f}")

    # Q-quantile per-fold values
    if all_fold_q:
        print("\n" + "=" * 80)
        print("Q-QUANTILE PER-FOLD VALUES (sensitivity to alpha)")
        print("=" * 80)
        qdf = pd.DataFrame(all_fold_q)
        for col in ["mhat_Q05", "mhat_Q10", "mhat_Q20", "mhat_Q30"]:
            sub = qdf[qdf["strategy"] == col]
            if sub.empty:
                continue
            vals = sub["quantile"].values
            print(f"  {col:12s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
