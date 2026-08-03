"""P0-1 + P0-2: Corrected split-conformal LCB at N=30.

Fixes:
  1. Use one-sided conformal score: s_j = Mhat_j - M_j (not |residual|)
  2. Finite-sample quantile correction: ceil((n_cal+1)*(1-alpha))/n_cal
  3. Report empirical coverage, mean LCB width, calibration subset size
  4. Paired seed-level CI for ConformalLCB vs GlobalCalib
  5. Run at N=30 seeds (not 10)

Output: results/conformal_lcb_n30.csv + results/conformal_lcb_summary.txt
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MULTISEED_WORK = HERE / "_multiseed_work"
V1_ROOT = r"D:\项目\在研\四川\Dataset\Sichuan2024KGSimDataset"

N_SEEDS = 30
ALPHA = 0.10  # 90% one-sided coverage target


def conformal_lcb_fit(train_pr, train_y, rng_seed=42):
    """Fit split-conformal LCB with proper one-sided score.

    Returns: (beta_ols, conformal_quantile, n_fit, n_cal)
    """
    n_tr = len(train_pr)
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n_tr)
    n_fit = int(0.8 * n_tr)
    n_cal = n_tr - n_fit

    fit_idx = perm[:n_fit]
    cal_idx = perm[n_fit:]

    # Fit OLS on fit subset
    X_fit = np.column_stack([np.ones(n_fit), train_pr[fit_idx]])
    beta, _, _, _ = lstsq(X_fit, train_y[fit_idx], rcond=None)

    # One-sided conformal scores: s_j = Mhat_j - M_j (over-prediction error)
    X_cal = np.column_stack([np.ones(n_cal), train_pr[cal_idx]])
    cal_pred = X_cal @ beta
    scores = cal_pred - train_y[cal_idx]  # positive when model overestimates

    # Finite-sample corrected quantile
    # q = ceil((n_cal + 1) * (1 - alpha)) / n_cal  quantile of scores
    q_level = math.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal
    q_level = min(q_level, 1.0)  # clip to valid range
    conformal_q = np.quantile(scores, q_level, method="higher")

    return beta, conformal_q, n_fit, n_cal


def run_seed(seed_id):
    """Run ConformalLCB + GlobalCalib for one seed, return metrics."""
    data_root = MULTISEED_WORK / f"seed_{seed_id:03d}" / "dataset_v2"
    if not data_root.exists():
        return None

    obs = pd.read_csv(data_root / "observational" / "task_assessments_v2_enhanced.csv")
    tasks = pd.read_csv(os.path.join(V1_ROOT, "tasks.csv"))
    obs["M_routine"] = obs["delivery"].clip(lower=0)
    pr = (obs["nominal_capacity_kw"] * obs["response_rate"]).clip(lower=0)

    task_p = {t["task_id"]: t["required_capacity_kw"] for _, t in tasks.iterrows()}

    # Leave-one-task-out cross-fitting
    obs["mhat_gcal"] = np.nan
    obs["mhat_lcb"] = np.nan
    obs["mhat_s1"] = pr

    total_n_fit = 0
    total_n_cal = 0
    total_q = []

    for hold_task in sorted(obs["task_id"].unique()):
        train_mask = obs["task_id"] != hold_task
        test_mask = obs["task_id"] == hold_task
        train = obs[train_mask]
        pr_tr = (train["nominal_capacity_kw"] * train["response_rate"]).values
        y_tr = train["M_routine"].values
        pr_te = pr[test_mask].values

        # GlobalCalib
        X_g = np.column_stack([np.ones(len(train)), pr_tr])
        beta_g, _, _, _ = lstsq(X_g, y_tr, rcond=None)
        obs.loc[test_mask, "mhat_gcal"] = np.clip(
            np.column_stack([np.ones(test_mask.sum()), pr_te]) @ beta_g, 0, None)

        # ConformalLCB (corrected)
        beta_c, conf_q, n_fit, n_cal = conformal_lcb_fit(pr_tr, y_tr, rng_seed=seed_id * 100 + int(hold_task[1:]))
        te_pred = np.column_stack([np.ones(test_mask.sum()), pr_te]) @ beta_c
        obs.loc[test_mask, "mhat_lcb"] = np.clip(te_pred - conf_q, 0, None)

        total_n_fit += n_fit
        total_n_cal += n_cal
        total_q.append(conf_q)

    # Compute empirical coverage on test folds
    lcb = obs["mhat_lcb"].values
    actual = obs["M_routine"].values
    covered = (lcb <= actual).sum()
    coverage = covered / len(obs)
    mean_lcb_width = (actual - lcb).mean()
    mean_lcb = lcb.mean()

    # Run replay at 5x
    def replay(pool, mhat_col, p_req, oracle_col):
        ranked = pool.sort_values(mhat_col, ascending=False).reset_index(drop=True)
        if ranked.empty or ranked[mhat_col].iloc[0] <= 0:
            return dict(success=0.0, n_selected=0, ocr=0.0, shortfall=float(p_req))
        cum = ranked[mhat_col].cumsum()
        n_sel = min(int((cum < p_req).sum()) + 1, len(ranked))
        sel = ranked.iloc[:n_sel]
        p_commit = sel[mhat_col].sum()
        p_deliv = sel[oracle_col].sum()
        shortfall = max(0.0, p_req - p_deliv)
        ocr = max(0.0, p_commit - p_deliv) / p_commit if p_commit > 0 else 0.0
        return dict(success=1.0 if p_deliv >= p_req else 0.0,
                    n_selected=n_sel, ocr=ocr, shortfall=shortfall)

    results = []
    for city in sorted(obs["city"].unique()):
        for task_id in sorted(obs["task_id"].unique()):
            pool = obs[(obs["city"] == city) & (obs["task_id"] == task_id)].copy()
            if pool.empty:
                continue
            p_req = task_p[task_id] * 5
            for strat, col in [("S1_CapRR", "mhat_s1"),
                               ("S4_GlobalCalib", "mhat_gcal"),
                               ("S8_ConformalLCB", "mhat_lcb")]:
                r = replay(pool, col, p_req, "M_routine")
                results.append(dict(seed_id=seed_id, city=city, task_id=task_id,
                                    strategy=strat, **r))

    df = pd.DataFrame(results)

    # Per-seed TSR
    tsr_s1 = df[df["strategy"] == "S1_CapRR"]["success"].mean() * 100
    tsr_gcal = df[df["strategy"] == "S4_GlobalCalib"]["success"].mean() * 100
    tsr_lcb = df[df["strategy"] == "S8_ConformalLCB"]["success"].mean() * 100
    nsel_lcb = df[df["strategy"] == "S8_ConformalLCB"]["n_selected"].mean()
    nsel_gcal = df[df["strategy"] == "S4_GlobalCalib"]["n_selected"].mean()
    nsel_s1 = df[df["strategy"] == "S1_CapRR"]["n_selected"].mean()

    return dict(
        seed_id=seed_id,
        tsr_s1=tsr_s1, tsr_gcal=tsr_gcal, tsr_lcb=tsr_lcb,
        nsel_s1=nsel_s1, nsel_gcal=nsel_gcal, nsel_lcb=nsel_lcb,
        coverage=coverage, mean_lcb_width=mean_lcb_width,
        mean_lcb=mean_lcb,
        mean_n_fit=total_n_fit / 5, mean_n_cal=total_n_cal / 5,
        mean_conformal_q=np.mean(total_q),
    )


def main():
    print(f"Corrected ConformalLCB: N={N_SEEDS} seeds, alpha={ALPHA}")
    summaries = []
    for sid in range(N_SEEDS):
        s = run_seed(sid)
        if s:
            summaries.append(s)
            print(f"  seed {sid:2d}: LCB TSR={s['tsr_lcb']:.1f}%  "
                  f"GCal TSR={s['tsr_gcal']:.1f}%  "
                  f"coverage={s['coverage']:.3f}  "
                  f"nsel_lcb={s['nsel_lcb']:.1f}")

    sdf = pd.DataFrame(summaries)
    sdf.to_csv(RESULTS / "conformal_lcb_n30.csv", index=False)

    # Aggregate
    print("\n" + "=" * 80)
    print(f"CORRECTED CONFORMAL LCB RESULTS (N={N_SEEDS} seeds)")
    print("=" * 80)

    for col, label in [("tsr_s1", "S1 CapRR"),
                       ("tsr_gcal", "S4 GlobalCalib"),
                       ("tsr_lcb", "S8 ConformalLCB"),
                       ("nsel_s1", "|S| CapRR"),
                       ("nsel_gcal", "|S| GlobalCalib"),
                       ("nsel_lcb", "|S| ConformalLCB"),
                       ("coverage", "Empirical coverage"),
                       ("mean_lcb_width", "Mean LCB width (kW)"),
                       ("mean_n_cal", "Mean n_cal per fold"),
                       ("mean_conformal_q", "Mean conformal q")]:
        vals = sdf[col].dropna().values
        print(f"  {label:25s}: mean={vals.mean():8.3f}  std={vals.std():8.3f}  "
              f"[{vals.min():.3f}, {vals.max():.3f}]")

    # Paired CI: LCB - GlobalCalib
    diff = sdf["tsr_lcb"].values - sdf["tsr_gcal"].values
    print(f"\n  Paired diff (LCB - GCal):")
    print(f"    mean={diff.mean():.2f}  std={diff.std():.2f}")
    # Bootstrap CI for the mean difference
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(5000):
        idx = rng.integers(0, len(diff), size=len(diff))
        boot_means.append(diff[idx].mean())
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    print(f"    bootstrap 95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"    Significant (CI excludes 0): {ci_low > 0 or ci_high < 0}")

    # Coverage check
    print(f"\n  Coverage target: {1-ALPHA:.0%}")
    print(f"  Empirical coverage: {sdf['coverage'].mean():.3f} (should be >= {1-ALPHA:.1f})")

    # Write summary
    with open(RESULTS / "conformal_lcb_summary.txt", "w") as f:
        f.write(f"Corrected Split-Conformal LCB Results (N={N_SEEDS})\n")
        f.write(f"Alpha = {ALPHA} (target coverage = {1-ALPHA:.0%})\n\n")
        f.write(f"TSR @5x routine:\n")
        f.write(f"  S1 CapRR:        {sdf['tsr_s1'].mean():.1f} ± {sdf['tsr_s1'].std():.1f}%\n")
        f.write(f"  S4 GlobalCalib:  {sdf['tsr_gcal'].mean():.1f} ± {sdf['tsr_gcal'].std():.1f}%\n")
        f.write(f"  S8 ConformalLCB: {sdf['tsr_lcb'].mean():.1f} ± {sdf['tsr_lcb'].std():.1f}%\n")
        f.write(f"  Paired diff (LCB-GCal): {diff.mean():.1f} [{ci_low:.1f}, {ci_high:.1f}]\n\n")
        f.write(f"Empirical coverage: {sdf['coverage'].mean():.3f} (target: {1-ALPHA:.1f})\n")
        f.write(f"Mean calibration subset size: {sdf['mean_n_cal'].mean():.0f}\n")
        f.write(f"Mean conformal quantile: {sdf['mean_conformal_q'].mean():.2f} kW\n")
        f.write(f"Mean LCB width: {sdf['mean_lcb_width'].mean():.1f} kW\n")

    print(f"\nSaved: {RESULTS / 'conformal_lcb_n30.csv'}")
    print(f"Saved: {RESULTS / 'conformal_lcb_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
