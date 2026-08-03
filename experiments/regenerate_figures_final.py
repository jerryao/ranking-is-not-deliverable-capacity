"""Regenerate Figures 3, 4, 5 from frozen N=30 data.

Figure 3: Dual-oracle scatter (routine vs stress |tau|), with mean and
          median ratio annotations distinguished.
Figure 4: Routine-regime TSR vs requirement multiplier (7 strategies).
Figure 5: Cross-regime TSR vs requirement multiplier (7 strategies).

All figures use the pooled-cell Wilson 95% CI as error bands.
Data source: results/multiseed_tsr_with_ci.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES_DIR = HERE.parent / "manuscript" / "figures"
MULTISEED_WORK = HERE / "_multiseed_work"
V1_ROOT = r"D:\项目\在研\四川\Dataset\Sichuan2024KGSimDataset"

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ============================================================
# Figure 3: Dual-oracle scatter
# ============================================================
def generate_figure_3():
    """Scatter of |tau_regime| vs |tau_stress| with ratio annotations."""
    print("[Figure 3] Dual-oracle scatter...")

    # Load seed 0 paired data
    data_root = MULTISEED_WORK / "seed_000" / "dataset_v2"
    if not data_root.exists():
        print("  SKIP: seed_000 data not found")
        return

    ip = pd.read_csv(data_root / "intervention_pairs" / "intervention_pairs_tau.csv")
    obs = pd.read_csv(data_root / "observational" / "task_assessments_v2_enhanced.csv")

    # Compute tau_regime and tau_stress from the paired arms
    # Y_delivery_0 = routine (with violations), Y_delivery_1 = full stress
    tau_stress = ip["true_tau_delivery"].values  # Y1 - Y0
    # For tau_regime: need the observational violation dose
    WEIGHTS = {"physical": -0.55, "mutex": -0.18, "comfort": -0.08,
               "hierarchy": -0.04, "contract": 0.0}
    s_obs = np.zeros(len(obs))
    for k, w in WEIGHTS.items():
        s_obs += w * obs[f"V_{k}_flag"].values * obs[f"V_{k}_severity"].values * obs[f"V_{k}_scope"].values

    # tau_regime = nominal * rr * (exp(s_obs) - 1) * evf * xi
    # But we don't have xi directly. Instead, use the relationship:
    # tau_stress = C * (exp(s_stress) - 1) where C = nominal*rr*evf*xi
    # M_routine = Y(0) + C * (exp(s_obs) - 1) = Y(0) + tau_stress * (exp(s_obs)-1)/(exp(s_stress)-1)
    s_stress = sum(WEIGHTS.values()) * 1.0 * 0.5  # -0.425
    C = tau_stress / (np.exp(s_stress) - 1)
    evf = 1.0 + 0.15 * (obs["event_intensity"].values - 0.3)
    tau_regime = C * (np.exp(s_obs) - 1)

    treated = obs["V_any_flag"].values == 1
    tau_regime_t = np.abs(tau_regime[treated])
    tau_stress_t = np.abs(tau_stress[treated])

    # Compute statistics
    mean_ratio = np.mean(tau_stress_t) / np.mean(tau_regime_t) if np.mean(tau_regime_t) > 0 else float("inf")
    paired_ratios = tau_stress_t / np.clip(tau_regime_t, 0.01, None)
    median_paired_ratio = np.median(paired_ratios)
    rho = stats.spearmanr(tau_regime_t, tau_stress_t)[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): scatter
    ax1.scatter(tau_regime_t, tau_stress_t, alpha=0.3, s=10, c="steelblue")
    max_val = max(tau_regime_t.max(), tau_stress_t.max())
    ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=1)
    ax1.set_xlabel(r"$|\tau^{\mathrm{regime}}|$ (kW)")
    ax1.set_ylabel(r"$|\tau^{\mathrm{stress}}|$ (kW)")
    ax1.set_title(f"(a) Regime vs stress effects (Spearman " + r"$\rho$" + f" = {rho:.3f})")

    # Annotate ratios
    ax1.text(0.05, 0.95, f"Mean ratio: {mean_ratio:.1f}" + r"$\times$" + "\n"
             f"Median paired ratio: {median_paired_ratio:.1f}" + r"$\times$",
             transform=ax1.transAxes, fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel (b): boxplot
    bp_data = [tau_regime_t, tau_stress_t]
    bp = ax2.boxplot(bp_data, labels=["Routine\nregime", "Stress\nregime"],
                     patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightsalmon")
    ax2.set_ylabel("Absolute effect (kW)")
    ax2.set_title(f"(b) Effect magnitude shift\n"
                  f"Mean ratio " + r"$\approx$" + f" {mean_ratio:.0f}" + r"$\times$" +
                  f"; Median paired ratio " + r"$\approx$" + f" {median_paired_ratio:.0f}" + r"$\times$")

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig3_dual_oracle.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Figures 4 and 5: TSR vs requirement multiplier
# ============================================================
def generate_figures_4_5():
    """Routine and cross-regime TSR curves with seed-clustered bootstrap CI bands."""
    print("[Figures 4/5] TSR vs requirement multiplier (bootstrap CI)...")

    # Load raw replay data for bootstrap computation
    replay_dfs = [pd.read_csv(f) for f in sorted((HERE / "results").glob("replay_seed_*.csv"))]
    pooled = pd.concat(replay_dfs, ignore_index=True)

    rng = np.random.default_rng(42)

    STRATEGY_STYLE = {
        "S0a_RoutineOracle":  {"color": "black",   "marker": "s", "ls": "--", "label": "Routine Oracle"},
        "S1_CapRR":           {"color": "red",     "marker": "o", "ls": "-",  "label": "Cap" + r"$\times$" + "RR"},
        "S2_Platform":        {"color": "orange",  "marker": "D", "ls": "-",  "label": "Platform Proxy"},
        "S3_Lin4":            {"color": "green",   "marker": "^", "ls": "-",  "label": "Linear4"},
        "S4_GlobalCalib":     {"color": "blue",    "marker": "v", "ls": "-",  "label": "GlobalCalib"},
        "S6_Q10":             {"color": "purple",  "marker": "P", "ls": "-",  "label": "Q10"},
    }

    LEVELS = ["1x", "2x", "5x", "10x"]
    LEVEL_POS = [1, 2, 5, 10]

    for regime, fig_num, title_suffix in [("routine", 4, "Routine regime"),
                                            ("cross_regime", 5, "Cross-regime (stress delivery)")]:
        fig, ax = plt.subplots(figsize=(8, 5.5))

        for strat, style in STRATEGY_STYLE.items():
            tsr_vals = []
            ci_low = []
            ci_high = []
            for level in LEVELS:
                sub = pooled[(pooled["regime"] == regime)
                             & (pooled["p_req_level"] == level)
                             & (pooled["strategy"] == strat)]
                if sub.empty:
                    tsr_vals.append(np.nan); ci_low.append(np.nan); ci_high.append(np.nan)
                    continue
                per_seed = sub.groupby("seed_id")["success"].mean().values
                n = len(per_seed)
                boot = []
                for _ in range(2000):
                    idx = rng.integers(0, n, size=n)
                    boot.append(per_seed[idx].mean())
                boot = np.array(boot) * 100
                tsr_vals.append(per_seed.mean() * 100)
                ci_low.append(np.percentile(boot, 2.5))
                ci_high.append(np.percentile(boot, 97.5))

            tsr_vals = np.array(tsr_vals)
            ci_low = np.array(ci_low)
            ci_high = np.array(ci_high)

            ax.plot(LEVEL_POS, tsr_vals, color=style["color"], marker=style["marker"],
                    linestyle=style["ls"], linewidth=1.5, markersize=6, label=style["label"])
            ax.fill_between(LEVEL_POS, ci_low, ci_high, alpha=0.12, color=style["color"])

        ax.set_xscale("log", base=10)
        ax.set_xticks(LEVEL_POS)
        ax.set_xticklabels([r"$1\times$", r"$2\times$", r"$5\times$", r"$10\times$"])
        ax.set_xlabel("Requirement multiplier")
        ax.set_ylabel("Task success rate (%)")
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.legend(loc="lower left", framealpha=0.9)
        ax.set_title(f"Figure {fig_num}: {title_suffix}\n"
                     r"($N=30$ seeds, $n=450$ cells; shaded bands = Wilson 95% CI)")
        ax.grid(True, alpha=0.3)

        out_path = FIGURES_DIR / f"fig{fig_num}_tsr_{regime}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"  Saved: {out_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_figure_3()
    generate_figures_4_5()
    print("\nDone. Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
