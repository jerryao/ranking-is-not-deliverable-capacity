# Experiment Protocol: Paper 2 — Task-Level Operational Replay

**Frozen:** 2026-07-12
**Paper:** *Ranking Is Not Deliverable Capacity: Stress-Calibrated Demand Response for Fine-Grained Load Management in Sichuan*
**Status:** Frozen BEFORE full-scale execution. No metric, strategy, or oracle definition changes after this point.

---

## 1. Data Roots

| Artifact | Path |
|---|---|
| V1 benchmark (tasks, users, events) | `D:\项目\在研\四川\论文投稿\20260306\Sichuan2024KGSimDataset` |
| V2 enhanced observational | `D:\项目\在研\四川\Dataset\Sichuan2024KGSimDataset_v2\observational\task_assessments_v2_enhanced.csv` |
| V2 intervention pairs | `...\intervention_pairs\intervention_pairs.csv` |
| V2 tau (stress oracle) | `...\intervention_pairs\intervention_pairs_tau.csv` |
| Dual oracle results | `D:\项目\在研\四川\Dataset\paper1_architecture\validation\dual_oracle_results.csv` |
| Tasks definition | `...\Sichuan2024KGSimDataset\tasks.csv` |
| Users per city | `...\Sichuan2024KGSimDataset\City-{A,B,C}\users.csv` |

## 2. Join Keys

Primary key for all joins: **(city, user_id, task_id, event_id)**

Additional key in V2: `pair_id` (maps intervention_pairs to intervention_pairs_tau).

Verified: observational ↔ dual_oracle = 100% match on 9,750 rows.

## 3. Task Definitions (Frozen from tasks.csv)

| Task | P_req (kW) | Duration (h) | r_req | η_req | Season |
|---|---|---|---|---|---|
| T1 summer_peak_relief | 180 | 2 | 0.52 | 0.76 | summer |
| T2 summer_peak_extended | 260 | 3 | 0.56 | 0.78 | summer |
| T3 winter_evening_relief | 200 | 2 | 0.48 | 0.80 | winter |
| T4 daytime_balancing | 140 | 3 | 0.44 | 0.72 | all |
| T5 evening_regulation | 160 | 2 | 0.50 | 0.77 | all |

## 4. Candidate Resource Pools

For each (city, task_id), the candidate pool = all users in that city assessed under that task.

| City | Users | Tasks | Assessments |
|---|---|---|---|
| City-A | 800 | 5 | 4,000 |
| City-B | 500 | 5 | 2,500 |
| City-C | 650 | 5 | 3,250 |
| **Total** | **1,950** | **5** | **9,750** |

## 5. Requirement Levels for Sweep

Beyond the native P_req from tasks.csv, sweep at:
`P_req ∈ {P_task, 2×P_task, 5×P_task, 10×P_task}`

This covers both the native task scale (140–260 kW) and stress scales (up to 2,600 kW).

## 6. Oracle Definitions

### 6.1 Routine oracle (observational regime)
```
M_i^routine = actual_delivered_kw  (from task_assessments_v2_enhanced.csv)
```
This is the kW actually delivered under the user's observational violation state.

### 6.2 Stress oracle (maximum violation regime)
Reconstructed from the dose-response formula (verified in dual_oracle.py):
```
Y(V) = base × rr × exp(s) × event_factor + eps
tau_stress = base × rr × (exp(s_stress) - 1) × event_factor
```
Where:
- base = `pred_reliable_deliverable_capacity_kw`
- rr = `response_rate`
- s_stress = Σ_k(w_k × 1.0 × 1.0 × 0.5) = -0.425 (all classes ON, sev=1.0, scope=0.5)
- event_factor = 1.0 + 0.15 × (event_intensity - 0.3)

Stress delivery at the `actual_delivered_kw` scale is reconstructed as:
```
M_i^stress = actual_delivered_kw × [Y(V_max) / Y(V_obs)]
           = actual_delivered_kw × exp(s_stress - s_obs)
```
where `s_obs = Σ_k(w_k × V_k_flag × V_k_severity × V_k_scope)`.

This preserves the per-user delivery ratio shift and is consistent with the dose-response surface.

## 7. Strategy Definitions

### Group 1: Immediately Available (no training needed)

| ID | Strategy | M_hat_i | Ranking key | Data source |
|---|---|---|---|---|
| S0 | **Oracle** | `actual_delivered_kw` | `actual_delivered_kw` | Ground truth upper bound |
| S1 | **CapacityOnly** | `nominal_capacity_kw × response_rate` | Same | Naive capacity shortcut |
| S2 | **PlatformModel** | `pred_reliable_deliverable_capacity_kw` | Same | V2 platform assessment model |
| S3 | **Linear4** | Fitted `a + b₁×cap + b₂×rr + b₃×availability + b₄×event_intensity` | Same | 4-feature OLS (fit on validation split) |

### Group 2: Calibration Strategies (post-hoc, fit on validation)

| ID | Strategy | M_hat_i |
|---|---|---|
| S4 | **GlobalCalib** | `a + b × pred_reliable_deliverable_capacity_kw` (global OLS) |
| S5 | **CityCalib** | `a_c + b_c × pred_reliable_deliverable_capacity_kw` (per-city OLS) |
| S6 | **ConservativeQ10** | 10th percentile prediction (quantile regression, q=0.10) |

### Group 3: CATE Model (Phase 4, not blocking)

| ID | Strategy | M_hat_i |
|---|---|---|
| S7 | **TauMLP** | `Ŷ(0) + τ̂_tau-mlp` (requires checkpoint re-inference on 9,750) |
| S8 | **DragonNet** | `Ŷ(0) + τ̂_dragonnet` (requires retrain or re-inference) |

**Note:** S7/S8 require resolving the Y(0) baseline question (see §9).

## 8. Operational Metrics (Frozen)

For each (city, task, P_req level, strategy) combination:

| Metric | Formula | Interpretation |
|---|---|---|
| **P_commit** | Σ M_hat_i over selected S | What the platform promises |
| **P_deliv** | Σ M_i^oracle over selected S | What users actually deliver |
| **Shortfall** | max(0, P_req − P_deliv) | Capacity deficit (kW) |
| **OCR** | max(0, P_commit − P_deliv) / P_commit | Over-commit rate |
| **TSR** | I(P_deliv ≥ P_req) | Task success rate (per task) |
| **UAR** | N_admitted_but_insufficient / N_admitted | Unsafe admission rate |
| **N_selected** | \|S\| | Number of users called |
| **Reserve R95** | min{R : P(P_deliv + R ≥ P_req) ≥ 0.95} | 95% backup margin |

### Selection Rule (Frozen)
Greedy descending by M_hat_i until Σ M_hat_i ≥ P_req. Ties broken by user_id ascending.

## 9. Y(0) Baseline Question (Must Resolve Before S7/S8)

The dose-response formula gives: `delivery = base × rr × exp(s) × event_factor`.

- If `pred_reliable_deliverable_capacity_kw` = Y(0) baseline → cannot add τ̂ again (double-count)
- If `pred_reliable_deliverable_capacity_kw` = Y(V_obs) → must subtract τ_obs before adding τ̂

**Verification approach:** Regress `delivery` outcome on `pred_reliable × rr × event_factor`. If coefficient ≈ 1.0, then pred = base = Y(0) component.

Preliminary check (Sprint): for user 0, City-A, T1:
- delivery = 76.19, pred = 138.18, rr = 0.55, event_factor = 1.003
- pred × rr × event_factor = 76.27 ≈ delivery ✅
- **Conclusion: `pred_reliable_deliverable_capacity_kw` functions as the Y(0) base in the dose-response formula.**

Therefore for S7/S8: `M_hat = pred_reliable × rr × event_factor + τ̂_model` (where τ̂ is the model's effect prediction).

## 10. Train/Val/Test Protocol for Calibrated and Linear Strategies

Since the benchmark is a simulator (not real-world), chronological splitting is less critical. Use:

- **Test set:** All 9,750 instances (the full benchmark)
- **Validation set (for fitting S3–S6):** 30% random holdout with seed=42
- **Training set (for S3–S6):** Remaining 70%
- Metrics reported on the full 9,750

**Justification:** The paper's claim is about operational replay on the benchmark, not generalization to unseen data. The split prevents in-sample optimism for calibrated strategies but does not restrict the replay.

## 11. FORBIDDEN Fields (Anti-Leakage)

The following fields MUST NOT be used for ranking or M_hat construction:
- `actual_delivered_kw` (oracle routine delivery)
- `delivery_success_flag`
- `shortage_kw`
- `true_tau_delivery` (from intervention_pairs_tau)
- `tau_dual`, `tau_stress_formula` (from dual_oracle_results)
- `Y_delivery_1` (stress arm outcome)
- Any outcome column: `delivery`, `comfort_loss`, `rebound_risk`, `contract_penalty`, `instability`

**Exception:** S0 (Oracle) uses `actual_delivered_kw` as M_hat — but only as a theoretical upper bound, never as a deployable strategy.

## 12. Random Seeds

| Purpose | Seed |
|---|---|
| Validation split (S3–S6 fitting) | 42 |
| Future model training (S7/S8) | {42, 123, 456} (matching Paper 1 pool) |
| Greedy tie-breaking | user_id ascending (deterministic) |

## 13. Output Artifacts

```
experiments/
├── EXPERIMENT_PROTOCOL.md          (this file, frozen)
├── e3_replay_engine.py             (replay engine)
├── e3_minimal_prototype.py         (sprint smoke test)
├── e3_sweep.py                     (P_req sweep)
├── results/
│   ├── e3_routine_replay.csv       (3×5×4×8 = 480 rows)
│   ├── e3_stress_replay.csv        (3×5×4×8 = 480 rows)
│   ├── e3_summary_table.tex        (aggregated for paper)
│   └── e3_raw_predictions/         (per-strategy per-task)
└── predictions/                     (Phase 4, CATE model outputs)
    └── tau_mlp_seed42_full9750.csv
```

## 14. DragonNet/Tau-MLP Status (Phase 0 Finding)

| Item | Status |
|---|---|
| Training code | EXISTS: `submission_v5_scrl/code/dragonnet.py`, `run_p01_experiments.py` |
| Checkpoints | EXISTS: Tau-MLP + FCRA in `results_frozen/p01/.../checkpoint.pt` |
| SC-DragonNet | DEGRADED: ρ=0.750, inf PEHE at 1%/5% budgets |
| Tau-MLP | VIABLE: ρ=0.952@10%, PEHE=3.31 kW |
| Existing predictions | 3,900 training-pool instances only (NOT 9,750) |
| 4-key join | NOT available for predictions |
| **Action** | Re-infer Tau-MLP checkpoint on 9,750; if incompatible, retrain |

**DragonNet does NOT block E3.** S0–S6 are fully executable now.

## 15. Decision Gate

E3 full replay (S0–S6) must complete and freeze before:
1. Adding S7/S8 (CATE models) to the main table
2. Starting paper rewrite (Phase 5)

If S0–S6 already demonstrate the core finding (ranking ≠ deliverable capacity), S7/S8 become confirmatory rather than load-bearing.
