# Response to Reviewer

We sincerely thank the reviewer for the careful, technically detailed, and constructive assessment. The review identified two issues that proved fundamental: the base-field specification (Item 3) and the absence of execution-level delivery degradation (Items 2, 5). Addressing these led to a substantially revised manuscript with a literature-calibrated execution realization factor, $N=30$ independent benchmark realisations, Wilson 95\% confidence intervals, and four new sensitivity analyses.

**Summary of major changes:**

1. **Base field corrected** (Item 3): $P_i^{\mathrm{base}}$ changed from `pred_reliable_deliverable_capacity_kw` to `nominal_capacity_kw` (connected load). Response rate is now applied exactly once.

2. **Execution realization factor $\xi$ added** (Items 2, 5): A per-event multiplicative factor $\xi \sim \mathrm{Beta}(5.92, 2.91)$ with $E[\xi] = 0.67$, calibrated to empirical PJM/MISO/NYISO delivery ratios. This captures delivery degradation from sources not modelled by the violation dose (equipment, behaviour, communications, baseline error).

3. **Multi-seed analysis** (Items 6, 7): All results report $N=30$ independent benchmark realisations with Wilson 95\% CI and seed-clustered bootstrap CI.

4. **New calibration methods** (Item 11): Isotonic regression (S7) and split-conformal lower bound (S8) added.

5. **Q-quantile sensitivity** (Item 12): Q05/Q10/Q20/Q30 reported with per-fold quantile values.

6. **Holdout protocols** (Item 9): User-level and city-level holdout added.

7. **Top-k metrics** (Item 16): Top-k overlap and capacity regret reported.

8. **R$^{95}$ removed** (Item 18): The deterministic reserve-margin claim was deleted.

---

## Major Comments

### Comment 1 — Boundary between real and simulated context

**Response:** Fully addressed. The manuscript now explicitly states that all user-level delivery values are simulator-generated and that $\xi$ is calibrated to—but not independently validated against—external DR delivery data. The term "actual delivered capacity" has been replaced throughout with "simulated delivered capacity" or "benchmark-realised delivery." The abstract and conclusions clearly distinguish external calibration from external validation.

### Comment 2 — Simulator specification completeness

**Response:** Addressed. The execution realization factor $\xi$ is now fully specified: distribution (Beta(5.92, 2.91)), mean (0.67), std (0.15), support ((0, 1]), calibration source (PJM Summer 2025 average delivery ratio), and per-event independence assumption. A complete parameter table will appear in Supplementary Section S2. The $\xi$ sensitivity analysis (Supplementary Section S3) tests robustness across 4 means, 4 standard deviations, 3 distribution families, and 3 persistence structures.

### Comment 3 — Double application of response rate

**Response:** The reviewer's observation was prescient and correctly diagnosed. Code audit confirmed that `pred_reliable_deliverable_capacity_kw` already contained a partial response-rate adjustment from the v1 platform-assessment model, and the v2 generator multiplied by `response_rate` again, producing an effective $\sim$1.5$\times$ compounding. The base field has been changed to `nominal_capacity_kw` (connected load) throughout: `oracle_simulator.py`, `upgrade_dataset_v2.py`, and `e3_replay_engine_v3.py`. The dose--response equation (Eq. 12) now uses $P_i^{\mathrm{nom}}$ and applies $r_i$ exactly once.

We thank the reviewer for identifying this issue, which was the single most consequential finding of the review.

### Comment 4 — Stress contrast plausibility

**Response:** Addressed. The standardised severe stress contrast is now described as a deliberately adverse upper-bound benchmark scenario, not a representative system state. A stress-response sweep across 14 configurations ($K_{\mathrm{active}} \in \{1,2,3,5\}$, severity $\in \{0.25,0.50,0.75,1.00\}$, scope $\in \{0.25,0.50,0.75\}$, plus 5 individual-class activations) is reported, showing a monotonic stress--TSR response curve. The $\sim$92$\times$ median ratio is reported only as a consequence of the specific benchmark stress configuration.

### Comment 5 — External validation

**Response:** We added literature-based external calibration and cross-market plausibility checks rather than claiming direct external validation against the Sichuan platform. Nine peer-reviewed and industry sources document real-world DR delivery ratios of 26--80\% (PJM 67\%, Winter Storm Elliott 26--32\%, MISO 67\%, NYISO 51--92\%, Korean residential 83\% overestimated). $E[\xi] = 0.67$ is calibrated to the PJM Summer 2025 average. We explicitly state in the limitations that this is calibration, not independent validation.

### Comment 6 — Effective number of evaluation units

**Response:** Fully addressed. All TSR values now report both percentage and underlying count, e.g., "42.9\% (193/450)." Wilson 95\% confidence intervals are computed from $n = 30 \times 15 = 450$ pooled cells per strategy--requirement combination. Seed-clustered bootstrap intervals are also reported.

### Comment 7 — Single deterministic realisation

**Response:** Fully addressed. All results use $N = 30$ independent benchmark realisations with master seeds $2048 + i \times 10^9$ for $i \in \{0, \ldots, 29\}$. The full generation--replay pipeline is repeated for each seed. The $\xi$ robustness sweep (Supplementary Section S3) additionally tests 11 configurations at $N = 10$ seeds each.

### Comment 8 — Model taxonomy

**Response:** Addressed. A consolidated model table distinguishes target variable ($\tau_i$ vs $M_i$), feature set, training data, validation protocol, output unit, and operational role for each strategy. The delivery-capacity strategies (S1--S6) and new calibration methods (S7 Isotonic, S8 ConformalLCB) are clearly separated from the effect-estimation diagnostics (Table 2).

### Comment 9 — Leave-one-task-out vs generalisation

**Response:** Addressed. Three holdout protocols are now reported:
- Leave-one-task-out: known users, new task template
- User holdout: new users (all records held out)
- City holdout: new city configuration

Results show calibration generalises well: GlobalCalib TSR at $5\times$ is 87.3\% (LOTO), 87.3\% (user holdout), and 86.7\% (city holdout). City-C is the hardest transfer target (76.0\%).

### Comment 10 — Anti-leakage and negative prediction handling

**Response:** Confirmed. All `mhat` predictions are clipped to non-negative before ranking and aggregation (`e3_replay_engine_v3.py` L155). Under the corrected base, GlobalCalib is near-identity ($\hat{M} \approx 0.4 + 0.99 \times \mathrm{nominal} \times r$), so negative predictions essentially never occur (0/9750 across 30 seeds).

### Comment 11 — Nonlinear and uncertainty-aware calibration

**Response:** Addressed. Two new methods added:
- **S7 Isotonic**: monotonic nonlinear calibration. TSR@5$\times$ = 82.7\%.
- **S8 ConformalLCB**: split-conformal lower prediction bound (90\% coverage). TSR@5$\times$ = 94.7\%.

The conformal estimator outperforms linear calibration while selecting only 1.5$\times$ more resources than the uncalibrated heuristic, directly addressing the reviewer's concern about uncertainty-aware methods.

### Comment 12 — ConservativeQ10 transparency

**Response:** Fully addressed. The quantile estimation is formally defined, and four quantile levels are reported:

| $\alpha$ | Per-fold quantile | TSR@5$\times$ | $|\mathcal{S}_t|$ | Unused (kW) |
|---|---|---|---|---|
| 0.05 | 0.399 | 100.0\% | 4.0 | 842 |
| 0.10 | 0.460 | 100.0\% | 3.6 | 656 |
| 0.20 | 0.537 | 96.7\% | 3.1 | 433 |
| 0.30 | 0.593 | 94.7\% | 2.9 | 355 |

### Comment 13 — Task vector enforcement

**Response:** Partially addressed. We clarify that the greedy selection rule uses predicted capacity as the binding quantity; availability windows, duration, and reliability requirements are not currently enforced as hard constraints. The formulation has been simplified to match the experiment.

### Comment 14 — Greedy rule limitations

**Response:** Acknowledged in the limitations. Under the simplified equal-cost formulation, descending greedy selection yields the minimum-cardinality subset. The revised limitations section discusses the omission of feeder concentration, correlated failures, rebound, invitation cost, and user fatigue. A constrained sensitivity analysis (30\% random exclusion) shows the ranking--calibration gap persists.

### Comment 15 — Pool feasibility values

**Response:** Addressed. A feasibility table reports total nominal capacity, total routine delivery, total stress delivery, and utilisation ratios for all 15 city--task cells. All pools remain stress-feasible at $5\times$ (utilisation 0.7--2.4\%).

### Comment 16 — Top-k metrics

**Response:** Addressed. Top-k overlap with the oracle and capacity regret are reported at portfolio sizes corresponding to $1\times$--$10\times$ requirements:

| Strategy | Req | mean $k$ | Top-k overlap | Capacity regret |
|---|---|---|---|---|
| S1 CapRR | $5\times$ | 1.9 | 0.427 | 0.134 |
| S4 GlobalCalib | $5\times$ | 2.6 | 0.506 | 0.112 |

The gap is driven more by magnitude bias (capacity regret 11--13\%) than by pure ranking errors (overlap 43--51\%).

### Comment 17 — Requirement relative to pool capacity

**Response:** Addressed. Pool-relative requirements ($\alpha \times \mathrm{pool\_total}$) were tested. At $\alpha = 0.10$--$0.40$, a partial routine gap appears (S1 TSR 40--73\%), driven primarily by cardinality effects (92--96\%), not ranking errors.

### Comment 18 — Reserve margin

**Response:** Fully resolved. The deterministic reserve-margin claim has been removed. The 35\% stress-induced reduction is described only as the average benchmark loss under the selected stress scenario. $R^{95}$ is not estimated; reserve sizing is deferred to future work.

### Comment 19 — Number discrepancies

**Response:** All discrepancies corrected through a systematic check of every numerical statement against the $N=30$ frozen results. The manuscript now uses a single validated results file (`multiseed_tsr_with_ci.csv`) for all tables and headline numbers.

### Comment 20 — Mean vs median ratio

**Response:** The $\sim$32$\times$ mean ratio and $\sim$92$\times$ median paired ratio are now distinguished wherever cited. The sensitivity of the median ratio to near-zero denominators is noted.

---

## Minor Comments

### Comments 21--40

| # | Response summary |
|---|---|
| 21 | Ethics statement clarifies benchmark is fully synthetic with $\xi$ calibrated to published aggregate delivery ratios. |
| 22 | "Routine regime" and "standardised stress regime" used consistently. |
| 23 | Notation table added. |
| 24 | $\tau_i \leq 0$ stated as imposed by design. |
| 25 | Zero-floor analysis: under corrected base + $\xi$, delivery min $\geq$ 18 kW across all 30 seeds; floor never triggered. |
| 26 | Database field names replaced with $P_i^{\mathrm{nom}}$ and $P_i^{\mathrm{rated}}$ in main text; original field names retained only in Supplementary data dictionary. |
| 27 | "RoutineOracle" retained but its scope is clearly stated. |
| 28 | Figures have been regenerated with larger labels at final dimensions. |
| 29 | Tables report percentages with CI reference to Supplementary S3; OCR/Shortfall table cites Supplementary for SD, IQR, and clustered-bootstrap intervals. |
| 30 | Literature review strengthened with PJM, MISO, NYISO empirical sources. |
| 31 | Categorical novelty claim softened to "limited prior work has explicitly evaluated." |
| 32 | S2 described as "benchmark platform-rating proxy," not the deployed Sichuan model. |
| 33 | CityCalib parameters: per-fold $(a_c, b_c)$, training sample sizes, and parameter variability reported in Supplementary S3; smaller effective training samples noted as source of parameter variance. |
| 34 | $1\times$ portfolio size distribution reported. |
| 35 | Bias statistics: denominator, regime, and fold structure stated. |
| 36 | Data availability statements harmonised. |
| 37 | CRediT statement completed with role-specific contributions for all six authors. |
| 38 | Dedicated reproducibility repository created at `https://github.com/jerryao/ranking-is-not-deliverable-capacity`, including simulator, parameters, 30 seeds, fold assignments, calibration/replay/analysis scripts, frozen results, and figure/table reproduction scripts. Archived under release `v1.0.0-review` at commit `67d9fb7`; Zenodo DOI upon publication. |
| 39 | Supplementary S1–S3 and Figure S1 included with numerical test outputs. |
| 40 | Duplicated phrase removed. $\xi$ terminology unified as "execution realization factor." American English spelling standardized. Notation table added. |

---

## Closing

We thank the reviewer again for identifying the base-field specification issue (Item 3) and the absence of execution-level degradation (Items 2, 5), which were the two most consequential observations. The revised manuscript incorporates a literature-calibrated execution realization factor, $N=30$ independent benchmark realisations with Wilson 95\% CI, four sensitivity analyses ($\xi$ robustness, stress sweep, calibration methods, holdout protocols), and top-k ranking diagnostics. The central finding—that high ranking ability does not guarantee deliverable capacity accuracy, and that calibration substantially closes this gap—is robust across all tested configurations.
