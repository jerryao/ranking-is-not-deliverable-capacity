# Sichuan2024KGSimDataset → Dataset v2 Toolkit

Upgrades the original (v1) closed-loop simulation dataset into a **causal-identifiable,
RL-learnable, safety-calibrated** v2 dataset by adding:

1. Counterfactual intervention pairs (`do(V=0)` / `do(V=1)`)
2. A 5-class violation tensor (indicator + severity + duration + scope)
3. A transparent dose-response mechanism for five outcome channels
4. Overlap-repair augmentation so that `P(V|X) ∈ [0.05, 0.95]` per stratum
5. An oracle simulator that yields ground-truth τ with shared noise
6. A safety-cost variable and safety-aware reward
7. Decision-boundary scenarios (Easy / Boundary / Stress, 30 / 50 / 20)
8. Customer-level clustering for cluster-robust inference
9. Quantile-fixed labels (`pv_label_v2`, `work_rest_label_v2`)
10. Cross-city anchor set with shared `(X, T, demand, event)` features

## Directory layout

```
Sichuan2024KGSimDataset_v2_toolkit/
├── config.json                  # all tunable parameters
├── upgrade_dataset_v2.py        # orchestrator
├── validate_v2.py               # validator (produces 11-pass report)
├── README.md                    # this file
├── SCHEMA_V2.md                 # full schema definitions
└── v2lib/                       # generation modules
    ├── __init__.py
    ├── config.py
    ├── io_utils.py
    ├── cluster_utils.py
    ├── label_fix.py
    ├── violation_tensor.py
    ├── dose_response.py
    ├── oracle_simulator.py
    ├── intervention_pairs.py
    ├── overlap_repair.py
    ├── safety_cost.py
    ├── scenarios.py
    └── anchor_set.py
```

## Quick start

```bash
# 1) Smoke test on 200 rows (≈30 seconds)
python upgrade_dataset_v2.py \
  --input-root  "D:/项目/在研/四川/Dataset/Sichuan2024KGSimDataset" \
  --output-root "D:/项目/在研/四川/Dataset/Sichuan2024KGSimDataset_v2_toolkit/_smoke" \
  --max-rows 200

python validate_v2.py \
  --dataset-root "D:/项目/在研/四川/Dataset/Sichuan2024KGSimDataset_v2_toolkit/_smoke"

# 2) Full run on 9,750 rows (≈ 3–5 minutes)
python upgrade_dataset_v2.py \
  --input-root  "D:/项目/在研/四川/Dataset/Sichuan2024KGSimDataset" \
  --output-root "D:/项目/在研/四川/Dataset/Sichuan2024KGSimDataset_v2"

python validate_v2.py \
  --dataset-root "D:/项目/在研/四川/Dataset/Sichuan2024KGSimDataset_v2"
```

If `validate_v2.py` prints `all_pass: true` at the end, the v2 dataset
satisfies all 11 v2 contract checks.

## Outputs

| Subdirectory | File | What it contains |
|---|---|---|
| `observational/`     | `task_assessments_v2_enhanced.csv` | source rows + 20 V columns + 5 outcomes + safety_cost/reward + overlap-augmented rows |
| `intervention_pairs/`| `intervention_pairs.csv`           | long-form (2 rows per pair) with V=0/V=1 indicator |
| `intervention_pairs/`| `intervention_pairs_tau.csv`       | wide-form: τ for each outcome channel |
| `scenario_sets/`     | `decision_scenarios.csv`           | all three classes with ratios 30/50/20 |
| `scenario_sets/`     | `boundary_scenarios.csv`           | boundary subset |
| `scenario_sets/`     | `stress_scenarios.csv`             | stress subset |
| `anchor_cross_city/` | `anchor_scenarios.csv`             | cross-city shared-X counterfactuals |
| `safety_cost_labels/`| `safety_cost_labels.csv`           | per-row safety_cost and reward |
| `metadata/`          | `structured_labels_v2.csv`         | quantile-fixed labels |
| `metadata/`          | `generation_summary.json`          | pipeline run summary |
| `metadata/`          | `v2_generation_config.json`        | exact config used |
| `metadata/`          | `source_data_card.json`            | source files consumed |
| `validation/`        | `validation_results.json`          | 11-pass contract report |

## Key design choices

- **Shared ε noise** between `do(V=0)` and `do(V=1)` arms: counterfactual
  τ is identified at the row level.
- **Customer-level clustering** is the primary causal unit; chronological
  split does not re-shuffle customers within a cluster.
- **5-class violation tensor** prevents the binary-treatment pathology
  where the propensity becomes degenerate.
- **Quantile-fixed labels** repair the pv_label / work_rest_label collapse
  in the original schema.

See `SCHEMA_V2.md` for full field definitions and the dose-response weight table.

## Reproducibility

All randomness flows from the seed formula in `config.json`:

```
seed = 2048 + city_code * 1,000,000 + integer_user_id + offset
```

where `offset` is `intervention_pair_offset = 7`, `scenario_offset = 13`,
or `anchor_offset = 19` for each subsystem. Generator is
`numpy.random.default_rng(SeedSequence(seed))` (PCG64 family).