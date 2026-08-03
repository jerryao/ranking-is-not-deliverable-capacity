# Dataset v2 — Schema Definitions

This document specifies every output field of the upgraded `Dataset v2`. All
files are written under the directory created by `--output-root` of
`upgrade_dataset_v2.py`.

## High-level layout

```
Dataset_v2/
├── observational/                      # Instruction 4 (overlap-repaired)
│   └── task_assessments_v2_enhanced.csv
├── intervention_pairs/                 # Instructions 1 + 5
│   ├── intervention_pairs.csv           # long-form: 2 rows per pair
│   └── intervention_pairs_tau.csv       # wide-form: τ per outcome
├── scenario_sets/                      # Instruction 7
│   ├── decision_scenarios.csv           # all three classes
│   ├── boundary_scenarios.csv
│   └── stress_scenarios.csv
├── anchor_cross_city/                  # Instruction 10
│   └── anchor_scenarios.csv
├── safety_cost_labels/                 # Instruction 6
│   └── safety_cost_labels.csv
├── metadata/
│   ├── structured_labels_v2.csv        # Instruction 9 (label-collapse fix)
│   ├── generation_summary.json
│   ├── v2_generation_config.json
│   └── source_data_card.json
└── validation/
    └── validation_results.json
```

---

## 1. Violation Tensor (Instruction 2)

For every row we materialize a 5-class violation tensor. Each class contributes
4 columns. Total = 20 columns.

| Column pattern | Type | Range | Meaning |
|---|---|---|---|
| `V_<class>_flag` | int ∈ {0,1} | 0/1 | Indicator of whether this class of violation occurred |
| `V_<class>_severity` | float | R+ | Severity in lognormal(μ=-1.5, σ=0.7) |
| `V_<class>_duration_h` | float | R+ | Duration in hours, exponential(scale=1.5) |
| `V_<class>_scope` | float | [0,1] | Affected scope, Beta(a=2, b=5) |

`<class>` ∈ {`physical`, `mutex`, `comfort`, `hierarchy`, `contract`}.

`V_any_flag` = OR of all five `V_<class>_flag` (used as a coarse treatment).

---

## 2. Outcome channels (Instruction 3)

All five channels are produced by the same dose-response function. Violations
affect each channel through a transparent linear-formula weight in `config.json`.

| Channel | Range | Increasing direction |
|---|---|---|
| `delivery` | R+ (kW) | Higher is better |
| `comfort_loss` | [0,1] | Higher is worse |
| `rebound_risk` | [0,1] | Higher is worse |
| `contract_penalty` | [0,1] | Higher is worse |
| `instability` | [0,1] | Higher is worse |

Mechanism:
```
Y_outcome = clip_(outcome-specific)(
    baseline_outcome + Σ_k  w_{k→outcome} · I_k · severity_k · scope_k
                              + outcome-specific noise ε_outcome
)
```

Weights (sign convention: + means violation increases the channel):

| k → outcome | delivery | comfort_loss | rebound_risk | contract_penalty | instability |
|---|---:|---:|---:|---:|---:|
| physical  | -0.55 | +0.05 | +0.10 | +0.02 | +0.15 |
| mutex     | -0.18 | +0.08 | +0.25 | +0.04 | +0.30 |
| comfort   | -0.08 | +0.40 | +0.05 | +0.01 | +0.10 |
| hierarchy | -0.04 | +0.02 | +0.10 | +0.02 | +0.20 |
| contract  |  0.00 |  0.00 |  0.00 | +0.50 | +0.05 |

---

## 3. Intervention pair fields (Instruction 1)

The long-form `intervention_pairs.csv` contains 2 rows per base row.

| Field | Notes |
|---|---|
| `pair_id` | 16-char SHA1 of (city, user_id, task_id, event_id, salt) |
| `source` | `"do(V=0)"` or `"do(V=1)"` |
| `Y_delivery_0`, `Y_delivery_1` | kW under each intervention |
| `Y_comfort_0`, `Y_comfort_1` | comfort_loss under each intervention |
| `Y_rebound_0`, `Y_rebound_1` | rebound_risk under each intervention |

The wide-form `intervention_pairs_tau.csv` contains one row per base row.

| Field | Notes |
|---|---|
| `pair_id`, `city`, `user_id`, `task_id`, `event_id`, `seed` | identifiers |
| `true_tau_delivery` = `Y_delivery_1 - Y_delivery_0` | guaranteed ≤ 0 |
| `true_tau_comfort_loss` = `Y_comfort_1 - Y_comfort_0` | guaranteed ≥ 0 |
| `true_tau_rebound_risk` | guaranteed ≥ 0 |
| `true_tau_contract_penalty` | guaranteed ≥ 0 |
| `true_tau_instability` | guaranteed ≥ 0 |

Both arms share the same ε noise (verified by `validate_v2.py::shared_noise_stable`).

---

## 4. Safety cost and reward (Instruction 6)

```
safety_cost = 0.40·comfort_loss + 0.25·rebound_risk
            + 0.20·contract_penalty + 0.15·instability

reward      = delivery_kw_normalized − λ · safety_cost     (λ = 1.5)
```

`delivery_kw_normalized` divides by the 99th percentile of observed delivery.

---

## 5. Scenario classes (Instruction 7)

| Class | Ratio | demand / capacity | Notes |
|---|---:|---|---|
| `easy`      | 30% | U(0.30, 0.60) | no constraint hits |
| `boundary`  | 50% | U(0.85, 1.15) | demands approximately equal capacity |
| `stress`    | 20% | U(1.10, 1.50) + capacity clipped to 60% | overload under tight capacity |

Each scenario row contains all 20 V columns and all 5 outcome channels plus
`safety_cost`, `reward`, `scenario_class`. Identifiers use `pair_id = SCNxxxxxxx`.

---

## 6. Anchor set (Instruction 10)

Each `anchor_id` is replicated across the three cities, holding
`(shared_X_cluster, task_id, demand_kw, event_intensity_band)` fixed and only
varying `city_dynamics` (`response_rate`, `response_delay_min`).

This disentangles **distribution shift** (P(X)) from **mechanism shift**
(P(Y|X)).

---

## 7. Customer clustering (Instruction 8)

```
customer_cluster_id := hash(city | base_cluster | industry_type | capacity_quartile)
```

`customer_cluster_id` is the primary causal unit. Use it for cluster-robust
standard errors (`df.groupby("customer_cluster_id").cluster...`).

---

## 8. Label fix (Instruction 9)

`pv_label` and `work_rest_label` from the original dataset are 100% / 99.5%
degenerate. v2 produces:

- `pv_label_v2`     = quantile bin (3 bins, names `low`/`medium`/`high`)
- `work_rest_label_v2` = quantile bin (3 bins)

Saved in `metadata/structured_labels_v2.csv`.

---

## 9. Validation contract

`validate_v2.py` produces `validation/validation_results.json` with exactly:

```json
{
  "customer_split_no_leakage": true,
  "propensity_valid": true,
  "typed_violation_columns_present": true,
  "pair_ids_unique_per_intervention": true,
  "shared_noise_stable": true,
  "delivery_tau_nonpositive": true,
  "safety_tau_nonnegative": true,
  "scenario_classes_valid": true,
  "scenario_classes_complete": true,
  "anchors_cover_three_cities": true,
  "all_pass": true
}
```

`all_pass=true` is the success criterion for the v2 generation pipeline.