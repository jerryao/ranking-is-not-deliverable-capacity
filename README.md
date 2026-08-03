# Ranking Is Not Deliverable Capacity

Reproducibility repository for the paper:

**"Ranking Is Not Deliverable Capacity: Operational Calibration and Stress Testing in a Sichuan-Informed Demand-Response Benchmark."**

## Contents

| Directory / File | Description |
|---|---|
| `experiments/v2_toolkit_paper2/` | Modified benchmark generator (base-field fix + execution realization factor ξ) |
| `experiments/e3_replay_engine_v3.py` | Operational replay engine (consumes v2 toolkit outputs only) |
| `experiments/multiseed_driver.py` | N=30 multi-seed orchestrator (generation + replay) |
| `experiments/compute_multiseed_ci.py` | Wilson CI + seed-clustered bootstrap CI computation |
| `experiments/conformal_lcb_corrected_n30.py` | Corrected split-conformal LCB at N=30 (one-sided score + finite-sample correction) |
| `experiments/calibration_sensitivity.py` | 8-strategy calibration comparison (CapRR, GlobalCalib, Isotonic, ConformalLCB, Q05/Q10/Q20/Q30) |
| `experiments/holdout_sensitivity.py` | Task/user/city holdout protocols |
| `experiments/stress_sweep.py` | 14-configuration stress scenario sweep |
| `experiments/topk_bootstrap_feasibility.py` | Top-k overlap, capacity regret, bootstrap CI, feasibility tables |
| `experiments/xi_robustness_driver.py` | 11-configuration ξ robustness sweep |
| `experiments/regenerate_figures_final.py` | Figure 3/4/5 generation (seed-clustered bootstrap CI bands) |
| `experiments/results/` | Frozen numerical results (CSV) |
| `manuscript/main_paper2_frontiers.tex` | Revised manuscript source |
| `manuscript/references.bib` | Bibliography |
| `manuscript/figures/` | Generated figures |
| `manuscript/RESPONSE_TO_REVIEWER.md` | Response to reviewer (40 items) |
| `manuscript/COVER_LETTER.md` | Cover letter |
| `supplementary/supplementary_S2_S3.tex` | Supplementary Sections S2 (ξ parameters) and S3 (robustness results) |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate a single seed (smoke test, ~25 seconds)
python experiments/v2_toolkit_paper2/upgrade_dataset_v2.py \
    --input-root <PATH_TO_V1_DATASET> \
    --output-root /tmp/seed0 \
    --config experiments/v2_toolkit_paper2/config.json

python experiments/e3_replay_engine_v3.py \
    --data-v2 /tmp/seed0 \
    --output /tmp/replay_seed0.csv

# Full 30-seed run (~13 minutes)
python experiments/multiseed_driver.py
python experiments/compute_multiseed_ci.py

# Regenerate figures
python experiments/regenerate_figures_final.py
```

## Data Provenance

This repository does **not** contain identifiable customer-level records or proprietary operational data from the Sichuan load-management platform. Sichuan provides the operational context for the study, while all user-level delivery outcomes, stress losses, task-success results, and portfolio-replay results are generated within the synthetic benchmark.

The execution realization factor ξ is mean-anchored to PJM's reported 67% delivery ratio and plausibility-checked against delivery ranges reported by MISO, NYISO, and other empirical studies.

## License

[To be specified upon publication]
