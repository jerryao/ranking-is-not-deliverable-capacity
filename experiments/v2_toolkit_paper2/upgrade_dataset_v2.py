"""Sichuan2024KGSimDataset → Dataset v2 upgrade orchestrator.

Implements all 10 engineering instructions from the v2 spec:

    1. Intervention pair data          → intervention_pairs/intervention_pairs.csv
    2. 5-class violation tensor         → 20 new V_* columns (per-row)
    3. Dose-response mechanism          → delivery/comfort_loss/rebound_risk/
                                          contract_penalty/instability
    4. Overlap repair                  → observational/task_assessments_v2_enhanced.csv
    5. Oracle simulator                → pair-level ground-truth tau
    6. Safety-cost + safety-aware R    → safety_cost, reward columns
    7. Decision-boundary scenarios      → scenario_sets/*.csv  (Easy/Boundary/Stress)
    8. Customer-level clustering       → customer_cluster_id in all outputs
    9. Label collapse fix              → pv_label_v2, work_rest_label_v2
   10. Cross-city anchor set           → anchor_cross_city/anchor_scenarios.csv

Usage:
    python upgrade_dataset_v2.py \\
        --input-root  D:/项目/在研/四川/Dataset/Sichuan2024KGSimDataset \\
        --output-root D:/项目/在研/四川/Dataset/Sichuan2024KGSimDataset_v2 \\
        --config      config.json \\
        [--max-rows N]   # for smoke-testing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from v2lib.config import Config
from v2lib.io_utils import (
    load_task_assessments, load_users, load_structured_labels,
    write_csv, write_json, ensure_dirs,
)
from v2lib.cluster_utils import assign_customer_clusters
from v2lib.label_fix import fix_labels
from v2lib.intervention_pairs import generate_intervention_pairs
from v2lib.overlap_repair import repair_overlap
from v2lib.violation_tensor import generate_violation_tensor
from v2lib.safety_cost import add_safety_columns
from v2lib.scenarios import generate_decision_scenarios
from v2lib.anchor_set import generate_anchor_set


# Module-level mutable state shared by step()/finish_step()/register_output()
# and populated by main(). This is the simplest way to let helpers append
# provenance records without threading an explicit context object through
# every call site in the orchestrator.
summary: dict = {}


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
    except Exception:
        return "unversioned"


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def step(msg: str, step_id: str | None = None,
         seed: int | None = None,
         inputs: list[str] | None = None,
         script: str | None = None) -> None:
    """Print a step banner AND append a structured record to summary['steps']."""
    print(f"\n[step] {msg}", flush=True)
    rec = {
        "step_id": step_id or msg.lower().replace(" ", "_").replace("(", "").replace(")", ""),
        "msg": msg,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "script": script or Path(__file__).name,
        "code_version": _git_commit(HERE),
        "master_seed": seed,
        "inputs": list(inputs) if inputs else [],
        "outputs": [],
        "warnings": [],
    }
    summary["steps"].append(rec)


def finish_step(warnings: list[str] | None = None) -> None:
    """Mark the most recent step record as completed. Captures warnings."""
    if summary["steps"]:
        rec = summary["steps"][-1]
        rec["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
        if warnings:
            rec["warnings"] = list(warnings)


def register_output(rel_path: str, n_rows: int | None = None) -> None:
    """Append an output artifact record to the current step."""
    if not summary["steps"]:
        return
    rec = summary["steps"][-1]
    full = Path(summary["_out_root"]) / rel_path
    entry = {
        "path": rel_path.replace("\\", "/"),
        "sha256": _sha256_of(full),
        "rows": n_rows,
    }
    rec["outputs"].append(entry)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate Dataset v2 from v1 source.")
    p.add_argument("--input-root", required=True, type=str)
    p.add_argument("--output-root", required=True, type=str)
    p.add_argument("--config", default=str(HERE / "config.json"), type=str)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Optional row cap on task_assessments for smoke tests.")
    args = p.parse_args(argv)

    src_root = Path(args.input_root)
    out_root = Path(args.output_root)
    cfg = Config.from_json(args.config)
    print(f"[init] config: schema_version={cfg.raw['schema_version']}")
    print(f"[init] seed_policy: {cfg.raw['seed_policy']}")

    paths = ensure_dirs(out_root)
    summary.clear()
    summary.update({"steps": [], "warnings": [], "config": cfg.raw,
                    "_out_root": str(out_root),
                    "toolkit_version": _git_commit(HERE),
                    "toolkit_root": str(HERE)})

    # ---------- Load source ----------
    step("load source task_assessments + users + structured_labels", step_id="v1_load",
         seed=cfg.seed,
         inputs=[f"{src_root}/City-{{A,B,C}}/task_assessments.csv",
                 f"{src_root}/City-{{A,B,C}}/users.csv",
                 f"{src_root}/City-{{A,B,C}}/structured_labels.csv"])
    ta = load_task_assessments(src_root)
    users = load_users(src_root)
    sl = load_structured_labels(src_root)
    if args.max_rows is not None and args.max_rows > 0:
        ta = ta.sample(n=min(args.max_rows, len(ta)), random_state=cfg.seed).reset_index(drop=True)
        print(f"[init] capped task_assessments to {len(ta):,} rows for smoke run")
    summary["source_rows"] = {
        "task_assessments": len(ta),
        "users": len(users),
        "structured_labels": len(sl),
    }
    print(f"  task_assessments rows: {len(ta):,}")
    print(f"  users rows: {len(users):,}")
    print(f"  structured_labels rows: {len(sl):,}")
    finish_step()

    # ---------- Customer clusters (Instruction 8) ----------
    step("assign customer_cluster_id (Instruction 8)", step_id="customer_clusters")
    users = assign_customer_clusters(users)
    cluster_map = dict(zip(zip(users["city"], users["user_id"]), users["customer_cluster_id"]))
    ta["customer_cluster_id"] = [
        cluster_map.get((c, int(u)), -1)
        for c, u in zip(ta["city"], ta["user_id"])
    ]
    summary["unique_clusters"] = int(ta["customer_cluster_id"].nunique())
    finish_step()

    # ---------- Label fix (Instruction 9) ----------
    step("label collapse fix (Instruction 9)", step_id="label_fix")
    sl_fixed = fix_labels(sl, cfg)
    write_csv(sl_fixed, paths["metadata"] / "structured_labels_v2.csv")
    register_output("metadata/structured_labels_v2.csv", n_rows=len(sl_fixed))
    print(f"  pv_label_v2 distribution: {sl_fixed['pv_label_v2'].value_counts().to_dict()}")
    print(f"  work_rest_label_v2 distribution: {sl_fixed['work_rest_label_v2'].value_counts().to_dict()}")
    finish_step()

    # ---------- Step 1: Build base frame ----------
    step("merge users + structured_labels + task_assessments into base frame", step_id="base_frame")
    base = ta.merge(
        users[["city","user_id","industry_type","user_type","customer_cluster_id",
               "nominal_capacity_kw","response_ramp_score","availability_rate",
               "dr_history_success","process_constraint_score","comfort_constraint_score",
               "rebound_tendency","delivery_uncertainty_score","base_cluster",
               "production_mode","amplitude"]],
        on=["city","user_id"], how="left", suffixes=("","_u")
    )
    base = base.merge(
        sl_fixed[["city","user_id","pv_label_v2","work_rest_label_v2"]],
        on=["city","user_id"], how="left"
    )
    # Ensure required columns exist (fillna)
    for col in ["event_intensity","pred_reliable_deliverable_capacity_kw",
                "response_rate","response_delay_min"]:
        if col not in base.columns:
            base[col] = np.nan
    # nominal_capacity proxy: if missing, derive from pred / response_rate
    if "nominal_capacity_kw" not in base.columns:
        base["nominal_capacity_kw"] = (
            base["pred_reliable_deliverable_capacity_kw"] /
            base["response_rate"].clip(lower=0.05)
        )
    # event_type fallback
    if "event_type" not in base.columns:
        base["event_type"] = "SHOCK"
    finish_step()

    # ---------- Step 2: Generate natural violations (no do()) for the observational layer ----------
    step("generate natural violation tensor for observational layer (Instruction 2)", step_id="violation_tensor_obs")
    vt_nat = generate_violation_tensor(
        n_rows=len(base),
        base_rates={k: float(cfg.raw["violation_tensor"]["base_rates"][k])
                    for k in cfg.violation_classes},
        severity_dist=cfg.raw["violation_tensor"]["severity_dist"],
        duration_dist=cfg.raw["violation_tensor"]["duration_dist"],
        scope_dist=cfg.raw["violation_tensor"]["scope_dist"],
        seed=cfg.seed + 3,
    )
    v_frame = vt_nat.to_frame()
    for col in v_frame.columns:
        base[col] = v_frame[col].to_numpy()
    base["V_any_flag"] = (base[[c for c in v_frame.columns if c.endswith("_flag")]].sum(axis=1) > 0).astype(int)
    finish_step()

    # ---------- Step 3: Dose-response on observational rows ----------
    step("apply dose-response to observational rows (Instruction 3)", step_id="dose_response_obs")
    rng_obs = np.random.default_rng(cfg.seed + 5)
    rng_xi = np.random.default_rng(cfg.seed + 999)
    from v2lib.xi_sampler import sample_xi
    xi_cfg = cfg.raw.get("execution_factor", {})
    xi_obs = sample_xi(
        rng_xi, len(base),
        user_ids=base["user_id"].to_numpy(dtype=np.int64) if "user_id" in base.columns else None,
        mean=xi_cfg.get("mean", 0.67),
        std=xi_cfg.get("std", 0.15),
        dist=xi_cfg.get("dist", "beta"),
        persistence=xi_cfg.get("persistence", "none"),
    )
    from v2lib.dose_response import compute_outcomes, sample_shared_noise
    # Item 3 fix (Paper 2 review): use nominal_capacity_kw (connected load)
    # as the pre-response-rate base; see v2lib/oracle_simulator.py for details.
    obs_out = compute_outcomes(
        base_delivery_kw=base["nominal_capacity_kw"].to_numpy(),
        response_rate=base["response_rate"].to_numpy(),
        response_delay_min=base["response_delay_min"].to_numpy(),
        event_intensity=base["event_intensity"].to_numpy(),
        V=vt_nat, weights=cfg.dose_weights,
        noise=sample_shared_noise(rng_obs, len(base)),
        execution_factor=xi_obs,
    )
    for ch in ["delivery","comfort_loss","rebound_risk","contract_penalty","instability"]:
        base[ch] = getattr(obs_out, ch)
    base = add_safety_columns(base, cfg)
    finish_step()

    # ---------- Step 4: Overlap repair ----------
    step("overlap repair (Instruction 4)", step_id="overlap_repair")
    base_aug = repair_overlap(base, config=cfg, treatment_col="V_any_flag")
    write_csv(base_aug, paths["observational"] / "task_assessments_v2_enhanced.csv")
    register_output("observational/task_assessments_v2_enhanced.csv", n_rows=len(base_aug))
    finish_step()

    # ---------- Step 5: Intervention pairs ----------
    step("intervention pair generation (Instructions 1 + 5)", step_id="intervention_pairs")
    # Item 3 fix (Paper 2 review): pass `base` (merged with users.csv, hence
    # contains nominal_capacity_kw) rather than raw `ta`. The oracle simulator
    # now reads nominal_capacity_kw as the connected-load base.
    long_df, tau_df = generate_intervention_pairs(base, config=cfg)
    write_csv(long_df, paths["intervention_pairs"] / "intervention_pairs.csv")
    write_csv(tau_df,   paths["intervention_pairs"] / "intervention_pairs_tau.csv")
    register_output("intervention_pairs/intervention_pairs.csv", n_rows=len(long_df))
    register_output("intervention_pairs/intervention_pairs_tau.csv", n_rows=len(tau_df))
    finish_step()

    # ---------- Step 6: Decision boundary scenarios ----------
    step("decision-boundary scenarios (Instruction 7)", step_id="scenarios")
    scn = generate_decision_scenarios(base_aug, config=cfg, n_total=1500)
    write_csv(scn, paths["scenario_sets"] / "decision_scenarios.csv")
    write_csv(scn[scn["scenario_class"] == "boundary"],
              paths["scenario_sets"] / "boundary_scenarios.csv")
    write_csv(scn[scn["scenario_class"] == "stress"],
              paths["scenario_sets"] / "stress_scenarios.csv")
    register_output("scenario_sets/decision_scenarios.csv", n_rows=len(scn))
    register_output("scenario_sets/boundary_scenarios.csv",
                    n_rows=int((scn["scenario_class"] == "boundary").sum()))
    register_output("scenario_sets/stress_scenarios.csv",
                    n_rows=int((scn["scenario_class"] == "stress").sum()))
    print(f"  decision scenarios: {scn['scenario_class'].value_counts().to_dict()}")
    finish_step()

    # ---------- Step 7: Cross-city anchor set ----------
    step("cross-city anchor set (Instruction 10)", step_id="anchor_set")
    anchors = generate_anchor_set(base_aug, config=cfg)
    write_csv(anchors, paths["anchor_cross_city"] / "anchor_scenarios.csv")
    register_output("anchor_cross_city/anchor_scenarios.csv", n_rows=len(anchors))
    finish_step()

    # ---------- Step 8: Safety cost labels (per-row) ----------
    step("safety-cost labels (Instruction 6)", step_id="safety_cost_labels")
    safety_labels = base_aug[[
        "city","user_id","task_id","event_id","customer_cluster_id",
        "delivery","comfort_loss","rebound_risk","contract_penalty","instability",
        "safety_cost","reward",
    ]].copy()
    write_csv(safety_labels, paths["safety_cost_labels"] / "safety_cost_labels.csv")
    register_output("safety_cost_labels/safety_cost_labels.csv", n_rows=len(safety_labels))
    finish_step()

    # ---------- Step 9: Metadata ----------
    step("metadata write", step_id="metadata_write")
    summary["outputs"] = {
        "observational": str((paths["observational"] / "task_assessments_v2_enhanced.csv").relative_to(out_root)),
        "intervention_pairs": str((paths["intervention_pairs"] / "intervention_pairs.csv").relative_to(out_root)),
        "intervention_pairs_tau": str((paths["intervention_pairs"] / "intervention_pairs_tau.csv").relative_to(out_root)),
        "decision_scenarios": str((paths["scenario_sets"] / "decision_scenarios.csv").relative_to(out_root)),
        "boundary_scenarios": str((paths["scenario_sets"] / "boundary_scenarios.csv").relative_to(out_root)),
        "stress_scenarios": str((paths["scenario_sets"] / "stress_scenarios.csv").relative_to(out_root)),
        "anchor_scenarios": str((paths["anchor_cross_city"] / "anchor_scenarios.csv").relative_to(out_root)),
        "structured_labels_v2": str((paths["metadata"] / "structured_labels_v2.csv").relative_to(out_root)),
        "safety_cost_labels": str((paths["safety_cost_labels"] / "safety_cost_labels.csv").relative_to(out_root)),
    }
    write_json(summary, paths["metadata"] / "generation_summary.json")
    write_json(cfg.raw, paths["metadata"] / "v2_generation_config.json")
    register_output("metadata/generation_summary.json")
    register_output("metadata/v2_generation_config.json")
    finish_step()
    # Source coverage file
    src_card = {
        "source_root": str(src_root),
        "source_files": [
            "City-A/task_assessments.csv","City-B/task_assessments.csv","City-C/task_assessments.csv",
            "City-A/users.csv","City-B/users.csv","City-C/users.csv",
            "City-A/structured_labels.csv","City-B/structured_labels.csv","City-C/structured_labels.csv",
        ],
        "loaded_rows": summary["source_rows"],
    }
    write_json(src_card, paths["metadata"] / "source_data_card.json")
    register_output("metadata/source_data_card.json")

    print("\n[done] generation complete")
    print(f"  output root: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())