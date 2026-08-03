"""Unify V1 source + V2 generated outputs into one complete dataset.

Usage:
    python build_complete.py \\
        --v1-root D:/.../Sichuan2024KGSimDataset \\
        --v2-root D:/.../Sichuan2024KGSimDataset_v2 \\
        --out-root D:/.../Sichuan2024KGSimDataset_complete

Resulting layout:
    Sichuan2024KGSimDataset_complete/
    ├── README.md
    ├── MANIFEST.json            # file registry with roles + sizes + sha256
    ├── complete_meta.json       # merged V1 meta + V2 generation summary
    ├── load_complete.py         # manifest-driven loader (copied here for portability)
    ├── validate_complete.py     # unified validator
    ├── source_v1/               # V1 essential-data mirror (no eval artefacts)
    ├── observational/           # V2 layer
    ├── intervention_pairs/      # V2 layer
    ├── scenario_sets/           # V2 layer
    ├── anchor_cross_city/       # V2 layer
    ├── safety_cost_labels/      # V2 layer
    ├── metadata/                # V2 metadata + V1 carryforward snapshot
    └── validation/              # V2 validation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

V1_ESSENTIAL_TOP_LEVEL = ["meta.json", "tasks.csv", "task_event_bindings.csv"]
V1_ESSENTIAL_PER_CITY = [
    "users.csv", "events.csv", "profiles_daily.csv",
    "structured_labels.csv", "structured_labels_static.csv",
    "structured_labels_dynamic.csv",
    "task_assessments.csv", "task_assessments_v2.csv",
]
V1_ESSENTIAL_DIRS = ["City-A", "City-B", "City-C"]

V2_LAYERS = ["observational", "intervention_pairs", "scenario_sets",
             "anchor_cross_city", "safety_cost_labels", "metadata", "validation"]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _count_csv_rows(path: Path) -> int | None:
    """Cheap row count for CSV; -1 if unreadable / not CSV."""
    if path.suffix.lower() != ".csv":
        return None
    try:
        n = 0
        with open(path, "rb") as f:
            for line in f:
                n += 1
        return n - 1  # exclude header
    except Exception:
        return None


def _copy_or_skip(src: Path, dst: Path) -> str:
    """Copy file if missing or size differs. Returns 'copied' or 'kept'."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return "kept"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def mirror_v1(v1_root: Path, src_dir: Path) -> dict:
    """Copy V1 essential files into <out>/source_v1/. Returns audit dict."""
    audit = {"files": [], "bytes_total": 0}
    # top-level
    for name in V1_ESSENTIAL_TOP_LEVEL:
        s = v1_root / name
        if not s.exists():
            print(f"  [warn] V1 file missing: {s}")
            continue
        d = src_dir / name
        action = _copy_or_skip(s, d)
        size = s.stat().st_size
        rel = f"source_v1/{name}"
        audit["files"].append({"rel": rel.replace("\\", "/"),
                               "size_bytes": size,
                               "sha256": sha256_of(s),
                               "rows": _count_csv_rows(s),
                               "copy_action": action,
                               "layer": "source_v1/root"})
        audit["bytes_total"] += size
    # per-city
    for city in V1_ESSENTIAL_DIRS:
        cdir = v1_root / city
        if not cdir.exists():
            print(f"  [warn] V1 city dir missing: {cdir}")
            continue
        for name in V1_ESSENTIAL_PER_CITY:
            s = cdir / name
            if not s.exists():
                print(f"  [warn] V1 file missing: {s}")
                continue
            d = src_dir / city / name
            action = _copy_or_skip(s, d)
            size = s.stat().st_size
            rel = f"source_v1/{city}/{name}"
            audit["files"].append({
                "rel": rel.replace("\\", "/"),
                "size_bytes": size,
                "sha256": sha256_of(s),
                "rows": _count_csv_rows(s),
                "copy_action": action,
                "layer": f"source_v1/{city}",
            })
            audit["bytes_total"] += size
    return audit


def mirror_v2(v2_root: Path, out_root: Path) -> dict:
    """Copy all V2 outputs into the unified tree. Returns audit dict."""
    audit = {"files": [], "bytes_total": 0}
    for layer in V2_LAYERS:
        src = v2_root / layer
        if not src.exists():
            print(f"  [warn] V2 layer missing: {src}")
            continue
        for f in sorted(src.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(v2_root)
            dst = out_root / rel
            action = _copy_or_skip(f, dst)
            size = f.stat().st_size
            audit["files"].append({
                "rel": str(rel).replace("\\", "/"),
                "size_bytes": size,
                "sha256": sha256_of(f),
                "rows": _count_csv_rows(f),
                "copy_action": action,
                "layer": f"v2/{rel.parts[0]}",
            })
            audit["bytes_total"] += size
    return audit


def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_complete_meta(v1_root: Path, v2_root: Path) -> dict:
    """Merge V1 meta.json with V2 generation summary into one structure."""
    v1_meta = load_json(v1_root / "meta.json") or {}
    v2_summary = load_json(v2_root / "metadata" / "generation_summary.json") or {}
    v2_config  = load_json(v2_root / "metadata" / "v2_generation_config.json") or {}
    v2_validation = load_json(v2_root / "validation" / "validation_results.json") or {}
    return {
        "complete_version": "1.0",
        "v1_meta_snapshot": v1_meta,
        "v2_generation_summary": v2_summary,
        "v2_generation_config": v2_config,
        "v2_validation_results": v2_validation,
        "narrative": (
            "Dataset v1 (raw observational) + v2 (causal / RL / safety layer) "
            "combined into a single, manifest-driven unified dataset. The V1 "
            "source files are mirrored under source_v1/ so the unified root "
            "is self-contained. The V2 outputs (observational/intervention/"
            "scenario/anchor/safety layers + metadata + validation) are "
            "preserved at the unified root."
        ),
    }


def write_manifest(out_root: Path, v1_audit: dict, v2_audit: dict,
                   complete_meta: dict) -> None:
    manifest = {
        "manifest_version": "1.0",
        "generated_by": "build_complete.py",
        "unified_root": str(out_root),
        "totals": {
            "files":  len(v1_audit["files"]) + len(v2_audit["files"]),
            "bytes":  v1_audit["bytes_total"] + v2_audit["bytes_total"],
            "v1_files": len(v1_audit["files"]),
            "v1_bytes": v1_audit["bytes_total"],
            "v2_files": len(v2_audit["files"]),
            "v2_bytes": v2_audit["bytes_total"],
        },
        "files": v1_audit["files"] + v2_audit["files"],
        "complete_meta_pointer": "complete_meta.json",
    }
    with open(out_root / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(out_root / "complete_meta.json", "w", encoding="utf-8") as f:
        json.dump(complete_meta, f, indent=2, ensure_ascii=False)


def copy_portability_helpers(src_toolkit: Path, out_root: Path) -> None:
    """Drop load_complete.py and validate_complete.py into the unified root
    so the unified dataset is self-contained for distribution."""
    for name in ["load_complete.py", "validate_complete.py"]:
        s = src_toolkit / name
        if s.exists():
            shutil.copy2(s, out_root / name)


def _scan_unified_root(out: Path, helper_rel: list[str]) -> dict:
    audit = {"files": [], "bytes_total": 0}
    for rel in helper_rel:
        p = out / rel
        if not p.exists():
            continue
        size = p.stat().st_size
        audit["files"].append({
            "rel": rel.replace("\\", "/"),
            "size_bytes": size,
            "sha256": sha256_of(p),
            "rows": _count_csv_rows(p),
            "copy_action": "kept",
            "layer": "unified_root",
        })
        audit["bytes_total"] += size
    return audit


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build unified V1+V2 dataset.")
    p.add_argument("--v1-root", required=True, type=str)
    p.add_argument("--v2-root", required=True, type=str)
    p.add_argument("--out-root", required=True, type=str)
    p.add_argument("--toolkit-root", default=str(HERE), type=str,
                   help="Path to v2_toolkit (for load_complete / validate_complete).")
    args = p.parse_args(argv)

    v1 = Path(args.v1_root)
    v2 = Path(args.v2_root)
    out = Path(args.out_root)
    toolkit = Path(args.toolkit_root)

    if not v1.exists():
        raise SystemExit(f"V1 root not found: {v1}")
    if not v2.exists():
        raise SystemExit(f"V2 root not found: {v2}")

    print(f"[init] v1 = {v1}")
    print(f"[init] v2 = {v2}")
    print(f"[init] out = {out}")

    out.mkdir(parents=True, exist_ok=True)
    src_dir = out / "source_v1"
    src_dir.mkdir(exist_ok=True)

    print("\n[step] mirroring V1 essential files into source_v1/")
    v1_audit = mirror_v1(v1, src_dir)
    print(f"  V1 mirror: {len(v1_audit['files'])} files, "
          f"{v1_audit['bytes_total']:,} bytes")

    print("\n[step] mirroring V2 outputs into unified root")
    v2_audit = mirror_v2(v2, out)
    print(f"  V2 mirror: {len(v2_audit['files'])} files, "
          f"{v2_audit['bytes_total']:,} bytes")

    print("\n[step] copying portability helpers (load/validate)")
    copy_portability_helpers(toolkit, out)

    print("\n[step] building complete_meta.json")
    complete_meta = build_complete_meta(v1, v2)
    print(f"  complete_meta keys: {list(complete_meta.keys())}")
    with open(out / "complete_meta.json", "w", encoding="utf-8") as f:
        json.dump(complete_meta, f, indent=2, ensure_ascii=False)

    helper_rel = ["load_complete.py", "validate_complete.py"]
    helper_audit = _scan_unified_root(out, helper_rel)

    print("\n[step] writing MANIFEST.json (data + loaders; bootstrap files excluded)")
    write_manifest(out, v1_audit, v2_audit, complete_meta)
    manifest_path = out / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].extend(helper_audit["files"])
    manifest["totals"]["files"] += len(helper_audit["files"])
    manifest["totals"]["bytes"] += helper_audit["bytes_total"]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    total_files = len(v1_audit["files"]) + len(v2_audit["files"]) + len(helper_audit["files"])
    total_bytes = v1_audit["bytes_total"] + v2_audit["bytes_total"] + helper_audit["bytes_total"]
    print("\n[done] unified dataset at:", out)
    print(f"  total files: {total_files}")
    print(f"  total bytes: {total_bytes:,} ({total_bytes / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())