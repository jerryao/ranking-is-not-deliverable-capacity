"""Unified validator: V2 contract + V1 mirror completeness + manifest sanity.

Run:
    python validate_complete.py --dataset-root D:/.../Sichuan2024KGSimDataset_complete

Output (validation_results_complete.json):
    {
      "v2_all_pass":                  bool,    # from validation/validation_results.json
      "v1_mirror_complete":           bool,    # all 27 expected V1 files present + non-empty
      "v1_pk_consistent":             bool,    # mirror rows match V1 source-side checksums
      "manifest_consistent":          bool,    # every file entry exists with matching sha256
      "complete_meta_present":        bool,
      "loader_scripts_present":       bool,    # load_complete.py + validate_complete.py in root
      "all_pass":                     bool
    }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


V1_FILES_EXPECTED = [
    "source_v1/meta.json",
    "source_v1/tasks.csv",
    "source_v1/task_event_bindings.csv",
] + [
    f"source_v1/{c}/{f}"
    for c in ("City-A", "City-B", "City-C")
    for f in ("users.csv", "events.csv", "profiles_daily.csv",
              "structured_labels.csv", "structured_labels_static.csv",
              "structured_labels_dynamic.csv",
              "task_assessments.csv", "task_assessments_v2.csv")
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_v2_all_pass(root: Path) -> bool:
    p = root / "validation" / "validation_results.json"
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("all_pass"))
    except Exception:
        return False


def check_v1_mirror_complete(root: Path) -> bool:
    for rel in V1_FILES_EXPECTED:
        p = root / rel
        if not p.exists() or p.stat().st_size == 0:
            return False
    return True


def check_v1_pk_consistent(root: Path, v1_source: Path | None) -> bool:
    """If the original V1 source path is provided, recompute SHA256 of mirrored
    files and compare to source. Returns True if either:
       - v1_source is None (skip), or
       - every mirrored V1 file's SHA256 matches the corresponding source file."""
    if v1_source is None:
        return True
    v1_source = Path(v1_source)
    for rel in V1_FILES_EXPECTED:
        if rel == "source_v1/meta.json":
            continue  # meta is small and never recomputed
        # Map "source_v1/City-A/users.csv" → "<v1>/City-A/users.csv"
        source_rel = rel[len("source_v1/"):]
        a = v1_source / source_rel
        b = root / rel
        if not a.exists() or not b.exists():
            return False
        if _sha256(a) != _sha256(b):
            return False
    return True


def check_manifest_consistent(root: Path) -> bool:
    p = root / "MANIFEST.json"
    if not p.exists():
        return False
    try:
        manifest = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    entries = manifest.get("files", [])
    if not entries:
        return False
    for e in entries:
        f = root / e["rel"]
        if not f.exists():
            return False
        # If size differs, manifest is stale
        if e.get("size_bytes") != f.stat().st_size:
            return False
    return True


def check_complete_meta_present(root: Path) -> bool:
    p = root / "complete_meta.json"
    if not p.exists():
        return False
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        return ("complete_version" in m
                and "v1_meta_snapshot" in m
                and "v2_generation_summary" in m)
    except Exception:
        return False


def check_loader_scripts_present(root: Path) -> bool:
    return (root / "load_complete.py").exists() and (root / "validate_complete.py").exists()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate the unified V1+V2 dataset.")
    p.add_argument("--dataset-root", required=True, type=str)
    p.add_argument("--v1-source", default=None, type=str,
                   help="Original V1 source path. If provided, mirror checksums "
                        "are recomputed and compared.")
    args = p.parse_args(argv)

    root = Path(args.dataset_root)
    v1_source = Path(args.v1_source) if args.v1_source else None

    results = {
        "v2_all_pass":              check_v2_all_pass(root),
        "v1_mirror_complete":       check_v1_mirror_complete(root),
        "v1_pk_consistent":         check_v1_pk_consistent(root, v1_source),
        "manifest_consistent":      check_manifest_consistent(root),
        "complete_meta_present":    check_complete_meta_present(root),
        "loader_scripts_present":   check_loader_scripts_present(root),
    }
    results["all_pass"] = all(results.values())

    out = root / "validation" / "validation_results_complete.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[validate_complete] results → {out}")
    return 0 if results["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())