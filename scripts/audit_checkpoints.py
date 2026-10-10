#!/usr/bin/env python3
"""Audit every checkpoint under a directory: existence, integrity (sha256),
parameter count, trained flag — spec §14.1 evidence table (IMPLEMENTED)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apolo_zenith.training.checkpoint import checkpoint_report  # noqa: E402


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "checkpoints"
    reports = []
    # recursive search: checkpoints may live in per-config subdirectories
    for meta in sorted(base.rglob("*.json")):
        raw = json.loads(meta.read_text(encoding="utf-8"))
        if "weights_sha256" not in raw:
            continue  # not checkpoint metadata (e.g. tokenizer.json)
        stem = meta.with_suffix("")
        rep = checkpoint_report(stem)
        # extra metadata (trained/steps/final_loss) recorded by training runs
        rep["trained"] = bool(raw.get("trained", False))
        rep["final_loss"] = raw.get("final_loss")
        reports.append(rep)
    if not reports:
        print(json.dumps({"checkpoints_found": 0, "directory": str(base)}, indent=2))
        return 0
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
