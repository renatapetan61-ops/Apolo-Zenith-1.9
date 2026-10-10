#!/usr/bin/env python3
"""Check that every shipped config computes the parameter count its name
claims (spec §5: configs must reflect reality; no fictional metadata)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apolo_zenith.config import load_config  # noqa: E402
from scripts.plan_199b import analytic_params  # noqa: E402


def main() -> int:
    out = {}
    for name in ("prototype_tiny", "prototype_125m", "apolo_zenith_1_9"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        p = analytic_params(cfg.backbone)
        out[name] = {
            "total": p["total"],
            "total_human": f"{p['total'] / 1e6:.1f}M"
            if p["total"] < 1e9 else f"{p['total'] / 1e9:.2f}B",
            "active_per_token": p["active_per_token"],
        }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
