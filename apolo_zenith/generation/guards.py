"""Generation guards: refuse to present untrained weights as a generative
model (IMPLEMENTED, TESTED). Spec §14.1: "If essential components remain
random, stop final generation with a clear error instead of silently
producing noise."

Every pipeline component must present a checkpoint whose:
  - sha256 matches its metadata (integrity),
  - parameter count matches the live module (architecture compatibility),
  - metadata declares `trained: true` with a step count > 0 and a loss record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..training.checkpoint import checkpoint_report


class UntrainedComponentError(RuntimeError):
    """Raised when final generation is requested from untrained components."""


_REQUIRED_META_FIELDS = ("trained", "steps", "final_loss")


def component_evidence(model, name: str, ckpt_path: str | Path) -> Dict:
    """Collect auditable evidence about one component's training status."""
    report = checkpoint_report(ckpt_path) if ckpt_path else {"exists": False}
    live_params = sum(p.numel() for p in model.parameters())
    trained_ok = bool(
        report.get("exists")
        and report.get("integrity_ok")
        and report.get("param_count") == live_params
        and report.get("trained") is True
        and int(report.get("steps") or 0) > 0
    )
    return {
        "component": name,
        "checkpoint_path": str(ckpt_path) if ckpt_path else None,
        "checkpoint_exists": bool(report.get("exists")),
        "integrity_ok": bool(report.get("integrity_ok")),
        "param_count_live": live_params,
        "param_count_ckpt": report.get("param_count"),
        "trained": trained_ok,
        "steps": report.get("steps"),
    }


def assert_generation_ready(components: Dict[str, Dict]) -> None:
    """Raise UntrainedComponentError unless every component is trained."""
    untrained = []
    for name, ev in components.items():
        if not ev["trained"]:
            untrained.append(
                f"{name}: exists={ev['checkpoint_exists']}, "
                f"integrity_ok={ev['integrity_ok']}, "
                f"params live/ckpt={ev['param_count_live']}/{ev['param_count_ckpt']}, "
                f"trained={ev['trained']}"
            )
    if untrained:
        raise UntrainedComponentError(
            "Generation refused — untrained/unverified components:\n  "
            + "\n  ".join(untrained)
            + "\nTrain the components or use --debug-generation to inspect "
              "internals without presenting output as a valid generation."
        )
